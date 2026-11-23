import torch
import triton
import triton.language as tl
import triton.testing

DEFAULT_GROUP_SIZE = 128
DEFAULT_BLOCK_K = 128
DEFAULT_NUM_WARPS = 4


MATMUL_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 64},
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64},
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 64},
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64},
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 128},
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 128},
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64},
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 128},
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64},
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64},
        num_warps=8,
    ),
]


@triton.autotune(
    configs=MATMUL_CONFIGS,
    key=["M", "N", "K"],
)
@triton.jit
def int4_matmul_kernel(
        X,
        P,
        S,
        B,
        Y,

        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        K_PAD: tl.constexpr,

        GROUP_SIZE: tl.constexpr,

        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,

        HAS_BIAS: tl.constexpr,
        USE_NATIVE_BF16: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = (
            pid_m * BLOCK_M
            + tl.arange(0, BLOCK_M)
    )

    offs_n = (
            pid_n * BLOCK_N
            + tl.arange(0, BLOCK_N)
    )

    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros(
        (BLOCK_M, BLOCK_N),
        dtype=tl.float32,
    )

    num_groups = K_PAD // GROUP_SIZE

    for k_start in range(tl.cdiv(K_PAD, BLOCK_K)):

        k = k_start * BLOCK_K + offs_k

        x = tl.load(
            X
            + offs_m[:, None] * K
            + k[None, :],

            mask=(
                    (offs_m[:, None] < M)
                    & (k[None, :] < K)
            ),

            other=0.0,
        )

        packed_idx = (
                offs_n[:, None] * (K_PAD // 2)
                + k[None, :] // 2
        )

        weight_mask = (
                (offs_n[:, None] < N)
                & (k[None, :] < K_PAD)
        )

        packed = tl.load(
            P + packed_idx,
            mask=weight_mask,
            other=0,
        ).to(tl.int32)

        low = packed & 0x0F
        high = (packed >> 4) & 0x0F

        q = tl.where(
            k[None, :] % 2 == 0,
            low,
            high,
        )

        q = tl.where(
            q >= 8,
            q - 16,
            q,
        )

        group_idx = k // GROUP_SIZE

        scale_idx = (
                offs_n[:, None] * num_groups
                + group_idx[None, :]
        )

        scale = tl.load(
            S + scale_idx,
            mask=weight_mask,
            other=0.0,
        )

        dequantized = (
                q.to(tl.float32)
                * scale.to(tl.float32)
        )

        if USE_NATIVE_BF16:
            x_operand = x.to(tl.bfloat16)
            w_operand = dequantized.to(tl.bfloat16)
        else:
            # T4 uses FP16 Tensor Core operations.
            # BF16 input values are converted to FP16.
            x_operand = x.to(tl.float16)
            w_operand = dequantized.to(tl.float16)

        acc += tl.dot(
            x_operand,
            tl.trans(w_operand),
        )

    if HAS_BIAS:
        bias = tl.load(
            B + offs_n,
            mask=offs_n < N,
            other=0.0,
        ).to(tl.float32)

        acc += bias[None, :]

    tl.store(
        Y + offs_m[:, None] * N + offs_n[None, :],

        acc.to(Y.dtype.element_ty),

        mask=(
                (offs_m[:, None] < M)
                & (offs_n[None, :] < N)
        ),
    )


def int4_matmul(
        x: torch.Tensor,
        packed_weights: torch.Tensor,
        scales: torch.Tensor,
        k: int,
        k_pad: int,
        group_size: int = DEFAULT_GROUP_SIZE,
        bias: torch.Tensor = None,
        output: torch.Tensor = None,
        block_k: int = DEFAULT_BLOCK_K,
):

    if x.ndim != 2:
        raise ValueError("x must have shape [M, K]")

    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("x must be FP16 or BF16")

    if not x.is_cuda:
        raise ValueError("x must be on CUDA")

    if not x.is_contiguous():
        raise ValueError("x must be contiguous")

    if packed_weights.dtype != torch.uint8:
        raise TypeError("packed_weights must be uint8")

    if not packed_weights.is_cuda:
        raise ValueError("packed_weights must be on CUDA")

    if scales.dtype != torch.float16:
        raise TypeError("scales must be FP16")

    if not scales.is_cuda:
        raise ValueError("scales must be on CUDA")

    if x.device != packed_weights.device or x.device != scales.device:
        raise ValueError("all tensors must be on the same CUDA device")

    if group_size <= 0 or (group_size & (group_size - 1)) != 0:
        raise ValueError("group_size must be a positive power of two")

    if block_k % 2:
        raise ValueError("block_k must be even")

    m, actual_k = x.shape
    n = packed_weights.shape[0]

    if actual_k != k:
        raise ValueError("activation K dimension mismatch")

    if k_pad < k or k_pad % group_size or k_pad % 2:
        raise ValueError("invalid K_PAD")

    if packed_weights.shape != (n, k_pad // 2):
        raise ValueError("invalid packed weight shape")

    if scales.shape != (n, k_pad // group_size):
        raise ValueError("invalid scale shape")

    if not packed_weights.is_contiguous() or not scales.is_contiguous():
        raise ValueError("weights and scales must be contiguous")

    if bias is not None:
        if bias.shape != (n,):
            raise ValueError("bias must have shape [N]")

        if bias.device != x.device:
            raise ValueError("bias must be on the same device")

        bias = bias.contiguous()

    if output is None:
        output = torch.empty(
            (m, n),
            device=x.device,
            dtype=x.dtype,
        )
    else:
        if output.shape != (m, n):
            raise ValueError("output must have shape [M, N]")

        if output.device != x.device or output.dtype != x.dtype:
            raise ValueError("output device and dtype must match x")

        if not output.is_contiguous():
            raise ValueError("output must be contiguous")

    capability = torch.cuda.get_device_capability(x.device)

    use_native_bf16 = (
            x.dtype == torch.bfloat16
            and capability >= (8, 0)
            and torch.cuda.is_bf16_supported()
    )

    # The bias pointer is unused when HAS_BIAS=False.
    bias_arg = bias if bias is not None else packed_weights

    grid = lambda META: (
        triton.cdiv(m, META["BLOCK_M"]),
        triton.cdiv(n, META["BLOCK_N"]),
    )

    int4_matmul_kernel[grid](
        x,
        packed_weights,
        scales,
        bias_arg,
        output,
        m,
        n,
        k,
        k_pad,
        group_size,
        bias is not None,
        use_native_bf16,

        num_warps=DEFAULT_NUM_WARPS,
        num_stages=2,
    )

    return output
