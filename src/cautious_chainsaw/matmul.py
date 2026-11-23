import torch
import triton
import triton.language as tl
import triton.testing


@triton.jit
def int4_matmul_kernel(
        X,
        P,
        S,
        Y,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        K_PAD: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    num_groups = K_PAD // GROUP_SIZE

    for k_start in range(tl.cdiv(K_PAD, BLOCK_K)):
        k = k_start * BLOCK_K + offs_k

        # Activations are only defined over the original, unpadded K.
        x = tl.load(
            X + offs_m[:, None] * K + k[None, :],
            (offs_m[:, None] < M) & (k[None, :] < K),
            other=0,
        )

        packed_idx = (
                offs_n[:, None] * (K_PAD // 2) + k[None, :] // 2
        )
        weight_mask = (
                (offs_n[:, None] < N) & (k[None, :] < K_PAD)
        )
        packed = tl.load(
            P + packed_idx, weight_mask, other=0
        ).to(tl.int32)

        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        q = tl.where(k[None, :] % 2 == 0, low, high)
        q = tl.where(q >= 8, q - 16, q)

        group_idx = k // GROUP_SIZE
        scale_idx = (
                offs_n[:, None] * num_groups + group_idx[None, :]
        )
        scale = tl.load(
            S + scale_idx, weight_mask, other=0
        )

        dequantized = q.to(tl.float32) * scale.to(tl.float32)

        # FP16 Tensor Core operands with FP32 accumulation.
        acc += tl.dot(
            x.to(tl.float16),
            tl.trans(dequantized.to(tl.float16)),
        )

    tl.store(
        Y + offs_m[:, None] * N + offs_n[None, :],
        acc.to(Y.dtype.element_ty),
        (offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def int4_matmul(
        x: torch.Tensor,
        packed_weights: torch.Tensor,
        scales: torch.Tensor,
        k: int,
        k_pad: int,
        group_size: int = 128,
        block_m: int = 32,
        block_n: int = 64,
        block_k: int = 64,
):
    """Multiply FP16 activations by packed, group-scaled INT4 weights."""
    if x.ndim != 2 or x.dtype != torch.float16:
        raise ValueError("x must be a 2D FP16 tensor")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")

    m, actual_k = x.shape
    n = packed_weights.shape[0]

    if actual_k != k:
        raise ValueError(f"x has K={actual_k}, expected K={k}")
    if k_pad < k or k_pad % group_size or k_pad % 2:
        raise ValueError("invalid padded K dimension")
    if packed_weights.shape != (n, k_pad // 2):
        raise ValueError("packed weight shape does not match K_PAD")
    if scales.shape != (n, k_pad // group_size):
        raise ValueError("scale shape does not match K_PAD/group_size")
    if not packed_weights.is_contiguous() or not scales.is_contiguous():
        raise ValueError("packed weights and scales must be contiguous")

    output = torch.empty((m, n), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))

    int4_matmul_kernel[grid](
        x,
        packed_weights,
        scales,
        output,
        m,
        n,
        k,
        k_pad,
        group_size,
        block_m,
        block_n,
        block_k,
        num_warps=4,
    )
    return output
