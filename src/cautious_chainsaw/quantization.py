import torch
import triton
import triton as tr
import triton.language as tl


def quantize_int4(weights: torch.Tensor, group_size: int = 128):
    if weights.ndim != 2:
        raise ValueError("weights must have shape [N, K]")
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    n, k = weights.shape
    k_pad = triton.cdiv(k, group_size) * group_size

    padded = torch.nn.functional.pad(
        weights.float(), (0, k_pad - k)
    )
    grouped = padded.reshape(n, k_pad // group_size, group_size)

    scales = (grouped.abs().amax(dim=-1) / 7.0).clamp(min=1e-8)
    q = (
        torch.round(grouped / scales.unsqueeze(-1))
        .clamp(-8, 7)
        .to(torch.int8)
        .reshape(n, k_pad)
        .contiguous()
    )

    return q, scales.to(torch.float16).contiguous(), k, k_pad


@triton.jit
def dequant_int4_kernel(
        P, S, W,
        N: tl.constexpr,
        K: tl.constexpr,
        K_PAD: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    pid = tl.program_id(1)

    k = pid * BLOCK + tl.arange(0, BLOCK)

    mask = k < K_PAD

    packed = tl.load(
        P + row * (K_PAD // 2) + k // 2,
        mask,
        other=0
    ).to(tl.int32)

    q = tl.where(
        k % 2 == 0,
        packed & 0x0F,
        (packed >> 4) & 0x0F
    )

    q = tl.where(q >= 8, q - 16, q)

    group = k // GROUP_SIZE

    scale = tl.load(
        S + row * (K_PAD // GROUP_SIZE) + group,
        mask,
        other=0
    )

    w = q.to(tl.float32) * scale.to(tl.float32)

    tl.store(
        W + row * K_PAD + k,
        w.to(tl.float16),
        mask
    )


def dequant_int4(P, S, N, K, K_PAD, group_size=128):
    W = torch.empty(
        (N, K_PAD),
        device=P.device,
        dtype=torch.float16
    )

    grid = (
        N,
        triton.cdiv(K_PAD, 512)
    )

    dequant_int4_kernel[grid](
        P, S, W,
        N, K, K_PAD,
        group_size,
        512
    )

    return W[:, :K].contiguous()
