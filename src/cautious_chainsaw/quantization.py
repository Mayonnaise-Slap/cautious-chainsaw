import torch
import triton
import triton as tr
import triton.language as tl

from matmul import DEFAULT_GROUP_SIZE


@triton.jit
def quantize_int4_kernel(
    W,
    Q,
    S,
    K: tl.constexpr,
    K_PAD: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)

    cols = group * GROUP_SIZE + tl.arange(0, GROUP_SIZE)

    weights = tl.load(
        W + row * K + cols,
        mask=cols < K,
        other=0.0,
    ).to(tl.float32)

    max_abs = tl.max(tl.abs(weights), axis=0)

    scale = tl.where(
        max_abs > 0.0,
        max_abs / 7.0,
        1.0,
    )

    normalized = weights / scale

    rounded = tl.where(
        normalized >= 0.0,
        tl.floor(normalized + 0.5),
        tl.ceil(normalized - 0.5),
    )

    quantized = tl.minimum(
        tl.maximum(rounded, -8.0),
        7.0,
    ).to(tl.int8)

    tl.store(
        Q + row * K_PAD + cols,
        quantized,
    )

    tl.store(
        S + row * (K_PAD // GROUP_SIZE) + group,
        scale.to(tl.float16),
    )


def quantize_int4(
    weights: torch.Tensor,
    group_size: int = DEFAULT_GROUP_SIZE,
):
    """
    Quantize a 2D weight matrix [N, K] into signed INT4 values.

    Returns:
        q:      [N, K_PAD], int8 temporary quantized values
        scales: [N, K_PAD // group_size], float16
        k:      original input dimension
        k_pad:  padded input dimension
    """

    if weights.ndim != 2:
        raise ValueError("weights must have shape [N, K]")

    if weights.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("weights must be FP16 or BF16")

    if not weights.is_cuda:
        raise ValueError("weights must be on CUDA")

    if group_size <= 0 or (group_size & (group_size - 1)) != 0:
        raise ValueError("group_size must be a positive power of two")

    weights = weights.contiguous()

    n, k = weights.shape
    k_pad = triton.cdiv(k, group_size) * group_size

    q = torch.empty(
        (n, k_pad),
        device=weights.device,
        dtype=torch.int8,
    )

    scales = torch.empty(
        (n, k_pad // group_size),
        device=weights.device,
        dtype=torch.float16,
    )

    grid = (
        n,
        k_pad // group_size,
    )

    quantize_int4_kernel[grid](
        weights,
        q,
        scales,
        k,
        k_pad,
        group_size,
        num_warps=4,
    )

    return q, scales, k, k_pad
