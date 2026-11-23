import torch
import triton
import triton as tr
import triton.language as tl


@triton.jit
def pack_int4_kernel(
    Q,
    P,
    K_PAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)

    pairs = (
        tl.program_id(1) * BLOCK
        + tl.arange(0, BLOCK)
    )

    mask = pairs < K_PAD // 2

    idx = row * K_PAD + 2 * pairs

    q0 = tl.load(
        Q + idx,
        mask=mask,
        other=0,
    ).to(tl.int32)

    q1 = tl.load(
        Q + idx + 1,
        mask=mask,
        other=0,
    ).to(tl.int32)

    packed = (
        (q0 & 0x0F)
        | ((q1 & 0x0F) << 4)
    )

    out_idx = row * (K_PAD // 2) + pairs

    tl.store(
        P + out_idx,
        packed.to(tl.uint8),
        mask=mask,
    )


def pack_int4(
    q: torch.Tensor,
    block_size: int = 1024,
):
    if q.ndim != 2 or q.dtype != torch.int8:
        raise ValueError("q must be a 2D int8 tensor")

    if not q.is_cuda or not q.is_contiguous():
        raise ValueError("q must be contiguous and on CUDA")

    n, k_pad = q.shape

    if k_pad % 2:
        raise ValueError("K_PAD must be even")

    packed = torch.empty(
        (n, k_pad // 2),
        device=q.device,
        dtype=torch.uint8,
    )

    grid = (
        n,
        triton.cdiv(k_pad // 2, block_size),
    )

    pack_int4_kernel[grid](
        q,
        packed,
        k_pad,
        block_size,
        num_warps=4,
    )

    return packed