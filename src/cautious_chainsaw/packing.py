import torch
import triton as tr
import triton.language as tl


@tr.jit
def pack_int4_kernel(
    Q,
    P,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    pid = tl.program_id(1)

    pairs = pid * BLOCK + tl.arange(0, BLOCK)

    idx = row * K + 2 * pairs

    q0 = tl.load(Q + idx, pairs < K // 2, other=0)
    q1 = tl.load(Q + idx + 1, pairs < K // 2, other=0)

    lo = q0.to(tl.int32) & 0x0F
    hi = q1.to(tl.int32) & 0x0F

    packed = lo | (hi << 4)

    out_idx = row * (K // 2) + pairs

    tl.store(
        P + out_idx,
        packed.to(tl.uint8),
        pairs < K // 2
    )


def pack_int4(Q, block_size=1024):
    assert Q.ndim == 2
    assert Q.dtype == torch.int8
    assert Q.is_contiguous()

    N, K = Q.shape
    assert K % 2 == 0

    P = torch.empty(
        (N, K // 2),
        device=Q.device,
        dtype=torch.uint8
    )

    grid = (
        N,
        tr.cdiv(K // 2, block_size)
    )

    pack_int4_kernel[grid](
        Q,
        P,
        K,
        block_size
    )

    return P
