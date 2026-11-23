import triton
import triton.language as tr
import torch

@tr.jit
def int4_matmul_kernel(
    X, P, S, Y,

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

    acc = tl.zeros(
        (BLOCK_M, BLOCK_N),
        dtype=tl.float32
    )

    num_groups = K_PAD // GROUP_SIZE

    for k_start in range(tl.cdiv(K_PAD, BLOCK_K)):

        k = k_start * BLOCK_K + offs_k

        # Load activations.
        # Padded positions are zero.
        x = tl.load(
            X + offs_m[:, None] * K + k[None, :],
            (offs_m[:, None] < M) &
            (k[None, :] < K),
            other=0
        )

        # Load packed weights.
        packed_idx = (
            offs_n[:, None] * (K_PAD // 2)
            + k[None, :] // 2
        )

        w_mask = (
            (offs_n[:, None] < N) &
            (k[None, :] < K_PAD)
        )

        packed = tl.load(
            P + packed_idx,
            w_mask,
            other=0
        ).to(tl.int32)

        # Unpack INT4.
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F

        q = tl.where(
            k[None, :] % 2 == 0,
            low,
            high
        )

        # Two's complement conversion.
        q = tl.where(q >= 8, q - 16, q)

        # Load scales.
        group_idx = k // GROUP_SIZE

        scale_idx = (
            offs_n[:, None] * num_groups
            + group_idx[None, :]
        )

        scale = tl.load(
            S + scale_idx,
            w_mask,
            other=0
        )

        # Dequantize.
        w = q.to(tl.float32) * scale.to(tl.float32)

        # T4: use FP16 Tensor Core operands.
        acc += tl.dot(
            x.to(tl.float16),
            tl.trans(w.to(tl.float16))
        )

    tl.store(
        Y + offs_m[:, None] * N + offs_n[None, :],
        acc.to(Y.dtype.element_ty),
        (offs_m[:, None] < M) &
        (offs_n[None, :] < N)
    )


def int4_matmul(
    X,
    P,
    S,
    K,
    K_PAD,
    group_size=128,
    block_m=32,
    block_n=64,
    block_k=64,
):
    M, K_actual = X.shape
    N = P.shape[0]

    assert K_actual == K
    assert K_PAD % group_size == 0
    assert K_PAD % 2 == 0
    assert P.shape == (N, K_PAD // 2)
    assert S.shape == (N, K_PAD // group_size)

    assert X.is_contiguous()
    assert P.is_contiguous()
    assert S.is_contiguous()

    Y = torch.empty(
        (M, N),
        device=X.device,
        dtype=X.dtype
    )

    grid = (
        tr.cdiv(M, block_m),
        tr.cdiv(N, block_n)
    )

    int4_matmul_kernel[grid](
        X, P, S, Y,
        M, N, K, K_PAD,
        group_size,
        block_m, block_n, block_k,
        num_warps=4
    )

    return Y
