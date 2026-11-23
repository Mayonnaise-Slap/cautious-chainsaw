@tr.jit
def int4_matmul_kernel(
    X,
    P,
    S,
    Y,

    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,

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

    num_groups = K // GROUP_SIZE

    for k_start in range(tl.cdiv(K, BLOCK_K)):

        k = k_start * BLOCK_K + offs_k

        x = tl.load(
            X + offs_m[:, None] * K + k[None, :],
            (offs_m[:, None] < M) &
            (k[None, :] < K),
            other=0
        )
        packed_idx = (
            offs_n[:, None] * (K // 2)
            + (k[None, :] // 2)
        )

        packed = tl.load(
            P + packed_idx,
            (offs_n[:, None] < N) &
            (k[None, :] < K),
            other=0
        ).to(tl.int32)

        low = packed & 0x0F
        high = (packed >> 4) & 0x0F

        q = tl.where(
            (k[None, :] % 2) == 0,
            low,
            high
        )

        q = tl.where(q >= 8, q - 16, q)

        group_idx = k // GROUP_SIZE

        scale_idx = (
            offs_n[:, None] * num_groups
            + group_idx[None, :]
        )

        scale = tl.load(
            S + scale_idx,
            (offs_n[:, None] < N) &
            (k[None, :] < K),
            other=0
        )

        w = q.to(tl.float32) * scale.to(tl.float32)

        acc += tl.dot(
            x,
            tl.trans(w.to(tl.bfloat16))
        )

    tl.store(
        Y + offs_m[:, None] * N + offs_n[None, :],
        acc.to(tl.bfloat16),
        (offs_m[:, None] < M) &
        (offs_n[None, :] < N)
    )


def int4_matmul(
    X,
    P,
    S,
    group_size=128,
    block_m=32,
    block_n=64,
    block_k=64,
):
    assert X.dtype == torch.bfloat16
    assert P.dtype == torch.uint8

    M, K = X.shape
    N = P.shape[0]

    assert K % 2 == 0
    assert K % group_size == 0
    assert P.shape == (N, K // 2)
    assert S.shape == (N, K // group_size)

    assert X.is_contiguous()
    assert P.is_contiguous()
    assert S.is_contiguous()

    Y = torch.empty(
        (M, N),
        device=X.device,
        dtype=torch.bfloat16
    )

    grid = (
        tr.cdiv(M, block_m),
        tr.cdiv(N, block_n)
    )

    int4_matmul_kernel[grid](
        X, P, S, Y,
        M, N, K,
        group_size,
        block_m, block_n, block_k,
        num_warps=4
    )

    return Y
