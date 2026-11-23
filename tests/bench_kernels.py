import torch
import triton
import triton.testing
import pandas as pd

from cautious_chainsaw.matmul import int4_matmul
from cautious_chainsaw.packing import pack_int4


def benchmark(fn, warmup=100, rep=500):
    return triton.testing.do_bench(
        fn,
        warmup=warmup,
        rep=rep
    )


def run_benchmark(
    M,
    N,
    K,
    group_size=128,
    warmup=100,
    rep=500,
):
    device = "cuda"

    assert torch.cuda.is_available()

    # FP16 activations and weights
    X = torch.randn(
        M, K,
        device=device,
        dtype=torch.float16
    )

    W = torch.randn(
        N, K,
        device=device,
        dtype=torch.float16
    )

    # Native FP16 reference
    def native_fp16():
        return X @ W.T

    # Quantization and packing
    Q, S, K_original, K_PAD = quantize_int4(
        W, group_size
    )

    P = pack_int4(Q)

    # Your fused INT4 kernel
    def fused_int4():
        return int4_matmul(
            X, P, S,
            K_original,
            K_PAD,
            group_size=group_size
        )

    # Pre-dequantize weights
    W_dequant = (
        Q.float()
        * S.repeat_interleave(
            group_size, dim=1
        ).float()
    )[:, :K].half().contiguous()

    def predequant_fp16():
        return X @ W_dequant.T

    # Correctness
    Y_ref = native_fp16()
    Y_int4 = fused_int4()
    Y_dequant = predequant_fp16()

    def error(Y):
        diff = (Y.float() - Y_ref.float()).abs()
        return {
            "max_abs_error": diff.max().item(),
            "mean_abs_error": diff.mean().item(),
            "relative_error": (
                diff.norm() / Y_ref.float().norm()
            ).item()
        }

    # Benchmark
    t_fp16 = benchmark(
        native_fp16, warmup, rep
    )

    t_fused = benchmark(
        fused_int4, warmup, rep
    )

    t_dequant = benchmark(
        predequant_fp16, warmup, rep
    )

    # Amortized pre-dequantization cost
    def dequant_plus_gemm():
        Wd = (
            Q.float()
            * S.repeat_interleave(
                group_size, dim=1
            ).float()
        )[:, :K].half().contiguous()

        return X @ Wd.T

    t_total = benchmark(
        dequant_plus_gemm, warmup, rep
    )

    results = {
        "M": M,
        "N": N,
        "K": K,
        "FP16_ms": t_fp16,
        "INT4_fused_ms": t_fused,
        "Predequant_GEMM_ms": t_dequant,
        "Dequant_plus_GEMM_ms": t_total,
        "INT4_vs_FP16": t_fused / t_fp16,
        "Predequant_vs_FP16": t_dequant / t_fp16,
        **{
            "INT4_" + k: v
            for k, v in error(Y_int4).items()
        }
    }

    return results


if __name__ == "__main__":

    print("GPU:", torch.cuda.get_device_name())
    print("CUDA:", torch.version.cuda)
    print("PyTorch:", torch.__version__)
    print("Triton:", triton.__version__)

    # Typical transformer GEMM configurations
    shapes = [
        (1, 4096, 4096),
        (16, 4096, 4096),
        (32, 4096, 4096),
        (64, 4096, 4096),
        (128, 4096, 4096),
        (256, 4096, 4096),
        (512, 4096, 4096),
        (128, 11008, 4096),
        (128, 4096, 11008),
    ]

    results = []

    for M, N, K in shapes:
        print(f"\nBenchmarking M={M}, N={N}, K={K}")

        result = run_benchmark(M, N, K)

        results.append(result)

        print(
            f"FP16: {result['FP16_ms']:.4f} ms | "
            f"INT4: {result['INT4_fused_ms']:.4f} ms | "
            f"Ratio: {result['INT4_vs_FP16']:.2f}x"
        )

    df = pd.DataFrame(results)

    print("\n\nFULL RESULTS")
    print(df.to_string(index=False))

    df.to_csv("t4_int4_benchmark.csv", index=False)

    print("\nSaved: t4_int4_benchmark.csv")
