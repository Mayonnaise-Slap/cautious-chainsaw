"""BF16 matrix and causal language model benchmarks."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _cuda_device(device: str, dtype: str = "bf16") -> Any:
    import torch

    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Benchmarking requires an available CUDA GPU.")
    result = torch.device(device)
    if dtype == "bf16" and torch.cuda.get_device_capability(result) < (8, 0):
        raise RuntimeError("BF16 benchmarking requires a GPU with native BF16 support.")
    if dtype not in {"bf16", "fp16"}:
        raise ValueError("dtype must be either 'bf16' or 'fp16'.")
    return result


def _write_csv(records: list[dict[str, Any]], output: str | Path | None) -> None:
    if output is None:
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = list(dict.fromkeys(key for record in records for key in record))
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _event_ms(fn: Any, warmup: int, repeats: int, device: Any) -> float:
    import torch

    for _ in range(warmup):
        with torch.inference_mode():
            fn()
    torch.cuda.synchronize(device)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    stream = torch.cuda.current_stream(device)
    for start, end in zip(starts, ends):
        start.record(stream)
        with torch.inference_mode():
            fn()
        end.record(stream)
    torch.cuda.synchronize(device)
    return sum(start.elapsed_time(end) for start, end in zip(starts, ends)) / repeats


def _event_ms_with_setup(setup: Any, fn: Any, warmup: int, repeats: int, device: Any) -> float:
    """Time fn while creating fresh state with setup outside each timed region."""
    import torch

    for _ in range(warmup):
        with torch.inference_mode():
            state = setup()
            fn(state)
    torch.cuda.synchronize(device)
    elapsed = 0.0
    for _ in range(repeats):
        with torch.inference_mode():
            state = setup()
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream(device)
        start.record(stream)
        with torch.inference_mode():
            fn(state)
        end.record(stream)
        torch.cuda.synchronize(device)
        elapsed += start.elapsed_time(end)
    return elapsed / repeats


def benchmark_matmul(
    backend: str = "bf16",
    m_sizes: Sequence[int] = (128, 512, 2048),
    weight_shapes: Sequence[tuple[int, int]] = (
        (2048, 2048),
        (512, 2048),
        (8192, 2048),
        (2048, 8192),
    ),
    device: str = "cuda",
    warmup: int = 10,
    repeats: int = 100,
    output: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Measure ``X[M,K] @ W[N,K].T`` with BF16 tensors on CUDA."""
    import torch

    if backend not in {"bf16", "fp16"}:
        raise ValueError("backend must be either 'bf16' or 'fp16'.")
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be >= 0 and repeats must be >= 1.")
    dev = _cuda_device(device, backend)
    dtype = torch.bfloat16 if backend == "bf16" else torch.float16
    gpu = torch.cuda.get_device_name(dev)
    records: list[dict[str, Any]] = []
    for m in m_sizes:
        if m <= 0:
            raise ValueError("Matrix dimensions must be positive.")
        for n, k in weight_shapes:
            if n <= 0 or k <= 0:
                raise ValueError("Matrix dimensions must be positive.")
            x = torch.randn((m, k), device=dev, dtype=dtype)
            w = torch.randn((n, k), device=dev, dtype=dtype)
            y = x @ w.T
            latency = _event_ms(lambda: torch.matmul(x, w.T), warmup, repeats, dev)
            records.append(
                {
                    "benchmark": "matmul",
                    "backend": backend,
                    "m": m,
                    "n": n,
                    "k": k,
                    "latency_ms": latency,
                    "tflops": (2 * m * n * k) / (latency * 1e9),
                    "dtype": str(dtype).removeprefix("torch."),
                    "device": str(dev),
                    "gpu": gpu,
                    "warmup": warmup,
                    "repeats": repeats,
                    "output_shape": str(tuple(y.shape)),
                }
            )
    _write_csv(records, output)
    return records


def benchmark_model(
    model: Any,
    tokenizer: Any,
    prefill_lengths: Sequence[int] = (128, 512, 2048),
    decode_tokens: int = 32,
    device: str = "cuda",
    warmup: int = 2,
    repeats: int = 5,
    output: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Measure prefill and cached autoregressive decode independently."""
    import torch

    if warmup < 0 or repeats < 1 or decode_tokens < 1:
        raise ValueError("warmup >= 0, repeats >= 1, and decode_tokens >= 1 are required.")
    model_dtype = next(model.parameters()).dtype
    dtype_name = (
        "bf16"
        if model_dtype == torch.bfloat16
        else "fp16"
        if model_dtype == torch.float16
        else "other"
    )
    dev = _cuda_device(device, "bf16" if dtype_name == "bf16" else "fp16")
    model.eval()
    records: list[dict[str, Any]] = []
    gpu = torch.cuda.get_device_name(dev)

    try:
        encoded = tokenizer.encode(
            "A short language modeling benchmark prompt.", add_special_tokens=False
        )
    except AttributeError as exc:
        raise TypeError("tokenizer must provide encode(text, add_special_tokens=False).") from exc
    if not encoded:
        raise ValueError("Tokenizer returned no tokens for the benchmark prompt.")

    for length in prefill_lengths:
        if length < 1:
            raise ValueError("Prefill lengths must be positive.")
        repeated = (encoded * ((length + len(encoded) - 1) // len(encoded)))[:length]
        ids = torch.tensor([repeated], device=dev, dtype=torch.long)
        mask = torch.ones_like(ids)

        def prefill() -> Any:
            return model(input_ids=ids, attention_mask=mask, use_cache=True)

        prefill_ms = _event_ms(prefill, warmup, repeats, dev)
        records.append(
            {
                "benchmark": "model",
                "phase": "prefill",
                "prompt_tokens": length,
                "tokens": length,
                "latency_ms": prefill_ms,
                "tokens_per_second": length * 1000 / prefill_ms,
                "device": str(dev),
                "gpu": gpu,
                "dtype": str(model_dtype),
                "warmup": warmup,
                "repeats": repeats,
            }
        )

        # Prepare a fresh prompt prefill outside each timed decode region.
        seed_ids = ids
        initial_mask = mask

        def setup_decode() -> tuple[Any, Any]:
            prefill_result = model(input_ids=seed_ids, attention_mask=initial_mask, use_cache=True)
            return prefill_result.past_key_values, prefill_result.logits[:, -1:].argmax(dim=-1)

        def decode(state: tuple[Any, Any]) -> None:
            past, token = state
            for index in range(decode_tokens):
                full_mask = torch.ones(
                    (1, seed_ids.shape[1] + index + 1), device=dev, dtype=initial_mask.dtype
                )
                result = model(
                    input_ids=token, attention_mask=full_mask, past_key_values=past, use_cache=True
                )
                past = result.past_key_values
                token = result.logits[:, -1:].argmax(dim=-1)

        decode_ms = _event_ms_with_setup(setup_decode, decode, warmup, repeats, dev)
        records.append(
            {
                "benchmark": "model",
                "phase": "decode",
                "prompt_tokens": length,
                "tokens": decode_tokens,
                "latency_ms": decode_ms,
                "tokens_per_second": decode_tokens * 1000 / decode_ms,
                "device": str(dev),
                "gpu": gpu,
                "dtype": str(model_dtype),
                "warmup": warmup,
                "repeats": repeats,
            }
        )
    _write_csv(records, output)
    return records
