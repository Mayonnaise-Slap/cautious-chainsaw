"""Command-line interface for low-precision model experiments."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import torch
import typer

from cautious_chainsaw import __version__
from cautious_chainsaw.model import DEFAULT_MODEL_ID

app = typer.Typer(help="Run BF16 or FP16 LLM experiments.", no_args_is_help=True)


def _print_records(records: list[dict[str, object]]) -> None:
    typer.echo(json.dumps(records, indent=2, default=str))


@contextmanager
def _artifact_output(output: Path, resume: bool) -> Iterator[Path | None]:
    """Yield a temporary output path, then publish it atomically after a successful run."""
    if resume and output.is_file() and output.stat().st_size > 0:
        typer.echo(f"Reusing completed output at {output}.")
        yield None
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=output.suffix, dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        yield temporary
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError("The experiment did not produce a nonempty output artifact.")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


@app.command("bench-matmul")
def bench_matmul(
    backend: Annotated[str, typer.Option(help="Matmul dtype: bf16 or fp16.")] = "bf16",
    device: Annotated[str, typer.Option(help="CUDA device, for example cuda:0.")] = "cuda",
    output: Annotated[Path, typer.Option(help="CSV output path.")] = Path("results/matmul.csv"),
    warmup: Annotated[int, typer.Option(min=0, help="Warmup iterations.")] = 10,
    repeats: Annotated[int, typer.Option(min=1, help="Timed repetitions.")] = 100,
    m_sizes: Annotated[
        list[int] | None, typer.Option("--m", help="Batch size M; may be repeated.")
    ] = None,
    weight_shapes: Annotated[
        list[str] | None, typer.Option("--weight-shape", help="Weight shape N,K; may be repeated.")
    ] = None,
    resume: Annotated[
        bool,
        typer.Option(help="Reuse a nonempty result at this path; the path identifies the run."),
    ] = False,
) -> None:
    """Benchmark BF16 matrix multiplication for Llama layer shapes."""
    if backend not in {"bf16", "fp16"}:
        raise typer.BadParameter("backend must be bf16 or fp16.")
    if m_sizes and any(size < 1 for size in m_sizes):
        raise typer.BadParameter("Batch sizes must be positive.")
    shapes = None
    if weight_shapes:
        try:
            shapes = tuple(
                tuple(int(part.strip()) for part in value.split(",")) for value in weight_shapes
            )
        except ValueError as exc:
            raise typer.BadParameter("Each --weight-shape must use N,K integer syntax.") from exc
        if any(len(shape) != 2 or min(shape) < 1 for shape in shapes):
            raise typer.BadParameter("Each --weight-shape must contain two positive integers: N,K.")

    try:
        with _artifact_output(output, resume) as temporary:
            if temporary is None:
                return
            from cautious_chainsaw.benchmark import benchmark_matmul

            records = benchmark_matmul(
                backend=backend,
                device=device,
                output=temporary,
                warmup=warmup,
                repeats=repeats,
                **({"m_sizes": tuple(m_sizes)} if m_sizes else {}),
                **({"weight_shapes": shapes} if shapes else {}),
            )
    except (ValueError, RuntimeError, NotImplementedError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print_records(records)


@app.command("bench-model")
def bench_model(
    backend: Annotated[str, typer.Option(help="Model dtype: bf16 or fp16.")] = "bf16",
    model_id: Annotated[
        str, typer.Option(help="Hugging Face model identifier.")
    ] = DEFAULT_MODEL_ID,
    device: Annotated[str, typer.Option(help="CUDA device, for example cuda:0.")] = "cuda",
    output: Annotated[Path, typer.Option(help="CSV output path.")] = Path("results/model.csv"),
    warmup: Annotated[int, typer.Option(min=0, help="Warmup iterations.")] = 2,
    repeats: Annotated[int, typer.Option(min=1, help="Timed repetitions.")] = 5,
    decode_tokens: Annotated[int, typer.Option(min=1, help="Number of generated tokens.")] = 32,
    resume: Annotated[
        bool,
        typer.Option(help="Reuse a nonempty result at this path; the path identifies the run."),
    ] = False,
) -> None:
    """Measure model prefill and autoregressive decode latency."""
    if backend not in {"bf16", "fp16"}:
        raise typer.BadParameter("backend must be bf16 or fp16.")
    try:
        with _artifact_output(output, resume) as temporary:
            if temporary is None:
                return
            from cautious_chainsaw.benchmark import benchmark_model
            from cautious_chainsaw.model import load_model

            model, tokenizer = load_model(model_id=model_id, device=device, dtype=backend)
            records = benchmark_model(
                model,
                tokenizer,
                decode_tokens=decode_tokens,
                device=device,
                warmup=warmup,
                repeats=repeats,
                output=temporary,
            )
    except (ValueError, RuntimeError, NotImplementedError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print_records(records)


@app.command("perplexity")
def perplexity_command(
    backend: Annotated[str, typer.Option(help="Model dtype: bf16 or fp16.")] = "bf16",
    model_id: Annotated[
        str, typer.Option(help="Hugging Face model identifier.")
    ] = DEFAULT_MODEL_ID,
    device: Annotated[str, typer.Option(help="Evaluation device.")] = "cuda",
    output: Annotated[Path, typer.Option(help="JSON output path.")] = Path(
        "results/perplexity.json"
    ),
    max_length: Annotated[int, typer.Option(min=2, help="Evaluation context length.")] = 1024,
    stride: Annotated[int, typer.Option(min=1, help="Window step in tokens.")] = 512,
    max_tokens: Annotated[
        int | None, typer.Option(min=1, help="Optional token limit for quick checks.")
    ] = None,
    max_windows: Annotated[
        int | None, typer.Option(min=1, help="Optional window limit for quick checks.")
    ] = None,
    resume: Annotated[
        bool,
        typer.Option(help="Reuse a nonempty result at this path; the path identifies the run."),
    ] = False,
) -> None:
    """Evaluate a model on WikiText-2 test data."""
    if backend not in {"bf16", "fp16"}:
        raise typer.BadParameter("backend must be bf16 or fp16.")
    if stride >= max_length:
        raise typer.BadParameter("stride must be smaller than max_length.")
    try:
        with _artifact_output(output, resume) as temporary:
            if temporary is None:
                return
            from cautious_chainsaw.model import load_model
            from cautious_chainsaw.perplexity import evaluate_perplexity

            model, tokenizer = load_model(model_id=model_id, device=device, dtype=backend)
            result = evaluate_perplexity(
                model,
                tokenizer,
                max_length=max_length,
                stride=stride,
                device=device,
                max_tokens=max_tokens,
                max_windows=max_windows,
                output=temporary,
            )
    except (ValueError, RuntimeError, NotImplementedError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(result, indent=2, default=str))


@app.command()
def doctor() -> None:
    """Show installed runtime and CUDA status without loading a model."""
    cuda_available = torch.cuda.is_available()
    capability = torch.cuda.get_device_capability(0) if cuda_available else None
    typer.echo(
        json.dumps(
            {
                "version": __version__,
                "python_torch": torch.__version__,
                "cuda_available": cuda_available,
                "cuda_version": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
                "compute_capability": capability,
                "native_bf16_supported": bool(capability and capability >= (8, 0)),
            },
            indent=2,
        )
    )


@app.command()
def smoke(
    backend: Annotated[str, typer.Option(help="CUDA dtype to check: bf16 or fp16.")] = "bf16",
) -> None:
    """Run a tiny CUDA low-precision operation to check the experiment runtime."""
    if backend not in {"bf16", "fp16"}:
        raise typer.BadParameter("backend must be bf16 or fp16.")
    if not torch.cuda.is_available():
        raise typer.BadParameter("CUDA is unavailable; run smoke in a CUDA-enabled Colab runtime.")
    capability = torch.cuda.get_device_capability(0)
    if backend == "bf16" and capability < (8, 0):
        version = f"{capability[0]}.{capability[1]}"
        raise typer.BadParameter(
            f"Native BF16 requires compute capability 8.0 or newer; detected {version}."
        )
    dtype = torch.bfloat16 if backend == "bf16" else torch.float16
    left = torch.ones((2, 3), dtype=dtype, device="cuda")
    right = torch.ones((3, 4), dtype=dtype, device="cuda")
    result = left @ right
    torch.cuda.synchronize()
    if result.shape != (2, 4) or not bool(torch.all(result == 3)):
        raise typer.Exit(code=1)
    typer.echo(f"CUDA {backend.upper()} smoke check passed.")


if __name__ == "__main__":
    app()
