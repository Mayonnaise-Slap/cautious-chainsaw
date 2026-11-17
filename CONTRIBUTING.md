# Contributing

This repository is a small research project for BF16/FP16 Llama baselines and a future INT4/Triton implementation. Keep changes easy to verify and aligned with the current package layout.

## Design principles

- Keep experiment logic in `src/cautious_chainsaw/`; keep `colab.ipynb` as a thin launcher and results viewer. Put behavior, retries, and output handling in the Python package, and let notebook cells call the CLI with visible `!` commands.
- Preserve the current public CLI and Python APIs when extending them. Keep the future INT4 integration at the existing backend/model seams so quantized execution can reuse benchmark and perplexity flows.
- Keep the scope minimal. Add modules or abstractions only for concrete behavior, and keep model, benchmark, evaluation, and CLI responsibilities in their existing modules.
- Keep CPU unit tests independent of downloaded model and dataset assets. Mark hardware and external-resource checks with the configured `gpu` and `integration` pytest markers.
- Keep generated weights, datasets, and experiment outputs out of Git; use `results/` for local outputs.

## Environment and checks

For normal project setup, use the commands declared by the repository:

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest -m 'not gpu and not integration'
```

Run the CUDA matmul smoke tests only on a compatible GPU:

```bash
uv run pytest -m 'gpu and not integration'
```

The model integration test downloads or loads `unsloth/Llama-3.2-1B-Instruct`. Opt into it explicitly in a CUDA Colab runtime:

```bash
RUN_HF_INTEGRATION=1 uv run pytest -m integration
```

On SSH sessions, run resource-heavy operations, including containers, through `systemd-run` with explicit resource limits.

## Commit messages

Use `type(scope): subject`. Name the affected area in `scope`; write the subject in lowercase as an imperative.

Examples:

```text
feat(benchmark): add FP16 matmul timing
docs(repo): add contribution guidelines
```
