# Cautious Chainsaw

A small research codebase for measuring BF16 and FP16 Llama baselines before adding INT4 weight quantization and Triton kernels. The current iteration includes model loading, matmul and model benchmarks, and WikiText-2 perplexity evaluation. INT4 quantization and INT4 matmul are not implemented yet.

## Requirements

- Python 3.12 or newer
- [`uv`](https://docs.astral.sh/uv/)
- NVIDIA GPU with CUDA; BF16 requires native support (for example, L4 or A100), while FP16 also supports T4

CPU-only unit tests do not download a model or dataset and do not require a GPU. Model and WikiText-2 evaluation commands download their Hugging Face assets on first use.

## Install

```bash
uv sync --locked
```

Run the runtime checks and CPU-only unit tests (no model or dataset downloads):

```bash
uv run chainsaw doctor
uv run chainsaw smoke
uv run pytest -m 'not gpu and not integration'
```

## Experiments

Run the BF16 matmul benchmark over the default Llama layer shapes and save a CSV:

```bash
uv run chainsaw bench-matmul --backend bf16 --output results/matmul.csv
```

Measure prefill and autoregressive decode separately:

```bash
uv run chainsaw bench-model --backend bf16 --model-id unsloth/Llama-3.2-1B-Instruct --output results/model.csv
```

Evaluate WikiText-2 test perplexity with overlapping context windows:

```bash
uv run chainsaw perplexity --backend bf16 --model-id unsloth/Llama-3.2-1B-Instruct --max-length 1024 --stride 512 --output results/perplexity.json
```

All three experiments also accept `--backend fp16`, which supports NVIDIA T4. For example:

```bash
uv run chainsaw bench-matmul --backend fp16 --output results/matmul-fp16.csv
uv run chainsaw bench-model --backend fp16 --model-id unsloth/Llama-3.2-1B-Instruct --output results/model-fp16.csv
uv run chainsaw perplexity --backend fp16 --model-id unsloth/Llama-3.2-1B-Instruct --output results/perplexity-fp16.json
uv run chainsaw smoke --backend fp16
```

By default, the CLI writes `results/matmul.csv`, `results/model.csv`, and `results/perplexity.json`. The commands above show how to choose explicit output paths. Matmul CSV rows contain latency and TFLOPS; model CSV rows keep `prefill` and `decode` phases separate; the JSON includes mean NLL, perplexity, and scored token/window counts.

Use `--max-tokens` or `--max-windows` for a bounded evaluation while debugging. `stride` must be smaller than `max-length` so consecutive windows overlap and each newly scored token has its preceding context. Run `--help` on the CLI or any command to see all available options.

## Google Colab

Open `colab.ipynb` in a Colab T4, L4, or A100 runtime. It selects BF16 when native BF16 is available and FP16 otherwise; set `BACKEND` to override. The notebook checks CUDA before installing dependencies, keeps Hugging Face caches and experiment files on Drive, and stores both the fresh source clone and `uv` environment under `/content`. Each run of the clone/install cell makes a new checkout, so code does not persist across runs; Drive caches and results do. Each precision has its own Drive run folder with `matmul.csv`, `model.csv`, `perplexity.json`, and `summary.png`. The visible `!` commands call the CLI with `--resume`: a nonempty file at the final output path is reused before loading the model, while a new run publishes its result atomically only after success. An interrupted experiment restarts from the beginning. Change `RUN_NAME` when changing the model or experiment parameters, or remove `--resume` to recompute an existing path.

### Colab Secrets

In the Colab **Secrets** panel, add:

- `GITHUB_TOKEN`: optional for public repositories; add a fine-grained personal access token restricted to this repository with **Contents: read** permission only when authenticated cloning is needed. Enable notebook access. The notebook uses anonymous HTTPS when the secret is absent, and uses a temporary `GIT_ASKPASS` script when it is present. The clean HTTPS URL is stored in Git metadata; the token is not written to notebook output, command arguments, Git configuration, or Drive. The token is exported only for the clone command and then removed, and the temporary script is deleted. If an organization requires approval for fine-grained tokens, wait for that approval. Fine-grained tokens have some collaborator-related limitations; see [GitHub's token documentation](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).
- `HF_TOKEN`: an optional Hugging Face read token. Public assets work without it. The notebook passes it to CLI processes through the `HF_TOKEN` environment variable and sets the Hugging Face token file path under `/content`, so it is not stored in the Drive cache. For a gated or private model, the token's Hugging Face account must already have access to that model. See [Hugging Face's environment variable reference](https://huggingface.co/docs/huggingface_hub/main/package_reference/environment_variables) and [token settings](https://huggingface.co/settings/tokens).

The notebook reads secrets with Colab's [`google.colab.userdata.get`](https://github.com/googlecolab/colabtools/blob/main/google/colab/userdata.py) API. Do not paste tokens into notebook cells or change the repository URL to include credentials. Every run of the clone/install cell clones the configured branch or tag into a new managed temporary folder under `/content`; it does not reuse or delete source directories on Drive.

To run the opt-in GPU matmul tests in a compatible Colab runtime, use `uv run pytest -m 'gpu and not integration'`. The tests cover BF16/FP16 matmul; the BF16 case skips when native BF16 is unavailable. The model integration smoke can download Llama-3.2-1B-Instruct, so opt in explicitly with `RUN_HF_INTEGRATION=1 uv run pytest -m integration` on a CUDA runtime.

## Project layout

```text
src/cautious_chainsaw/   experiment logic and CLI
tests/                   CPU unit tests and GPU/integration tests
colab.ipynb              thin Colab launcher and result viewer
results/                 local experiment outputs
```

## Current scope

- Implemented: BF16/FP16 model loading, matmul benchmark, model prefill/decode benchmark, and token-weighted WikiText-2 perplexity evaluation.
- Not implemented: INT4 quantization, packed-weight kernels, INT4 matmul, and replacement of Llama linear layers.
- GPU performance and full-model evaluation require a compatible CUDA runtime and are intentionally not asserted by CPU unit tests.
