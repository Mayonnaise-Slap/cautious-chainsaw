"""Perplexity evaluation for causal language models."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def evaluate_perplexity(
    model: Any,
    tokenizer: Any,
    *,
    max_length: int = 1024,
    stride: int = 512,
    device: str | torch.device = "cuda",
    max_tokens: int | None = None,
    max_windows: int | None = None,
    output: str | Path | None = None,
    dataset_name: str = "Salesforce/wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
) -> dict[str, Any]:
    """Evaluate token-weighted NLL over a text dataset and optionally save JSON.

    Each window advances by ``stride``. The overlap provides context, while a
    target mask scores only tokens not scored by a previous window. The first
    token in the corpus has no preceding context and is therefore not scored.
    """
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    if not 1 <= stride < max_length:
        raise ValueError("stride must be between 1 and max_length - 1")
    if max_tokens is not None and max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if max_windows is not None and max_windows < 1:
        raise ValueError("max_windows must be positive")

    from datasets import load_dataset

    dataset = load_dataset(dataset_name, dataset_config, split=split)
    text = "\n\n".join(value for value in dataset["text"] if value)
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    token_ids = encoded["input_ids"]
    if token_ids.ndim == 1:
        token_ids = token_ids.unsqueeze(0)
    total_tokens = int(token_ids.shape[1])
    if max_tokens is not None:
        total_tokens = min(total_tokens, max_tokens)
    token_ids = token_ids[:, :total_tokens]
    if total_tokens < 2:
        raise ValueError("dataset must contain at least two tokens")

    target_device = torch.device(device)
    model.eval()
    nll_sum = 0.0
    scored_tokens = 0
    previous_end = 0
    windows = 0

    with torch.inference_mode():
        for start in range(0, total_tokens, stride):
            if max_windows is not None and windows >= max_windows:
                break
            end = min(start + max_length, total_tokens)
            inputs = token_ids[:, start:end].to(target_device)
            outputs = model(input_ids=inputs, use_cache=False)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            # Causal LM logits at position j predict input token j+1.
            target_start = max(previous_end, start + 1)
            target_start = min(target_start, end)
            if target_start < end:
                local_targets = torch.arange(
                    target_start - start, end - start, device=logits.device
                )
                target_ids = inputs[:, local_targets]
                prediction_logits = logits[:, local_targets - 1, :]
                losses = F.cross_entropy(
                    prediction_logits.float().reshape(-1, prediction_logits.shape[-1]),
                    target_ids.reshape(-1),
                    reduction="sum",
                )
                nll_sum += float(losses.item())
                scored_tokens += int(target_ids.numel())
            previous_end = end
            windows += 1
            if end == total_tokens:
                break

    if scored_tokens == 0:
        raise ValueError("no tokens were scored; increase max_length or max_tokens")
    mean_nll = nll_sum / scored_tokens
    dtype = _model_dtype(model)
    result: dict[str, Any] = {
        "dataset": dataset_name,
        "config": dataset_config,
        "split": split,
        "perplexity": math.exp(mean_nll),
        "mean_nll": mean_nll,
        "token_count": scored_tokens,
        "input_token_count": total_tokens,
        "window_count": windows,
        "max_length": max_length,
        "stride": stride,
        "model": getattr(getattr(model, "config", None), "_name_or_path", None),
        "dtype": dtype,
    }
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def _model_dtype(model: Any) -> str | None:
    """Return a model parameter dtype when available, including for test doubles."""
    parameters = getattr(model, "parameters", None)
    if parameters is None:
        return None
    try:
        parameter = next(parameters())
    except StopIteration:
        return None
    return str(parameter.dtype).removeprefix("torch.")
