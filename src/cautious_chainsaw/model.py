"""Hugging Face model loading helpers."""

from __future__ import annotations

from typing import Any

DEFAULT_MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"


def load_model(
    quantized: bool = False,
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "cuda",
    dtype: str = "bf16",
) -> tuple[Any, Any]:
    """Load a causal LM and its tokenizer in BF16 or FP16.

    Hugging Face's standard cache environment variables (HF_HOME,
    HF_HUB_CACHE, TRANSFORMERS_CACHE) are respected by ``from_pretrained``.
    """
    if quantized:
        raise NotImplementedError("INT4 model quantization is not implemented yet.")
    if dtype not in {"bf16", "fp16"}:
        raise ValueError("dtype must be either 'bf16' or 'fp16'.")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install PyTorch and Transformers to load a model.") from exc

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Loading on CUDA requires an available CUDA GPU.")
    if device.startswith("cuda") and dtype == "bf16":
        if torch.cuda.get_device_capability(torch.device(device)) < (8, 0):
            raise RuntimeError("BF16 model loading requires a GPU with native BF16 support.")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if dtype == "bf16" else torch.float16,
    )
    model.to(device)
    model.eval()
    return model, tokenizer
