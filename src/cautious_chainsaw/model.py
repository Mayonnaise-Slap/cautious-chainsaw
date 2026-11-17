"""Hugging Face model loading helpers."""

from __future__ import annotations

from tempfile import TemporaryDirectory
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

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as exc:
        if str(exc).strip() != "expected value at line 1 column 1":
            raise
        # Colab Drive caches can contain a corrupt snapshot file that force_download
        # will still see as present. Isolate the tokenizer retry from that cache;
        # model weights continue using the user's normal Hugging Face cache.
        with TemporaryDirectory(prefix="chainsaw-tokenizer-") as tokenizer_cache:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=tokenizer_cache,
                force_download=True,
            )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if dtype == "bf16" else torch.float16,
    )
    model.to(device)
    model.eval()
    return model, tokenizer
