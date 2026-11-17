import os
import sys
import types

import pytest

from cautious_chainsaw.model import DEFAULT_MODEL_ID
from cautious_chainsaw.model import load_model as load_hf_model


def test_quantized_loader_is_explicitly_unimplemented():
    with pytest.raises(NotImplementedError, match="INT4"):
        load_hf_model(quantized=True)


def test_loader_returns_model_and_tokenizer(monkeypatch):
    calls = {}
    fake_torch = types.SimpleNamespace(
        bfloat16="bf16", float16="fp16", cuda=types.SimpleNamespace(is_available=lambda: False)
    )

    class FakeModel:
        def to(self, device):
            calls["device"] = device
            return self

        def eval(self):
            calls["eval"] = True
            return self

    model = FakeModel()
    tokenizer = object()
    transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda model_id: tokenizer),
        AutoModelForCausalLM=types.SimpleNamespace(
            from_pretrained=lambda model_id, torch_dtype: model
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    actual_model, actual_tokenizer = load_hf_model(device="cpu")
    assert actual_model is model
    assert actual_tokenizer is tokenizer
    assert calls == {"device": "cpu", "eval": True}
    assert DEFAULT_MODEL_ID == "unsloth/Llama-3.2-1B-Instruct"


def test_loader_routes_fp16_dtype(monkeypatch):
    torch_values = []
    fake_torch = types.SimpleNamespace(
        bfloat16="bf16", float16="fp16", cuda=types.SimpleNamespace(is_available=lambda: False)
    )
    model = types.SimpleNamespace(to=lambda device: None, eval=lambda: None)
    tokenizer = object()

    def fake_model_loader(model_id, torch_dtype):
        torch_values.append(torch_dtype)
        return model

    transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda model_id: tokenizer),
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=fake_model_loader),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    actual_model, actual_tokenizer = load_hf_model(device="cpu", dtype="fp16")
    assert actual_model is not None and actual_tokenizer is tokenizer
    assert torch_values == ["fp16"]


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_HF_INTEGRATION") != "1",
    reason="Set RUN_HF_INTEGRATION=1 to load the Hugging Face model.",
)
def test_hf_model_benchmark_fp16_smoke():
    from cautious_chainsaw.benchmark import benchmark_model

    model, tokenizer = load_hf_model(dtype="fp16")
    records = benchmark_model(
        model,
        tokenizer,
        prefill_lengths=(4,),
        decode_tokens=2,
        warmup=0,
        repeats=1,
    )
    assert [record["phase"] for record in records] == ["prefill", "decode"]
    assert all(record["latency_ms"] > 0 for record in records)
