import os
import sys
import types
from pathlib import Path

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


def test_loader_refetches_only_tokenizer_after_json_parse_error(monkeypatch):
    calls = []
    temporary_caches = []
    fake_torch = types.SimpleNamespace(
        bfloat16="bf16", float16="fp16", cuda=types.SimpleNamespace(is_available=lambda: False)
    )
    tokenizer = object()
    model = types.SimpleNamespace(to=lambda device: None, eval=lambda: None)

    def load_tokenizer(model_id, **kwargs):
        calls.append((model_id, kwargs))
        if len(calls) == 1:
            raise Exception("expected value at line 1 column 1")
        temporary_caches.append(Path(kwargs["cache_dir"]))
        assert temporary_caches[-1].is_dir()
        assert kwargs["force_download"] is True
        return tokenizer

    transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=load_tokenizer),
        AutoModelForCausalLM=types.SimpleNamespace(
            from_pretrained=lambda model_id, torch_dtype: model
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    actual_model, actual_tokenizer = load_hf_model(device="cpu", dtype="fp16")

    assert actual_model is model
    assert actual_tokenizer is tokenizer
    assert calls[0] == (DEFAULT_MODEL_ID, {})
    assert calls[1][0] == DEFAULT_MODEL_ID
    assert set(calls[1][1]) == {"cache_dir", "force_download"}
    assert calls[1][1]["force_download"] is True
    assert temporary_caches[0].parent.exists()
    assert not temporary_caches[0].exists()


def test_loader_does_not_retry_other_tokenizer_errors(monkeypatch):
    parse_error = OSError("connection refused")
    calls = []
    fake_torch = types.SimpleNamespace(
        bfloat16="bf16", float16="fp16", cuda=types.SimpleNamespace(is_available=lambda: False)
    )

    def load_tokenizer(model_id, **kwargs):
        calls.append(kwargs)
        raise parse_error

    transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=load_tokenizer),
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: None),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    with pytest.raises(OSError) as raised:
        load_hf_model(device="cpu", dtype="fp16")
    assert raised.value is parse_error
    assert calls == [{}]


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
