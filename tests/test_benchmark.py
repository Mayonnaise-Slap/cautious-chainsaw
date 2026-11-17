import csv
from types import SimpleNamespace

import pytest
import torch

from cautious_chainsaw import benchmark as benchmark_module
from cautious_chainsaw.benchmark import _write_csv, benchmark_matmul, benchmark_model


def test_csv_writer_preserves_structured_records(tmp_path):
    destination = tmp_path / "nested" / "results.csv"
    records = [{"phase": "prefill", "latency_ms": 1.25}, {"phase": "decode", "latency_ms": 2.5}]
    _write_csv(records, destination)
    with destination.open(newline="", encoding="utf-8") as file:
        loaded = list(csv.DictReader(file))
    assert loaded == [
        {"phase": "prefill", "latency_ms": "1.25"},
        {"phase": "decode", "latency_ms": "2.5"},
    ]


def test_model_benchmark_uses_inference_mode_kv_cache_and_writes_both_phases(monkeypatch, tmp_path):
    import torch

    class FakeEvent:
        def __init__(self, enable_timing=True):
            pass

        def record(self, stream=None):
            pass

        def elapsed_time(self, other):
            return 2.0

    monkeypatch.setattr(
        benchmark_module, "_cuda_device", lambda device, dtype="bf16": torch.device("cpu")
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "fake-gpu")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)

    class FakeModel:
        config = SimpleNamespace(vocab_size=20)

        def __init__(self):
            self.parameter = torch.nn.Parameter(torch.ones(1))
            self.mask_lengths = []

        def eval(self):
            return self

        def parameters(self):
            return iter([self.parameter])

        def __call__(self, input_ids, attention_mask, use_cache, past_key_values=None):
            assert torch.is_inference_mode_enabled()
            self.mask_lengths.append(attention_mask.shape[-1])
            logits = torch.zeros((1, input_ids.shape[-1], 20))
            return SimpleNamespace(logits=logits, past_key_values=(object(),))

    model = FakeModel()
    output = tmp_path / "model.csv"
    records = benchmark_model(
        model,
        SimpleNamespace(encode=lambda text, add_special_tokens: [1, 2]),
        prefill_lengths=(3,),
        decode_tokens=2,
        warmup=0,
        repeats=1,
        device="cuda",
        output=output,
    )
    assert [row["phase"] for row in records] == ["prefill", "decode"]
    assert records[1]["prompt_tokens"] == 3
    assert 3 in model.mask_lengths and 4 in model.mask_lengths
    with output.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [row["phase"] for row in rows] == ["prefill", "decode"]


def test_matmul_rejects_unknown_backend_before_device_access():
    with pytest.raises(ValueError, match="bf16"):
        benchmark_matmul(backend="int4")


def test_matmul_rejects_invalid_repeat_count_before_device_access():
    with pytest.raises(ValueError, match="repeats"):
        benchmark_matmul(repeats=0)


def test_matmul_shape_and_tflops_are_structured(monkeypatch):
    monkeypatch.setattr(
        benchmark_module, "_cuda_device", lambda device, dtype="bf16": torch.device("cpu")
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "fake-gpu")
    monkeypatch.setattr(benchmark_module, "_event_ms", lambda fn, warmup, repeats, device: 2.0)

    record = benchmark_matmul(m_sizes=(2,), weight_shapes=((4, 8),), warmup=0, repeats=1)[0]
    assert record["output_shape"] == "(2, 4)"
    assert record["latency_ms"] == 2.0
    assert record["tflops"] == pytest.approx((2 * 2 * 4 * 8) / (2.0 * 1e9))


def test_fp16_matmul_selects_fp16_tensors(monkeypatch):
    monkeypatch.setattr(
        benchmark_module, "_cuda_device", lambda device, dtype="bf16": torch.device("cpu")
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "fake-gpu")
    monkeypatch.setattr(benchmark_module, "_event_ms", lambda fn, warmup, repeats, device: 1.0)
    record = benchmark_matmul(
        backend="fp16", m_sizes=(2,), weight_shapes=((4, 8),), warmup=0, repeats=1
    )[0]
    assert record["dtype"] == "float16"
    assert record["output_shape"] == "(2, 4)"


def test_default_matmul_dimensions_cover_twelve_combinations():
    import inspect

    defaults = inspect.signature(benchmark_matmul).parameters
    m_sizes = defaults["m_sizes"].default
    shapes = defaults["weight_shapes"].default
    assert len(m_sizes) * len(shapes) == 12


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["bf16", "fp16"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA GPU.")
def test_gpu_matmul_smoke(backend):
    if backend == "bf16" and torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("BF16 smoke requires native GPU BF16 support.")
    records = benchmark_matmul(
        backend=backend, m_sizes=(2,), weight_shapes=((4, 8),), warmup=1, repeats=2
    )
    record = records[0]
    assert record["output_shape"] == "(2, 4)"
    assert record["latency_ms"] > 0
    assert record["tflops"] > 0
