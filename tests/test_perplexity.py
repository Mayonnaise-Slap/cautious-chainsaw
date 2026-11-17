import json
import math
import sys
import types

import pytest
import torch

from cautious_chainsaw.perplexity import evaluate_perplexity


class _Tokenizer:
    def __init__(self, tokens):
        self.tokens = tokens

    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([self.tokens], dtype=torch.long)}


class _UniformModel:
    def __init__(self, vocab_size=11):
        self.vocab_size = vocab_size
        self.seen = []

    def eval(self):
        return self

    def parameters(self):
        yield torch.empty((), dtype=torch.float16)

    def __call__(self, input_ids, use_cache=True):
        self.seen.append(input_ids.cpu().tolist()[0])
        logits = torch.zeros((*input_ids.shape, self.vocab_size), device=input_ids.device)
        return types.SimpleNamespace(logits=logits)


class _NextTokenModel:
    def __init__(self, vocab_size=11):
        self.vocab_size = vocab_size

    def eval(self):
        return self

    def __call__(self, input_ids, use_cache=True):
        logits = torch.zeros((*input_ids.shape, self.vocab_size), device=input_ids.device)
        targets = (input_ids + 1).clamp_max(self.vocab_size - 1)
        logits.scatter_(-1, targets.unsqueeze(-1), 2.0)
        return types.SimpleNamespace(logits=logits)


def _install_dataset(monkeypatch, texts):
    dataset_module = types.ModuleType("datasets")
    dataset_module.load_dataset = lambda *args, **kwargs: {"text": texts}
    monkeypatch.setitem(sys.modules, "datasets", dataset_module)


def test_sliding_windows_score_every_next_token_once(monkeypatch):
    _install_dataset(monkeypatch, ["synthetic"])
    model = _UniformModel()

    result = evaluate_perplexity(
        model,
        _Tokenizer(list(range(7))),
        max_length=4,
        stride=2,
        device="cpu",
    )

    assert model.seen == [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6]]
    assert result["token_count"] == 6
    assert result["input_token_count"] == 7
    assert result["window_count"] == 3
    assert result["mean_nll"] == pytest.approx(math.log(11))
    assert result["perplexity"] == pytest.approx(11)
    assert result["dtype"] == "float16"


def test_stride_equal_max_length_is_rejected(monkeypatch):
    _install_dataset(monkeypatch, ["synthetic"])
    with pytest.raises(ValueError, match="max_length - 1"):
        evaluate_perplexity(
            _UniformModel(), _Tokenizer(list(range(9))), max_length=4, stride=4, device="cpu"
        )


def test_causal_shift_scores_next_token_using_previous_position(monkeypatch):
    _install_dataset(monkeypatch, ["synthetic"])
    result = evaluate_perplexity(
        _NextTokenModel(), _Tokenizer(list(range(8))), max_length=5, stride=3, device="cpu"
    )
    expected_nll = math.log(math.exp(2) + 10) - 2
    assert result["token_count"] == 7
    assert result["mean_nll"] == pytest.approx(expected_nll)


def test_max_tokens_and_max_windows_bound_evaluation_and_save_json(monkeypatch, tmp_path):
    _install_dataset(monkeypatch, ["synthetic"])
    path = tmp_path / "metrics" / "ppl.json"
    result = evaluate_perplexity(
        _UniformModel(),
        _Tokenizer(list(range(10))),
        max_length=5,
        stride=3,
        max_tokens=8,
        max_windows=1,
        device="cpu",
        output=path,
    )
    assert result["input_token_count"] == 8
    assert result["token_count"] == 4
    assert result["window_count"] == 1
    assert json.loads(path.read_text(encoding="utf-8")) == result


def test_invalid_window_configuration_rejected_before_loading(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("dataset must not be loaded for invalid arguments")

    module = types.ModuleType("datasets")
    module.load_dataset = unexpected
    monkeypatch.setitem(sys.modules, "datasets", module)
    with pytest.raises(ValueError, match="stride"):
        evaluate_perplexity(_UniformModel(), _Tokenizer([1, 2]), max_length=4, stride=5)
