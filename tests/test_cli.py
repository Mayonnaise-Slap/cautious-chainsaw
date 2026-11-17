import json

from typer.testing import CliRunner

from cautious_chainsaw.cli import app

runner = CliRunner()


def test_cli_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "bench-matmul" in result.stdout
    assert "perplexity" in result.stdout


def test_perplexity_rejects_stride_at_or_above_max_length() -> None:
    result = runner.invoke(app, ["perplexity", "--max-length", "128", "--stride", "128"])
    assert result.exit_code != 0
    assert "stride must be smaller" in result.output


def test_model_benchmark_rejects_unknown_backend_without_loading_model() -> None:
    result = runner.invoke(app, ["bench-model", "--backend", "int4"])
    assert result.exit_code != 0
    assert "backend must be bf16 or fp16" in result.output


def test_matmul_rejects_nonpositive_repetitions() -> None:
    result = runner.invoke(app, ["bench-matmul", "--repeats", "0"])
    assert result.exit_code != 0


def test_perplexity_routes_fp16_to_model_loader(monkeypatch, tmp_path) -> None:
    import cautious_chainsaw.model as model_module
    import cautious_chainsaw.perplexity as perplexity_module

    calls = {}

    def fake_load_model(**kwargs):
        calls.update(kwargs)
        return object(), object()

    def fake_evaluate(*args, **kwargs):
        result = {
            "perplexity": 1.0,
            "backend": "fp16",
            "dtype": "float16",
            "token_count": 10,
        }
        kwargs["output"].write_text(json.dumps(result))
        return result

    monkeypatch.setattr(model_module, "load_model", fake_load_model)
    monkeypatch.setattr(perplexity_module, "evaluate_perplexity", fake_evaluate)
    output = tmp_path / "perplexity.json"
    result = runner.invoke(
        app,
        [
            "perplexity",
            "--backend",
            "fp16",
            "--max-length",
            "16",
            "--stride",
            "8",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls["dtype"] == "fp16"
    assert json.loads(output.read_text())["dtype"] == "float16"
    assert '"backend": "fp16"' in result.output


def test_resume_reuses_existing_output_before_loading_model(monkeypatch, tmp_path) -> None:
    import cautious_chainsaw.model as model_module

    output = tmp_path / "model.csv"
    output.write_text("already complete")

    def fail_if_loaded(**kwargs):
        raise AssertionError("model must not load when resuming an existing output")

    monkeypatch.setattr(model_module, "load_model", fail_if_loaded)
    result = runner.invoke(
        app,
        ["bench-model", "--output", str(output), "--backend", "fp16", "--resume"],
    )
    assert result.exit_code == 0, result.output
    assert output.read_text() == "already complete"
    assert "Reusing completed output" in result.output


def test_failed_rerun_preserves_previous_output_and_cleans_temporary_file(
    monkeypatch, tmp_path
) -> None:
    import cautious_chainsaw.model as model_module
    import cautious_chainsaw.perplexity as perplexity_module

    output = tmp_path / "perplexity.json"
    original = '{"perplexity": 2.0, "token_count": 8}'
    output.write_text(original)
    monkeypatch.setattr(model_module, "load_model", lambda **kwargs: (object(), object()))

    def fail_after_partial_write(*args, **kwargs):
        kwargs["output"].write_text("partial")
        raise RuntimeError("simulated evaluation failure")

    monkeypatch.setattr(perplexity_module, "evaluate_perplexity", fail_after_partial_write)
    result = runner.invoke(
        app,
        [
            "perplexity",
            "--output",
            str(output),
            "--max-length",
            "16",
            "--stride",
            "8",
        ],
    )
    assert result.exit_code != 0
    assert output.read_text() == original
    assert not list(tmp_path.glob(".perplexity.json.*.json"))
