from __future__ import annotations

import json
from threading import Event
from pathlib import Path

import pytest

from src.evaluation import (
    EvaluationCancelled, EvaluationError, copy_selected_checkpoint, evaluate_checkpoints,
    load_examples, objective_metrics, repetition_metrics, save_report,
)


def _valid_checkpoint(tmp_path: Path):
    torch = pytest.importorskip("torch")
    from src.model import GPT
    from src.tokenizer import BPETokenizer
    from src.utils import save_checkpoint
    tokenizer = BPETokenizer().fit("olá certo", vocab_size=10, verbose=False)
    tokenizer_path = tmp_path / "tokenizer.json"; tokenizer.save(tokenizer_path)
    config = {"n_layer": 1, "n_embd": 8, "n_head": 1, "block_size": 8, "dropout": 0.0, "vocab_size": tokenizer.vocab_size}
    model = GPT(config); optimizer = torch.optim.AdamW(model.parameters())
    checkpoint = tmp_path / "valid.pt"; save_checkpoint(model, optimizer, 0, checkpoint, config=config, tokenizer=tokenizer)
    return checkpoint, tokenizer_path, config


def _preset(tmp_path: Path) -> Path:
    path = tmp_path / "preset.json"
    path.write_text(json.dumps({"name": "teste", "tests": [{"id": "stable", "category": "test", "prompt": "oi", "reference": "certo", "criteria": {"reference_mode": "exact", "must_include": ["certo"], "min_words": 1}}]}), encoding="utf8")
    return path


def test_preset_requires_stable_unique_ids(tmp_path):
    bad = tmp_path / "bad.json"; bad.write_text('{"tests":[{"id":"x","prompt":"a"},{"id":"x","prompt":"b"}]}', encoding="utf8")
    with pytest.raises(EvaluationError): load_examples(bad)


def test_metrics_detect_reference_instruction_structure_and_repetition():
    value = objective_metrics("Certo certo certo. Certo certo certo.", {"id": "x", "reference": "certo", "criteria": {"reference_mode": "exact", "must_include": ["certo"], "min_words": 1}})
    assert value["instruction"] == 1 and value["repetition"] > 0
    assert repetition_metrics("um dois um dois um dois")["repeated_ngrams"] > 0


def test_report_round_trip_and_copy_preserves_original(tmp_path):
    report = {"format_version": 1, "checkpoints": []}
    destination = save_report(report, tmp_path / "report.json")
    assert json.loads(destination.read_text(encoding="utf8")) == report
    original = tmp_path / "model.pt"; original.write_bytes(b"original")
    copied = copy_selected_checkpoint(original, tmp_path / "best")
    assert copied.read_bytes() == original.read_bytes() and original.exists()


def test_cancelled_evaluation_stops_before_loading(tmp_path):
    cancelled = Event(); cancelled.set()
    with pytest.raises(EvaluationCancelled): evaluate_checkpoints([tmp_path / "missing.pt"], _preset(tmp_path), cancel_event=cancelled)


def test_missing_checkpoint_is_recorded_not_silently_accepted(tmp_path):
    report = evaluate_checkpoints([tmp_path / "missing.pt"], _preset(tmp_path))
    assert report["checkpoints"][0]["error"] and report["recommended_checkpoint"] is None


def test_valid_and_corrupted_checkpoint_handling(tmp_path):
    from src.inference import InferenceError, load_inference_components
    checkpoint, tokenizer_path, config = _valid_checkpoint(tmp_path)
    config_path = tmp_path / "config.json"
    complete = {"n_layer": 1, "n_embd": 8, "n_head": 1, "block_size": 8, "dropout": 0.0, "batch_size": 1, "max_iters": 1, "eval_interval": 1, "eval_iters": 1, "save_interval": 1, "learning_rate": .001, "min_learning_rate": 0, "warmup_iters": 0, "weight_decay": 0, "grad_clip": 0, "early_stopping": False, "patience": 1, "early_stopping_warmup_iters": 0, "validation_fraction": .1, "tokenizer_vocab_size": 10, "tokenizer_training_chars": 0, "tokenizer_min_pair_frequency": 1, "hf_default_size_mb": 1, "amp": False, "device": "cpu", "seed": 1}
    config_path.write_text(json.dumps(complete), encoding="utf8")
    model, tokenizer, device, loaded = load_inference_components(config_path=str(config_path), tokenizer_path=str(tokenizer_path), checkpoint_path=str(checkpoint))
    assert loaded == checkpoint and tokenizer.vocab_size == config["vocab_size"]
    broken = tmp_path / "broken.pt"; broken.write_bytes(b"not a torch checkpoint")
    with pytest.raises(InferenceError): load_inference_components(config_path=str(config_path), tokenizer_path=str(tokenizer_path), checkpoint_path=str(broken))


def test_multiple_checkpoints_receive_same_test_ids(monkeypatch, tmp_path):
    class Tokenizer:
        fingerprint = "ok"
        def encode(self, text): return [1]
        def decode(self, ids): return "certo"
    class Model:
        block_size = 8
        def generate(self, idx, **kwargs): return idx
    monkeypatch.setattr("src.evaluation.load_inference_components", lambda checkpoint_path, **kwargs: (Model(), Tokenizer(), __import__("torch").device("cpu"), Path(checkpoint_path)))
    monkeypatch.setattr("src.evaluation._validation_loss", lambda *args: (None, "skip"))
    report = evaluate_checkpoints(["a.pt", "b.pt"], _preset(tmp_path), {"generations": 2, "seed": 7})
    assert [item["results"][0]["id"] for item in report["checkpoints"]] == ["stable", "stable"]
    assert all(len(item["results"][0]["generations"]) == 2 for item in report["checkpoints"])


def test_seeded_greedy_configuration_is_reproducible(monkeypatch, tmp_path):
    class Tokenizer:
        fingerprint = "ok"
        def encode(self, text): return [1]
        def decode(self, ids): return "certo"
    class Model:
        block_size = 8
        def generate(self, idx, **kwargs): return idx
    monkeypatch.setattr("src.evaluation.load_inference_components", lambda checkpoint_path, **kwargs: (Model(), Tokenizer(), __import__("torch").device("cpu"), Path(checkpoint_path)))
    monkeypatch.setattr("src.evaluation._validation_loss", lambda *args: (None, "skip"))
    settings = {"seed": 42, "greedy": True}
    assert evaluate_checkpoints(["a.pt"], _preset(tmp_path), settings)["checkpoints"][0]["metrics"] == evaluate_checkpoints(["a.pt"], _preset(tmp_path), settings)["checkpoints"][0]["metrics"]
