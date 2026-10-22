"""Testes pequenos da integração RAG QA; não acessam a rede nem baixam o dataset real."""
import json
import sys
import types

import numpy as np
import pytest

# Este ambiente de teste isolado pode não incluir colorama; em produção ela
# continua declarada normalmente em requirements.txt.
try:
    import colorama  # noqa: F401
except ImportError:
    colorama_stub = types.ModuleType("colorama")
    class _Colors:
        def __getattr__(self, _name): return ""
    colorama_stub.Fore = _Colors()
    colorama_stub.Style = _Colors()
    colorama_stub.init = lambda **_kwargs: None
    sys.modules["colorama"] = colorama_stub

import src.data as data


def test_ragqa_schema_mapping_preserves_instruction_context_answer():
    example = {"instruction": "Compare A e B", "input": "A e B", "output": "A é diferente de B."}
    text = data.example_to_text(example, "ragqa")
    assert "Compare A e B" in text
    assert "A e B" in text
    assert "A é diferente de B." in text


def test_ragqa_mapping_discards_missing_instruction_or_answer():
    assert data.example_to_text({"instruction": "Oi", "input": "", "output": ""}, "ragqa") == ""
    assert data.example_to_text({"instruction": "", "input": "contexto", "output": "Resposta"}, "ragqa") == ""


def test_ragqa_download_is_complete_deduplicated_and_atomic(tmp_path, monkeypatch):
    rows = [
        {"instruction": "Pergunta um", "input": "Contexto", "output": "Resposta um"},
        {"instruction": "Pergunta um", "input": "Contexto", "output": "Resposta um"},
        {"instruction": "Pergunta dois", "input": "", "output": "Resposta dois"},
        {"instruction": "Sem resposta", "input": "", "output": ""},
    ]
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = lambda *a, **kw: iter(rows)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    out = tmp_path / "train.jsonl"
    result = data._stream_ragqa("fake/repo", out, out.with_suffix(".download"), 100_000, "train", None)
    saved = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert result["records"] == 2
    assert result["duplicates"] == 1
    assert result["skipped"] == 1
    assert len(saved) == 2
    assert all(row["output"] for row in saved)
    assert not out.with_suffix(".download").exists()


def test_ragqa_preparation_splits_complete_examples_and_reports_stats(tmp_path, monkeypatch):
    raw = tmp_path / "ragqa.jsonl"
    rows = []
    for i in range(20):
        rows.append({"instruction": f"Pergunta {i}", "input": f"Contexto {i}", "output": f"Resposta completa {i}"})
    rows.append({"instruction": "invalida", "input": "", "output": ""})
    raw.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    out_dir = tmp_path / "processed"
    monkeypatch.setattr(data, "processed_paths", lambda: {
        "tokenizer": out_dir / "tokenizer.json", "train": out_dir / "train.bin",
        "val": out_dir / "val.bin", "metadata": out_dir / "metadata.json"})
    monkeypatch.setattr(data, "project_path", lambda value: __import__("pathlib").Path(value))
    cfg = {"tokenizer_vocab_size": 300, "tokenizer_training_chars": 100_000,
           "tokenizer_min_pair_frequency": 1, "validation_fraction": 0.25}
    metadata = data.prepare_corpus(raw, cfg, dataset_type="ragqa")
    assert metadata["valid_examples"] == 20
    assert metadata["discarded_examples"] == 1
    assert metadata["train_examples"] + metadata["val_examples"] == 20
    assert metadata["train_tokens"] > 2 and metadata["val_tokens"] > 2
    assert metadata["train_bytes"] == (out_dir / "train.bin").stat().st_size
    tokenizer, train, val, loaded_meta = data.load_processed_data()
    assert len(train) == metadata["train_tokens"]
    assert len(val) == metadata["val_tokens"]
    assert loaded_meta["dataset_type"] == "ragqa"


def test_ragqa_reuses_valid_local_jsonl_with_matching_source_marker_without_network(tmp_path, monkeypatch):
    out = tmp_path / "train.jsonl"
    out.write_text(json.dumps({"instruction": "Oi", "input": "", "output": "Olá"}) + "\n", encoding="utf-8")
    out.with_suffix(out.suffix + ".source.json").write_text(json.dumps({"dataset": "Madras1/rag-qa-fulltext-ptbr", "split": "train", "dataset_config": None}), encoding="utf-8")
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("um arquivo local válido deveria ser reutilizado")
    monkeypatch.setattr(data, "_stream_ragqa", fail_if_called)
    result = data.stream_dataset_to_file("Madras1/rag-qa-fulltext-ptbr", out,
                                         dataset_type="ragqa", size_mb=10)
    assert result["cached"] is True
    assert result["records"] == 1


def test_training_selection_routes_ragqa_into_existing_prepare_pipeline(monkeypatch, tmp_path):
    # Confirma que --dataset-type ragqa chega à função de preparação usada pelo train.py.
    try:
        import colorama  # noqa: F401
    except ImportError:
        colorama_stub = types.ModuleType("colorama")
        class _Colors:
            def __getattr__(self, _name): return ""
        colorama_stub.Fore = _Colors(); colorama_stub.Style = _Colors()
        colorama_stub.init = lambda **_kwargs: None
        monkeypatch.setitem(sys.modules, "colorama", colorama_stub)
    import train
    raw = tmp_path / "ragqa.jsonl"
    raw.write_text(json.dumps({"instruction": "Oi", "input": "", "output": "Olá"}) + "\n", encoding="utf-8")
    seen = {}
    monkeypatch.setattr(train, "project_path", lambda value: raw if str(value) == "custom.jsonl" else tmp_path / str(value))
    monkeypatch.setattr(train, "raw_path_for", lambda kind: raw)
    monkeypatch.setattr(train, "prepare_if_needed", lambda path, config, **kw: (seen.update(kw) or True, {
        "vocab_size": 12, "train_tokens": 8, "val_tokens": 3, "valid_examples": 1,
        "discarded_examples": 0, "raw_bytes": raw.stat().st_size}))
    args = types.SimpleNamespace(use_existing_data=False, source="local", dataset_type="ragqa",
                                 local_file=None, dataset=None, size_mb=None, split="train", dataset_config=None)
    train._select_and_prepare_data(args, {"hf_default_size_mb": 10})
    assert seen["dataset_type"] == "ragqa"


def test_prepare_only_stops_before_training(monkeypatch):
    import train
    monkeypatch.setattr(sys, "argv", ["train.py", "--prepare-only", "--source", "local", "--dataset-type", "ragqa"])
    monkeypatch.setattr(train, "load_config", lambda _path: {"max_iters": 1})
    calls = []
    monkeypatch.setattr(train, "_select_and_prepare_data", lambda *a, **kw: calls.append("prepared"))
    monkeypatch.setattr(train, "log", lambda *a, **kw: None)
    assert train.train() == 0
    assert calls == ["prepared"]


def test_ragqa_uses_same_question_answer_format_as_chat(tmp_path, monkeypatch):
    raw = tmp_path / "ragqa.jsonl"
    raw.write_text(json.dumps({
        "instruction": "###PERGUNTA\nQual é a capital?",
        "input": "",
        "output": "###RESPOSTA\nParis.",
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    monkeypatch.setattr(data, "project_path", lambda value: __import__("pathlib").Path(value))
    # O serializador compartilhado também precisa normalizar marcadores embutidos nos campos.
    rendered = data.example_to_text({"instruction": "###PERGUNTA\nQual é a capital?",
                                     "input": "", "output": "###RESPOSTA\nParis."}, "ragqa")
    assert rendered == "Pergunta: Qual é a capital?\nResposta: Paris.\n\n"


def test_clean_response_removes_ragqa_markers():
    from src.prompting import clean_response
    assert clean_response("###RESPOSTA\nParis.\n###PERGUNTA\nPróxima") == "Paris."
    assert clean_response("### Resposta: Paris é a capital.\n### Instruções:\ncontinuação") == "Paris é a capital."


def test_dataset_profile_points_to_madras1_rag_qa():
    spec = data.dataset_spec("ragqa")
    assert spec["dataset"] == "Madras1/rag-qa-fulltext-ptbr"
    assert spec["label"] == "Madras1 RAG QA Fulltext PT-BR"
    assert spec["raw_filename"] == "raw/rag-qa-fulltext-ptbr/train.jsonl"
    assert spec["config"] == "qa_pairs"


def test_ragqa_huggingface_default_uses_qa_pairs_config(monkeypatch, tmp_path):
    import train
    captured = {}
    monkeypatch.setattr(train, "raw_path_for", lambda _kind: tmp_path / "train.jsonl")
    monkeypatch.setattr(train, "stream_dataset_to_file", lambda *args, **kwargs: (captured.update(kwargs) or {"records": 1, "size_bytes": 1, "duplicates": 0, "skipped": 0}))
    monkeypatch.setattr(train, "prepare_if_needed", lambda *args, **kwargs: (True, {"vocab_size": 1, "train_tokens": 2, "val_tokens": 2, "valid_examples": 1, "discarded_examples": 0, "raw_bytes": 1}))
    args = types.SimpleNamespace(use_existing_data=False, source="hf", dataset_type="ragqa", local_file=None, dataset=None, size_mb=1, split="train", dataset_config=None)
    train._select_and_prepare_data(args, {"hf_default_size_mb": 1})
    assert captured["dataset_config"] == "qa_pairs"


def test_legacy_canarim_identifier_normalizes_to_public_ragqa_name():
    assert data.normalize_dataset_type("canarim") == "ragqa"
    assert data.dataset_spec("canarim")["dataset"] == "Madras1/rag-qa-fulltext-ptbr"


def test_tui_emits_ragqa_not_legacy_canarim_identifier():
    import tui
    assert "ragqa" in tui.DATASET_TYPES
    assert "canarim" not in tui.DATASET_TYPES


def test_streaming_uses_unique_temporary_paths_for_concurrent_windows_runs(tmp_path, monkeypatch):
    destination = tmp_path / "train.jsonl"
    seen = []

    def fake_stream(dataset_id, target, temporary, byte_limit, split, dataset_config):
        seen.append(temporary)
        return {"dataset": dataset_id, "output": str(target), "size_bytes": 0,
                "records": 0, "skipped": 0, "duplicates": 0}

    monkeypatch.setattr(data, "_stream_ragqa", fake_stream)
    for _ in range(2):
        data.stream_dataset_to_file("Madras1/rag-qa-fulltext-ptbr", destination,
                                    dataset_type="ragqa", size_mb=1)
    assert len(seen) == 2
    assert seen[0] != seen[1]
    assert all(path.parent == destination.parent for path in seen)
