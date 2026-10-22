"""Avaliação reproduzível de checkpoints do AbrobaGPT.

As métricas daqui são sinais diagnósticos, não uma medida de "inteligência".
O módulo usa exatamente o carregamento e a geração de ``src.inference``.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any, Callable, Iterable, Mapping, Optional

import torch

from src.inference import InferenceError, load_inference_components
from src.prompting import CHAT_STOP_SEQUENCES, clean_response, make_stop_fn
from src.utils import CheckpointError, project_path

PRESET_DIR = project_path("evaluation_presets")
REPORT_DIR = project_path("evaluation_reports")
DEFAULT_WEIGHTS = {"reference": .35, "instruction": .25, "structure": .20, "repetition": .15, "diversity": .05}


class EvaluationError(RuntimeError):
    pass


class EvaluationCancelled(EvaluationError):
    pass


def _atomic_json(value: Mapping[str, Any], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, destination)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise EvaluationError(f"Não foi possível salvar relatório: {exc}") from exc
    return destination


def list_presets(directory: Path = PRESET_DIR) -> list[Path]:
    return sorted(directory.glob("*.json")) if directory.exists() else []


def load_examples(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Preset inválido '{source}': {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("tests"), list):
        raise EvaluationError("Preset deve conter um objeto com a lista 'tests'.")
    examples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in data["tests"]:
        if not isinstance(item, dict):
            raise EvaluationError("Cada teste do preset deve ser um objeto.")
        identifier, prompt = item.get("id"), item.get("prompt")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise EvaluationError("Cada teste precisa de um id estável e único.")
        if not isinstance(prompt, str) or not prompt.strip():
            raise EvaluationError(f"Teste '{identifier}' não possui prompt.")
        seen.add(identifier)
        examples.append(item)
    return data, examples


def export_preset(source: str | Path, destination: str | Path) -> Path:
    metadata, _ = load_examples(source)
    return _atomic_json(metadata, Path(destination))


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", value.casefold())).strip()


def _word_ngrams(text: str, n: int) -> list[tuple[str, ...]]:
    words = normalize_text(text).split()
    return [tuple(words[index:index + n]) for index in range(len(words) - n + 1)]


def repetition_metrics(text: str) -> dict[str, float]:
    grams = _word_ngrams(text, 2) + _word_ngrams(text, 3)
    repeated = sum(count - 1 for count in Counter(grams).values() if count > 1)
    sentences = [normalize_text(part) for part in re.split(r"[.!?\n]+", text) if normalize_text(part)]
    duplicate_sentences = max(0, len(sentences) - len(set(sentences)))
    denominator = max(1, len(grams) + len(sentences))
    rate = min(1.0, (repeated + duplicate_sentences * 2) / denominator)
    return {"repeated_ngrams": float(repeated), "duplicate_sentences": float(duplicate_sentences), "rate": rate}


def structural_metrics(text: str, criteria: Mapping[str, Any]) -> dict[str, Any]:
    stripped = text.strip()
    words = normalize_text(stripped).split()
    min_words = int(criteria.get("min_words", 1))
    max_words = criteria.get("max_words")
    abrupt = bool(stripped and len(words) >= 4 and stripped[-1].isalnum())
    unwanted = any(marker.casefold() in stripped.casefold() for marker in criteria.get("forbid", []))
    ok = bool(stripped) and len(words) >= min_words and (max_words is None or len(words) <= int(max_words)) and not unwanted
    return {"ok": ok, "empty": not bool(stripped), "word_count": len(words), "abrupt": abrupt, "forbidden_pattern": unwanted}


def objective_metrics(text: str, example: Mapping[str, Any]) -> dict[str, float]:
    criteria = example.get("criteria") if isinstance(example.get("criteria"), dict) else {}
    normalized = normalize_text(text)
    reference = example.get("reference")
    reference_score: Optional[float] = None
    if isinstance(reference, str) and reference.strip():
        mode = str(criteria.get("reference_mode", "contains"))
        expected = normalize_text(reference)
        if mode == "exact":
            reference_score = float(normalized == expected)
        elif mode == "keywords":
            expected_words = [word for word in expected.split() if len(word) > 2]
            reference_score = sum(word in normalized for word in expected_words) / max(1, len(expected_words))
        else:
            reference_score = float(expected in normalized or normalized in expected)
    required = [normalize_text(str(item)) for item in criteria.get("must_include", [])]
    format_regex = criteria.get("format_regex")
    instruction_checks: list[bool] = [term in normalized for term in required]
    if isinstance(format_regex, str):
        try:
            instruction_checks.append(re.fullmatch(format_regex, text.strip(), flags=re.DOTALL) is not None)
        except re.error as exc:
            raise EvaluationError(f"Regex inválida no teste '{example.get('id')}': {exc}") from exc
    instruction_score = sum(instruction_checks) / len(instruction_checks) if instruction_checks else 1.0
    structure = structural_metrics(text, criteria)
    repetition = repetition_metrics(text)
    return {"reference": reference_score if reference_score is not None else 0.0, "reference_applicable": float(reference_score is not None), "instruction": instruction_score, "structure": float(structure["ok"]), "repetition": repetition["rate"], **structure, **{f"repetition_{key}": value for key, value in repetition.items()}}


def _mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return sum(values) / len(values) if values else None


def _generation_text(model: Any, tokenizer: Any, device: torch.device, prompt: str, settings: Mapping[str, Any], cancel: Optional[Event]) -> str:
    if cancel and cancel.is_set():
        raise EvaluationCancelled("Avaliação cancelada pelo usuário.")
    encoded = tokenizer.encode(prompt)
    if not encoded:
        raise EvaluationError("Prompt não contém tokens após tokenização.")
    max_tokens = int(settings.get("max_tokens", 80))
    stop_fn = make_stop_fn(tokenizer.decode, CHAT_STOP_SEQUENCES)
    generated = model.generate(torch.tensor([encoded], dtype=torch.long, device=device), max_new_tokens=max_tokens,
        temperature=float(settings.get("temperature", .75)), top_k=settings.get("top_k"), greedy=bool(settings.get("greedy", False)),
        repetition_penalty=float(settings.get("repetition_penalty", 1.0)), stop_fn=stop_fn)
    if cancel and cancel.is_set():
        raise EvaluationCancelled("Avaliação cancelada pelo usuário.")
    return clean_response(tokenizer.decode(generated[0, len(encoded):].tolist()))


def _validation_loss(model: Any, tokenizer: Any, device: torch.device, config_path: str, batches: int) -> tuple[Optional[float], Optional[str]]:
    """Loss só é emitida para os dados processados cujo tokenizer é o carregado."""
    try:
        from src.data import load_processed_data
        active_tokenizer, _, val_array, metadata = load_processed_data()
        if active_tokenizer.fingerprint != tokenizer.fingerprint:
            return None, "tokenizer dos dados de validação não confere com o checkpoint"
        block = min(int(model.block_size), len(val_array) - 1)
        if block < 1:
            return None, "dados de validação insuficientes"
        losses: list[float] = []
        model.eval()
        # Inícios fixos: não introduz aleatoriedade na comparação entre checkpoints.
        for start in [int(index * (len(val_array) - block - 1) / max(1, batches - 1)) for index in range(max(1, batches))]:
            x = torch.tensor(val_array[start:start + block], dtype=torch.long, device=device).unsqueeze(0)
            y = torch.tensor(val_array[start + 1:start + block + 1], dtype=torch.long, device=device).unsqueeze(0)
            _, loss = model(x, y)
            assert loss is not None
            losses.append(float(loss.item()))
        return _mean(losses), None
    except Exception as exc:  # dados opcionais nunca tornam a avaliação de geração inválida
        return None, f"loss de validação indisponível: {exc}"


def evaluate_checkpoints(checkpoints: Iterable[str | Path], preset_path: str | Path, settings: Optional[Mapping[str, Any]] = None, *,
                         cancel_event: Optional[Event] = None, progress: Optional[Callable[[dict[str, Any]], None]] = None) -> dict[str, Any]:
    settings = dict(settings or {})
    metadata, examples = load_examples(preset_path)
    limit = max(1, min(int(settings.get("questions", len(examples))), len(examples)))
    examples = examples[:limit]
    generations = max(1, int(settings.get("generations", 1)))
    paths = [Path(path) for path in checkpoints]
    if not paths:
        raise EvaluationError("Selecione ao menos um checkpoint.")
    seed = settings.get("seed")
    if seed is not None:
        random.seed(int(seed)); torch.manual_seed(int(seed))
    total = len(paths) * len(examples) * generations
    complete = 0
    outcomes: list[dict[str, Any]] = []
    for checkpoint_index, checkpoint in enumerate(paths, 1):
        if cancel_event and cancel_event.is_set(): raise EvaluationCancelled("Avaliação cancelada pelo usuário.")
        try:
            model, tokenizer, device, loaded = load_inference_components(checkpoint_path=str(checkpoint), allow_legacy_checkpoint=bool(settings.get("allow_legacy", False)))
        except (InferenceError, CheckpointError, RuntimeError) as exc:
            outcomes.append({"checkpoint": str(checkpoint), "error": str(exc), "results": [], "metrics": {}})
            continue
        per_test: list[dict[str, Any]] = []
        for example_index, example in enumerate(examples, 1):
            responses: list[dict[str, Any]] = []
            for generation_index in range(1, generations + 1):
                if seed is not None: torch.manual_seed(int(seed) + complete)
                text = _generation_text(model, tokenizer, device, str(example["prompt"]), settings, cancel_event)
                responses.append({"generation": generation_index, "text": text, "metrics": objective_metrics(text, example)})
                complete += 1
                if progress: progress({"completed": complete, "total": total, "checkpoint": checkpoint.name, "checkpoint_index": checkpoint_index, "checkpoint_total": len(paths), "question": str(example["id"]), "question_index": example_index, "question_total": len(examples)})
            per_test.append({"id": example["id"], "category": example.get("category", metadata.get("category", "custom")), "prompt": example["prompt"], "reference": example.get("reference"), "criteria": example.get("criteria", {}), "generations": responses})
        all_generations = [response for test in per_test for response in test["generations"]]
        metric = lambda key: _mean(float(item["metrics"][key]) for item in all_generations)
        diversity = _mean([len({normalize_text(item["text"]) for item in test["generations"]}) / len(test["generations"]) for test in per_test]) or 1.0
        val_loss, val_note = _validation_loss(model, tokenizer, device, "config/base_config.json", int(settings.get("validation_batches", 4))) if settings.get("validation_loss") else (None, None)
        metrics = {"reference_match": metric("reference"), "instruction_following": metric("instruction"), "structural_quality": metric("structure"), "repetition_rate": metric("repetition"), "diversity": diversity, "validation_loss": val_loss, "validation_note": val_note}
        weights = {**DEFAULT_WEIGHTS, **dict(settings.get("weights", {}))}
        metrics["overall_score"] = sum(float(weights[key]) * float({"reference": metrics["reference_match"], "instruction": metrics["instruction_following"], "structure": metrics["structural_quality"], "repetition": 1 - metrics["repetition_rate"], "diversity": metrics["diversity"]}[key]) for key in DEFAULT_WEIGHTS)
        outcomes.append({"checkpoint": str(loaded), "iteration": None, "error": None, "results": per_test, "metrics": metrics})
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    eligible = [outcome for outcome in outcomes if not outcome["error"]]
    recommended = max(eligible, key=lambda item: item["metrics"]["overall_score"])["checkpoint"] if eligible else None
    return {"format_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "preset": {"path": str(preset_path), "name": metadata.get("name"), "category": metadata.get("category")}, "settings": settings, "questions": [item["id"] for item in examples], "limitations": ["Pontuação geral é configurável e não mede inteligência geral.", "Loss só é comparável quando usa o mesmo tokenizer e dados de validação ativos.", "Referências aceitam regras por teste; não há comparação literal universal."], "checkpoints": outcomes, "recommended_checkpoint": recommended}


def save_report(report: Mapping[str, Any], destination: str | Path | None = None) -> Path:
    if destination is None:
        destination = REPORT_DIR / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    return _atomic_json(report, Path(destination))


def copy_selected_checkpoint(checkpoint: str | Path, destination_dir: str | Path = "checkpoints/evaluated_best") -> Path:
    source = Path(checkpoint)
    if not source.exists(): raise EvaluationError(f"Checkpoint não encontrado: {source}")
    directory = project_path(destination_dir); directory.mkdir(parents=True, exist_ok=True)
    target = directory / source.name
    counter = 1
    while target.exists():
        target = directory / f"{source.stem}_{counter}{source.suffix}"; counter += 1
    shutil.copy2(source, target)
    return target
