"""Carregamento seguro compartilhado por chat.py e generate.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch

from src.defaults import DEFAULT_TEMPERATURE
from src.model import GPT, ModelConfigError
from src.tokenizer import BPETokenizer, TokenizerError
from src.utils import (
    CheckpointError,
    checkpoint_compatibility,
    find_latest_compatible_checkpoint,
    get_device,
    inspect_checkpoint,
    load_checkpoint,
    load_config,
    project_path,
)


class InferenceError(RuntimeError):
    """Erro ao construir o modelo de inferência."""


def resolve_temperature(config_path: str, cli_value: Optional[float]) -> float:
    """Temperatura efetiva: argumento de linha de comando > base_config.json > padrão."""
    if cli_value is not None:
        return float(cli_value)
    try:
        return float(load_config(config_path)["temperature"])
    except (ValueError, KeyError, OSError):
        return DEFAULT_TEMPERATURE


def _load_tokenizer(path: str | Path) -> BPETokenizer:
    try:
        return BPETokenizer().load(project_path(path))
    except TokenizerError as exc:
        raise InferenceError(str(exc)) from exc


def _select_checkpoint(
    requested: Optional[str], config: dict[str, Any], tokenizer: BPETokenizer, allow_legacy: bool
) -> Path:
    if requested:
        candidate = project_path(requested)
        if not candidate.exists():
            raise InferenceError(f"Checkpoint não encontrado: '{candidate}'.")
        return candidate

    best = project_path("checkpoints/best_model.pt")
    if best.exists():
        try:
            summary = inspect_checkpoint(best)
            compatible, _ = checkpoint_compatibility(summary, config, tokenizer, allow_legacy=allow_legacy)
            if compatible:
                return best
        except CheckpointError:
            pass
    latest, _ = find_latest_compatible_checkpoint(
        "checkpoints", config, tokenizer, allow_legacy=allow_legacy
    )
    if latest is None:
        raise InferenceError(
            "Nenhum checkpoint compatível encontrado. Treine o modelo ou informe --ckpt."
        )
    return latest


def load_inference_components(
    *,
    config_path: str = "config/base_config.json",
    tokenizer_path: str = "data/processed/tokenizer.json",
    checkpoint_path: Optional[str] = None,
    allow_legacy_checkpoint: bool = False,
) -> tuple[GPT, BPETokenizer, torch.device, Path]:
    """Carrega tokenizer, metadados, arquitetura e pesos com validação estrita."""
    try:
        config = load_config(config_path)
        tokenizer = _load_tokenizer(tokenizer_path)
        config["vocab_size"] = tokenizer.vocab_size
        checkpoint = _select_checkpoint(checkpoint_path, config, tokenizer, allow_legacy_checkpoint)
        summary = inspect_checkpoint(checkpoint)
        if not summary["legacy"]:
            saved_config = summary["model_config"]
            # A arquitetura embutida no checkpoint é a fonte de verdade para
            # inferência; demais opções continuam vindo do config atual.
            config.update(saved_config)
        compatible, reason = checkpoint_compatibility(
            summary, config, tokenizer, allow_legacy=allow_legacy_checkpoint
        )
        if not compatible:
            raise InferenceError(f"Checkpoint incompatível: {reason}.")
        device = get_device(config["device"])
        model = GPT(config).to(device)
        load_checkpoint(model, None, checkpoint, map_location=device)
        model.eval()
        return model, tokenizer, device, checkpoint
    except (CheckpointError, ModelConfigError, ValueError) as exc:
        raise InferenceError(str(exc)) from exc
