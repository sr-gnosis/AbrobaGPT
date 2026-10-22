"""Configuração, dispositivo e checkpoints compartilhados pelo AbrobaGPT."""

from __future__ import annotations

import json
import os
from pathlib import Path
import pickle
import re
from typing import Any, Mapping, Optional

import torch

from src.defaults import OPTIONAL_CONFIG_DEFAULTS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE_KEYS = ("n_layer", "n_embd", "n_head", "block_size", "dropout", "vocab_size")
INFERRED_ARCHITECTURE_KEYS = ("n_layer", "n_embd", "block_size", "vocab_size")

CONFIG_KEYS = frozenset({
    "n_layer",
    "n_embd",
    "n_head",
    "block_size",
    "dropout",
    "batch_size",
    "max_iters",
    "eval_interval",
    "eval_iters",
    "save_interval",
    "learning_rate",
    "min_learning_rate",
    "warmup_iters",
    "weight_decay",
    "grad_clip",
    "early_stopping",
    "patience",
    "early_stopping_warmup_iters",
    "validation_fraction",
    "tokenizer_vocab_size",
    "tokenizer_training_chars",
    "tokenizer_min_pair_frequency",
    "hf_default_size_mb",
    "amp",
    "device",
    "seed",
})


# Chaves opcionais (com valor padrão em src/defaults.py): podem faltar no JSON.
OPTIONAL_CONFIG_KEYS = frozenset(OPTIONAL_CONFIG_DEFAULTS)


class ConfigError(ValueError):
    """Arquivo de configuração ausente ou inválido."""


class CheckpointError(RuntimeError):
    """Checkpoint ilegível, corrompido ou estruturalmente inválido."""


def project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _validate_config(config: Mapping[str, Any]) -> None:
    missing = CONFIG_KEYS - set(config)
    if missing:
        raise ConfigError(f"Chaves de configuração ausentes: {', '.join(sorted(missing))}.")
    integer_keys = (
        "n_layer",
        "n_embd",
        "n_head",
        "block_size",
        "batch_size",
        "max_iters",
        "eval_interval",
        "eval_iters",
        "save_interval",
        "warmup_iters",
        "patience",
        "early_stopping_warmup_iters",
        "tokenizer_vocab_size",
        "tokenizer_training_chars",
        "tokenizer_min_pair_frequency",
    )
    for key in integer_keys:
        if int(config[key]) < 0:
            raise ConfigError(f"'{key}' não pode ser negativo.")
    for key in ("n_layer", "n_embd", "n_head", "block_size", "batch_size", "max_iters", "eval_interval", "save_interval"):
        if int(config[key]) == 0:
            raise ConfigError(f"'{key}' deve ser maior que zero.")
    if int(config["n_embd"]) % int(config["n_head"]) != 0:
        raise ConfigError("n_embd deve ser divisível por n_head.")
    if not 0 <= float(config["dropout"]) < 1:
        raise ConfigError("dropout deve estar no intervalo [0, 1).")
    if float(config["learning_rate"]) <= 0 or float(config["min_learning_rate"]) < 0:
        raise ConfigError("As taxas de aprendizado são inválidas.")
    if not isinstance(config["early_stopping"], bool):
        raise ConfigError("'early_stopping' deve ser true ou false.")
    if not 0 < float(config["validation_fraction"]) < 0.5:
        raise ConfigError("validation_fraction deve estar entre 0 e 0.5.")
    if str(config["device"]).lower() not in {"auto", "cpu", "cuda"}:
        raise ConfigError("device deve ser 'auto', 'cpu' ou 'cuda'.")
    if float(config.get("temperature", OPTIONAL_CONFIG_DEFAULTS["temperature"])) <= 0:
        raise ConfigError("temperature deve ser maior que zero.")
    if int(config.get("keep_checkpoints", OPTIONAL_CONFIG_DEFAULTS["keep_checkpoints"])) < 0:
        raise ConfigError("keep_checkpoints não pode ser negativo (0 mantém todos).")


def load_config(path: str | Path = "config/base_config.json") -> dict[str, Any]:
    source = project_path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            supplied = json.load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"Arquivo de configuração não encontrado: '{source}'.") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Não foi possível ler a configuração '{source}': {exc}") from exc
    if not isinstance(supplied, dict):
        raise ConfigError("A configuração deve ser um objeto JSON.")
    unknown = set(supplied) - CONFIG_KEYS - OPTIONAL_CONFIG_KEYS - {"vocab_size"}
    if unknown:
        raise ConfigError(f"Chaves de configuração desconhecidas: {', '.join(sorted(unknown))}.")
    config = dict(supplied)
    # O vocab_size nunca vem do arquivo: ele é determinado pelo tokenizer ativo.
    config.pop("vocab_size", None)
    for key, default in OPTIONAL_CONFIG_DEFAULTS.items():
        config.setdefault(key, default)
    _validate_config(config)
    return config


def get_device(requested: str = "auto") -> torch.device:
    requested = str(requested).lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in ARCHITECTURE_KEYS if key not in config]
    if missing:
        raise ConfigError(f"Configuração do modelo incompleta: {', '.join(missing)}.")
    return {
        "n_layer": int(config["n_layer"]),
        "n_embd": int(config["n_embd"]),
        "n_head": int(config["n_head"]),
        "block_size": int(config["block_size"]),
        "dropout": float(config["dropout"]),
        "vocab_size": int(config["vocab_size"]),
    }


def tokenizer_metadata(tokenizer: Any) -> dict[str, Any]:
    return {
        "vocab_size": int(tokenizer.vocab_size),
        "fingerprint": str(tokenizer.fingerprint),
        "format": "abrobabpe",
    }


def _torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    source = project_path(path)
    try:
        payload = torch.load(source, map_location=map_location, weights_only=False)
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as exc:  # type: ignore[name-defined]
        raise CheckpointError(f"Não foi possível carregar checkpoint '{source}': {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state_dict"), dict):
        raise CheckpointError(f"Checkpoint '{source}' não contém model_state_dict válido.")
    return payload


def infer_model_config_from_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, int]:
    try:
        vocab_size, n_embd = state_dict["transformer.wte.weight"].shape
        block_size, position_embd = state_dict["transformer.wpe.weight"].shape
    except (KeyError, ValueError, AttributeError) as exc:
        raise CheckpointError("Não foi possível inferir a arquitetura do checkpoint.") from exc
    if n_embd != position_embd:
        raise CheckpointError("Embeddings de token e posição incompatíveis no checkpoint.")
    layers = {
        int(match.group(1))
        for key in state_dict
        if (match := re.match(r"^transformer\.h\.(\d+)\.", key)) is not None
    }
    if not layers or layers != set(range(max(layers) + 1)):
        raise CheckpointError("Blocos Transformer ausentes ou não contíguos no checkpoint.")
    if tuple(state_dict.get("lm_head.weight", torch.empty(0)).shape) != (vocab_size, n_embd):
        raise CheckpointError("lm_head incompatível com embeddings no checkpoint.")
    return {
        "n_layer": len(layers),
        "n_embd": int(n_embd),
        "block_size": int(block_size),
        "vocab_size": int(vocab_size),
    }


def inspect_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = _torch_load(path)
    inferred = infer_model_config_from_state_dict(payload["model_state_dict"])
    stored = payload.get("model_config")
    tokenizer = payload.get("tokenizer")
    is_new_format = int(payload.get("format_version", 0)) >= 2
    if stored is not None and not isinstance(stored, dict):
        raise CheckpointError("model_config no checkpoint é inválido.")
    if tokenizer is not None and not isinstance(tokenizer, dict):
        raise CheckpointError("Metadados do tokenizer no checkpoint são inválidos.")
    return {
        "path": project_path(path),
        "payload": payload,
        "iteration": int(payload.get("iteration", 0)),
        "model_config": dict(stored) if stored else inferred,
        "inferred_model_config": inferred,
        "tokenizer": dict(tokenizer) if tokenizer else None,
        "legacy": not is_new_format,
    }


def checkpoint_compatibility(
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
    tokenizer: Any,
    *,
    allow_legacy: bool = False,
) -> tuple[bool, str]:
    expected = model_config(config)
    inferred = summary["inferred_model_config"]
    for key in INFERRED_ARCHITECTURE_KEYS:
        if int(inferred[key]) != int(expected[key]):
            return False, f"arquitetura incompatível em {key}: checkpoint={inferred[key]}, atual={expected[key]}"
    if summary.get("legacy"):
        if not allow_legacy:
            return False, "checkpoint legado sem fingerprint do tokenizer; não é seguro retomar/carregar"
        return True, "checkpoint legado aceito por --allow-legacy-checkpoint; tokenizer não pôde ser verificado"

    saved_config = summary.get("model_config") or {}
    for key in ARCHITECTURE_KEYS:
        if key not in saved_config or saved_config[key] != expected[key]:
            return False, f"model_config incompatível em {key}"
    saved_tokenizer = summary.get("tokenizer") or {}
    current_tokenizer = tokenizer_metadata(tokenizer)
    if saved_tokenizer.get("vocab_size") != current_tokenizer["vocab_size"]:
        return False, "vocab_size do checkpoint não confere com o tokenizer atual"
    if saved_tokenizer.get("fingerprint") != current_tokenizer["fingerprint"]:
        return False, "fingerprint do tokenizer não confere com o checkpoint"
    return True, "compatível"


def _atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise CheckpointError(f"Não foi possível salvar checkpoint '{destination}': {exc}") from exc


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    path: str | Path,
    *,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    config: Optional[Mapping[str, Any]] = None,
    tokenizer: Optional[Any] = None,
    best_val_loss: Optional[float] = None,
    patience_counter: Optional[int] = None,
) -> None:
    """Salva de forma atômica; os argumentos básicos preservam API legada."""
    payload: dict[str, Any] = {
        "format_version": 2 if config is not None and tokenizer is not None else 1,
        "iteration": int(iteration),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    if config is not None:
        payload["model_config"] = model_config(config)
    if tokenizer is not None:
        payload["tokenizer"] = tokenizer_metadata(tokenizer)
    if best_val_loss is not None:
        payload["best_val_loss"] = float(best_val_loss)
    if patience_counter is not None:
        payload["patience_counter"] = int(patience_counter)
    _atomic_torch_save(payload, project_path(path))


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str | Path,
    *,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    map_location: str | torch.device = "cpu",
) -> int:
    """Restaura estado e devolve a próxima iteração a executar."""
    payload = _torch_load(path, map_location=map_location)
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
        saved_optimizer = payload.get("optimizer_state_dict")
        if optimizer is not None and saved_optimizer is not None:
            if len(saved_optimizer.get("param_groups", [])) != len(optimizer.param_groups):
                # Checkpoints antigos usavam um único grupo (weight decay em tudo).
                # Os pesos são carregados normalmente; só o estado do Adam recomeça.
                print(
                    "Aviso: checkpoint salvo com outra organização de grupos do otimizador; "
                    "o estado do AdamW foi reiniciado (os pesos do modelo foram mantidos)."
                )
            else:
                optimizer.load_state_dict(saved_optimizer)
        if scheduler is not None and payload.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        if scaler is not None and payload.get("scaler_state_dict") is not None:
            scaler.load_state_dict(payload["scaler_state_dict"])
    except (RuntimeError, ValueError, KeyError) as exc:
        raise CheckpointError(f"Checkpoint incompatível ao restaurar '{project_path(path)}': {exc}") from exc
    return int(payload.get("iteration", 0))


def find_latest_compatible_checkpoint(
    checkpoint_dir: str | Path,
    config: Mapping[str, Any],
    tokenizer: Any,
    *,
    allow_legacy: bool = False,
) -> tuple[Optional[Path], list[str]]:
    directory = project_path(checkpoint_dir)
    if not directory.exists():
        return None, []
    candidates: list[tuple[int, float, Path]] = []
    rejected: list[str] = []
    for candidate in directory.glob("ckpt_*.pt"):
        try:
            summary = inspect_checkpoint(candidate)
            compatible, reason = checkpoint_compatibility(
                summary, config, tokenizer, allow_legacy=allow_legacy
            )
            if compatible:
                candidates.append((summary["iteration"], candidate.stat().st_mtime, candidate))
            else:
                rejected.append(f"{candidate.name}: {reason}")
        except (CheckpointError, OSError) as exc:
            rejected.append(f"{candidate.name}: {exc}")
    if not candidates:
        return None, rejected
    return max(candidates, key=lambda item: (item[0], item[1]))[2], rejected


def safe_checkpoint_path(
    desired_path: str | Path,
    config: Mapping[str, Any],
    tokenizer: Any,
) -> Path:
    """Evita sobrescrever checkpoint de outro tokenizer/arquitetura."""
    desired = project_path(desired_path)
    if not desired.exists():
        return desired
    try:
        summary = inspect_checkpoint(desired)
        compatible, _ = checkpoint_compatibility(summary, config, tokenizer)
        if compatible:
            return desired
    except CheckpointError:
        pass
    fingerprint = tokenizer_metadata(tokenizer)["fingerprint"][:10]
    return desired.with_name(f"{desired.stem}_{fingerprint}{desired.suffix}")
