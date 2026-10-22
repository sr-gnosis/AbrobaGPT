"""Presets de learning rate ("começo baixo/médio/alto") sem dependência de torch.

Compartilhado por train.py (aplica) e tui.py (mostra). Para ajustar os valores
ou os nomes, edite só este arquivo.
"""

from __future__ import annotations

from typing import Any

# peak = LR de pico | min = LR final do cosseno | warmup_fraction = % das iterações
LR_PRESETS: dict[str, dict[str, Any]] = {
    "baixo": {"label": "Começo baixo", "peak": 1e-4, "min": 1e-5, "warmup_fraction": 0.10},
    "medio": {"label": "Começo médio", "peak": 5e-4, "min": 5e-5, "warmup_fraction": 0.05},
    "alto":  {"label": "Começo alto",  "peak": 1e-3, "min": 1e-4, "warmup_fraction": 0.05},
}

WARMUP_MIN = 50
WARMUP_MAX = 2000


def _fmt(value: float) -> str:
    return f"{value:.0e}".replace("e-0", "e-")


def resolve_lr_preset(key: str, max_iters: int) -> dict[str, Any]:
    """Converte o preset em learning_rate/min_learning_rate/warmup_iters.

    O warmup é proporcional ao total de iterações (nunca passa de metade dele),
    para que treinos curtos realmente cheguem ao LR de pico.
    """
    if key not in LR_PRESETS:
        raise ValueError(f"Preset de LR desconhecido: '{key}'. Use: {', '.join(LR_PRESETS)}.")
    preset = LR_PRESETS[key]
    total = max(1, int(max_iters))
    warmup = int(round(total * preset["warmup_fraction"]))
    warmup = min(max(warmup, WARMUP_MIN), WARMUP_MAX)
    warmup = min(warmup, max(1, total // 2))
    return {
        "learning_rate": float(preset["peak"]),
        "min_learning_rate": float(preset["min"]),
        "warmup_iters": warmup,
    }


def describe_preset(key: str) -> str:
    preset = LR_PRESETS[key]
    return f"pico {_fmt(preset['peak'])} · warmup {int(preset['warmup_fraction'] * 100)}%"
