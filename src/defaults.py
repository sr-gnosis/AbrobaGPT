"""Valores padrão compartilhados (sem dependência de torch).

As chaves opcionais do ``config/base_config.json`` ficam aqui: se não estiverem
no arquivo, assumem o valor abaixo. Assim configs antigos continuam válidos.
"""

from __future__ import annotations

from typing import Any

# Temperatura padrão de geração (chat, generate, TUI e GPT.generate).
DEFAULT_TEMPERATURE = 0.5

# Quantos checkpoints periódicos (ckpt_*.pt) criados pela execução de treino
# atual são mantidos em disco. 0 = manter todos.
DEFAULT_KEEP_CHECKPOINTS = 5

OPTIONAL_CONFIG_DEFAULTS: dict[str, Any] = {
    "temperature": DEFAULT_TEMPERATURE,
    "keep_checkpoints": DEFAULT_KEEP_CHECKPOINTS,
    # Scheduler: true preserva o comportamento cosseno usado pelo train.py.
    "cosine_decay": True,
}
