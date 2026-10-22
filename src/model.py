"""Implementação do Transformer decoder-only usado pelo AbrobaGPT.

Os nomes dos módulos foram mantidos compatíveis com o checkpoint recuperado
(`transformer.wte`, `transformer.h.*.attn.key`, etc.).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from src.defaults import DEFAULT_TEMPERATURE

_HAS_SDPA = hasattr(F, "scaled_dot_product_attention")


class ModelConfigError(ValueError):
    """Configuração de modelo inválida."""


def _config_value(config: Mapping[str, Any], name: str, default: Any = None) -> Any:
    value = config.get(name, default)
    if value is None:
        raise ModelConfigError(f"A configuração obrigatória '{name}' não foi informada.")
    return value


class CausalSelfAttention(nn.Module):
    """Atenção causal multi-cabeça com projeções separadas e máscara registrada."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        n_embd = int(_config_value(config, "n_embd"))
        n_head = int(_config_value(config, "n_head"))
        block_size = int(_config_value(config, "block_size"))
        if n_head <= 0 or n_embd % n_head != 0:
            raise ModelConfigError("n_embd deve ser divisível por n_head.")

        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(float(config.get("dropout", 0.0)))
        self.resid_dropout = nn.Dropout(float(config.get("dropout", 0.0)))
        # O buffer persistente preserva a estrutura dos checkpoints originais.
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, channels = x.shape
        k = self.key(x).view(batch_size, time_steps, self.n_head, self.head_size).transpose(1, 2)
        q = self.query(x).view(batch_size, time_steps, self.n_head, self.head_size).transpose(1, 2)
        v = self.value(x).view(batch_size, time_steps, self.n_head, self.head_size).transpose(1, 2)

        if _HAS_SDPA:
            # Atenção fundida (flash/mem-efficient quando disponível). O buffer `tril`
            # continua registrado só para manter os checkpoints compatíveis.
            dropout_p = self.attn_dropout.p if self.training else 0.0
            output = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
        else:
            weights = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
            if time_steps <= self.tril.size(0):
                mask = self.tril[:time_steps, :time_steps]
            else:
                mask = torch.tril(torch.ones(time_steps, time_steps, device=x.device, dtype=self.tril.dtype))
            weights = weights.masked_fill(mask == 0, float("-inf"))
            weights = F.softmax(weights, dim=-1)
            weights = self.attn_dropout(weights)
            output = weights @ v
        output = output.transpose(1, 2).contiguous().view(batch_size, time_steps, channels)
        return self.resid_dropout(self.proj(output))


class FeedForward(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        n_embd = int(_config_value(config, "n_embd"))
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(float(config.get("dropout", 0.0))),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        n_embd = int(_config_value(config, "n_embd"))
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.ffwd(self.ln2(x))


class GPT(nn.Module):
    """GPT compacto. `forward` retorna logits achatados para compatibilidade legada."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.config = dict(config)
        vocab_size = int(_config_value(config, "vocab_size"))
        block_size = int(_config_value(config, "block_size"))
        n_embd = int(_config_value(config, "n_embd"))
        n_layer = int(_config_value(config, "n_layer"))
        if vocab_size <= 1:
            raise ModelConfigError("vocab_size deve ser maior que 1.")
        if block_size <= 0 or n_layer <= 0 or n_embd <= 0:
            raise ModelConfigError("block_size, n_layer e n_embd devem ser positivos.")

        self.block_size = block_size
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(vocab_size, n_embd),
                wpe=nn.Embedding(block_size, n_embd),
                drop=nn.Dropout(float(config.get("dropout", 0.0))),
                h=nn.ModuleList([Block(config) for _ in range(n_layer)]),
                ln_f=nn.LayerNorm(n_embd),
            )
        )
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)
        # Inicialização escalada das projeções que escrevem no fluxo residual (GPT-2).
        residual_std = 0.02 / math.sqrt(2 * n_layer)
        for name, parameter in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("ffwd.net.2.weight"):
                nn.init.normal_(parameter, mean=0.0, std=residual_std)
        # O checkpoint recuperado tem embeddings de entrada e saída amarrados.
        self.lm_head.weight = self.transformer.wte.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizer(self, learning_rate: float, weight_decay: float) -> torch.optim.AdamW:
        """AdamW com weight decay só nas matrizes de pesos (sem LayerNorm, bias e embeddings)."""
        decay: list[nn.Parameter] = []
        no_decay: list[nn.Parameter] = []
        for name, parameter in self.named_parameters():  # pesos amarrados aparecem uma vez só
            if not parameter.requires_grad:
                continue
            if parameter.ndim >= 2 and "wte" not in name and "wpe" not in name:
                decay.append(parameter)
            else:
                no_decay.append(parameter)
        groups = [
            {"params": decay, "weight_decay": float(weight_decay)},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=float(learning_rate))

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if idx.ndim != 2:
            raise ValueError("idx deve ter formato (batch, tempo).")
        batch_size, time_steps = idx.shape
        if time_steps == 0:
            raise ValueError("Não é possível executar o modelo com uma sequência vazia.")
        # A geração limita o contexto a block_size. Para forward direto de uma
        # sequência maior, reutilizamos posições de forma cíclica (modulo block_size).
        # Nota: Isso mantém a compatibilidade com checkpoints legados, mas em sequências
        # muito longas (> block_size), a informação posicional se repetirá.
        positions = torch.arange(0, time_steps, dtype=torch.long, device=idx.device) % self.block_size
        token_embeddings = self.transformer.wte(idx)
        position_embeddings = self.transformer.wpe(positions)[None, :, :]
        x = self.transformer.drop(token_embeddings + position_embeddings)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits_3d = self.lm_head(x)
        logits = logits_3d.view(batch_size * time_steps, -1)

        loss: Optional[torch.Tensor] = None
        if targets is not None:
            if targets.shape != idx.shape:
                raise ValueError("targets deve ter o mesmo formato de idx.")
            loss = F.cross_entropy(logits, targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: Optional[int] = None,
        stop_fn: Optional[Callable[[list[int]], bool]] = None,
        token_callback: Optional[Callable[[list[int]], None]] = None,
        diagnostics_callback: Optional[Callable[[dict[str, Any]], None]] = None,
        greedy: bool = False,
        repetition_penalty: float = 1.0,
        repetition_window: int = 64,
    ) -> torch.Tensor:
        """Amostra tokens autoregressivamente sem perder o contexto acumulado.

        ``stop_fn`` (opcional, só com batch de 1) recebe a lista de tokens gerados até
        agora e interrompe a geração quando devolve True.
        """
        if idx.ndim != 2 or idx.shape[0] == 0 or idx.shape[1] == 0:
            raise ValueError("O prompt deve ter ao menos um token e formato (batch, tempo).")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 0:
            raise ValueError("max_new_tokens deve ser um inteiro não negativo.")
        if not isinstance(temperature, (int, float)) or not math.isfinite(float(temperature)) or temperature <= 0:
            raise ValueError("temperature deve ser finita e maior que zero.")
        vocab_size = self.transformer.wte.num_embeddings
        if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0):
            raise ValueError("top_k deve ser um inteiro positivo quando informado.")
        if not isinstance(greedy, bool):
            raise ValueError("greedy deve ser booleano.")
        if not isinstance(repetition_penalty, (int, float)) or not math.isfinite(float(repetition_penalty)) or repetition_penalty < 1.0:
            raise ValueError("repetition_penalty deve ser finita e >= 1.0.")
        if isinstance(repetition_window, bool) or not isinstance(repetition_window, int) or repetition_window < 1:
            raise ValueError("repetition_window deve ser um inteiro positivo.")
        if greedy and top_k is not None and top_k != 1:
            # Greedy é explícito e determinístico; não combina com amostragem top-k.
            top_k = 1
        if stop_fn is not None and idx.shape[0] != 1:
            raise ValueError("stop_fn só é suportado com batch de tamanho 1.")
        prompt_length = idx.shape[1]

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits.view(idx_cond.size(0), idx_cond.size(1), vocab_size)[:, -1, :]
            raw_logits = logits.clone()
            # Repetição: penaliza tokens presentes na janela recente (inclui prompt).
            if repetition_penalty > 1.0:
                recent = idx[:, -repetition_window:]
                for batch_index in range(idx.shape[0]):
                    unique_ids = torch.unique(recent[batch_index])
                    selected = logits[batch_index, unique_ids]
                    logits[batch_index, unique_ids] = torch.where(selected > 0, selected / repetition_penalty, selected * repetition_penalty)
            penalized_logits = logits.clone()
            temperature_logits = logits / float(temperature)
            post_topk_logits = temperature_logits.clone()
            if top_k is not None:
                k = min(int(top_k), post_topk_logits.size(-1))
                _, top_indices = torch.topk(post_topk_logits, k, dim=-1)
                keep_mask = torch.zeros_like(post_topk_logits, dtype=torch.bool)
                keep_mask.scatter_(-1, top_indices, True)
                post_topk_logits = post_topk_logits.masked_fill(~keep_mask, float("-inf"))
            probabilities = F.softmax(post_topk_logits, dim=-1)
            next_token = torch.argmax(post_topk_logits, dim=-1, keepdim=True) if greedy else torch.multinomial(probabilities, num_samples=1)
            if diagnostics_callback is not None and idx.shape[0] == 1:
                def top_entries(values, probs=False):
                    count = min(5, values.shape[-1])
                    vals, ids = torch.topk(values[0], count)
                    if probs:
                        return [(int(t), float(v)) for t, v in zip(ids.tolist(), vals.tolist())]
                    p = F.softmax(values[0], dim=-1)
                    return [(int(t), float(p[t].item())) for t in ids.tolist()]
                raw_probs = F.softmax(raw_logits, dim=-1)
                temp_probs = F.softmax(temperature_logits, dim=-1)
                post_probs = probabilities
                # Mede repetições de n-gramas no contexto recente (n=2..5).
                repeated_ngrams = {}
                recent_ids = idx[0, -repetition_window:].tolist()
                for n in range(2, 6):
                    grams = [tuple(recent_ids[i:i+n]) for i in range(max(0, len(recent_ids)-n+1))]
                    repeated_ngrams[str(n)] = max(0, len(grams) - len(set(grams)))
                diagnostics_callback({
                    "repeated_ngram_counts": repeated_ngrams,
                    "top_candidates": top_entries(temperature_logits),
                    "top5_raw_logits": top_entries(raw_logits),
                    "top5_after_temperature": top_entries(temperature_logits),
                    "top5_after_top_k": top_entries(post_topk_logits),
                    "sampled_token_id": int(next_token[0, 0].item()),
                    "sampled_probability": float(post_probs[0, next_token[0, 0]].item()),
                    "greedy": greedy,
                    "top_k": top_k,
                    "repetition_penalty": float(repetition_penalty),
                    "step": int(idx.shape[1] - prompt_length + 1),
                })
            idx = torch.cat((idx, next_token), dim=1)
            generated_ids = idx[0, prompt_length:].tolist() if idx.shape[0] == 1 else []
            should_stop = stop_fn is not None and stop_fn(generated_ids)
            # Não exponha tokens que completam um marcador de parada ao callback.
            if not should_stop and token_callback is not None:
                token_callback(generated_ids)
            if should_stop:
                break
        return idx
