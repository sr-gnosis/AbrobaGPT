"""Montagem de prompt e limpeza de resposta para o chat (sem dependência de torch).

O modelo é treinado em pares no formato::

    Pergunta: ...
    Resposta: ...

    Pergunta: ...

Por isso o chat usa exatamente esse formato e interrompe a geração quando o
modelo termina a resposta (linha em branco) ou começa a inventar a próxima
pergunta.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence
import re

QUESTION_PREFIX = "Pergunta:"
ANSWER_PREFIX = "Resposta:"

# Sequências que marcam o fim de uma resposta.
CHAT_STOP_SEQUENCES: tuple[str, ...] = (
    # Não pare em uma linha em branco isolada: respostas válidas podem ter parágrafos.
    f"\n{QUESTION_PREFIX}",
    "\n###PERGUNTA", "\n### PERGUNTA", "\n###QUESTÃO", "\n### QUESTÃO",
    "\n### Instruções:", "\n### Instrução:",
)


def format_turn(question: str, answer: str) -> str:
    return f"{QUESTION_PREFIX} {question.strip()}\n{ANSWER_PREFIX} {answer.strip()}\n\n"


def build_prompt(
    turns: Sequence[tuple[str, str]],
    question: str,
    count_tokens: Callable[[str], int],
    max_prompt_tokens: int,
) -> str:
    """Monta o prompt com o histórico mais recente que couber no orçamento de tokens.

    A pergunta atual sempre entra; turnos antigos são descartados do mais velho
    para o mais novo até caber em ``max_prompt_tokens``.
    """
    current = f"{QUESTION_PREFIX} {question.strip()}\n{ANSWER_PREFIX}"
    used = count_tokens(current)
    kept: list[str] = []
    for old_question, old_answer in reversed(turns):
        block = format_turn(old_question, old_answer)
        cost = count_tokens(block)
        if used + cost > max_prompt_tokens:
            break
        kept.append(block)
        used += cost
    return "".join(reversed(kept)) + current


def cut_at_stop(text: str, stops: Iterable[str]) -> str:
    """Corta ``text`` na primeira ocorrência de qualquer sequência de parada."""
    end = len(text)
    for stop in stops:
        if not stop:
            continue
        position = text.find(stop)
        if position != -1:
            end = min(end, position)
    return text[:end]


def clean_response(text: str, stops: Iterable[str] = CHAT_STOP_SEQUENCES) -> str:
    """Remove marcadores de continuação que alguns checkpoints antigos aprenderam."""
    cleaned = cut_at_stop(text, stops).strip()
    # Se a geração começar pelo cabeçalho da resposta, ele não é conteúdo para o usuário.
    cleaned = re.sub(
        r"(?is)^\s*(?:#{2,}\s*(?:RESPOSTA|ANSWER)\s*:?\s*|Resposta\s*:\s*)",
        "", cleaned,
    )
    # Também interrompe se o modelo começar a escrever o próximo turno sem quebra de linha.
    cleaned = re.split(
        r"(?i)#{2,}\s*(?:PERGUNTA|QUESTÃO|QUESTAO|INSTRUÇÕES?|QUESTION)\s*:?",
        cleaned, maxsplit=1,
    )[0]
    return cleaned.strip()


def make_stop_fn(decode: Callable[[list[int]], str], stops: Sequence[str]) -> Callable[[list[int]], bool]:
    """Detecta marcadores incrementalmente, decodificando apenas tokens novos.

    O tokenizer BPE deste projeto decodifica por concatenação das peças, então
    a cauda acumulada basta para detectar marcadores que cruzam fronteiras de tokens.
    """
    active = tuple(stop for stop in stops if stop)
    tail_size = max((len(stop) for stop in active), default=1)
    seen = 0
    tail = ""

    def stop_fn(new_ids: list[int]) -> bool:
        nonlocal seen, tail
        if len(new_ids) < seen:
            # Também permite reutilização segura caso o chamador reinicie a sequência.
            seen, tail = 0, ""
        for token_id in new_ids[seen:]:
            piece = decode([token_id])
            tail = (tail + piece)[-tail_size:]
            if any(stop in tail for stop in active):
                seen = len(new_ids)
                return True
        seen = len(new_ids)
        return False

    return stop_fn
