"""Tokenizer BPE simples, serializável e independente de bibliotecas externas.

Ele mantém ``CharacterTokenizer`` como alias de compatibilidade com os scripts
de teste recuperados, mas os arquivos novos usam BPE e ``tokenizer.json``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import heapq
import json
from pathlib import Path
import re
from typing import Any, Iterable, Optional

class TokenizerError(ValueError):
    """Erro de treinamento, leitura ou uso do tokenizer."""

# Pré-tokenizadores. "legacy" separa o espaço em um token próprio (formato dos
# tokenizer.json antigos); "gpt2" cola o espaço à palavra seguinte (" casa"),
# o que reduz bastante a quantidade de tokens por texto. Os dois usam apenas
# `re` da biblioteca padrão e cobrem todos os caracteres (nada é descartado).
_PRETOKENIZERS: dict[str, "re.Pattern[str]"] = {
    "legacy": re.compile(r"\S+|\s+", re.UNICODE),
    "gpt2": re.compile(r" ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+", re.UNICODE),
}


class BPETokenizer:
    FORMAT_VERSION = 1
    UNKNOWN_TOKEN = "<UNK>"
    DEFAULT_PRETOKENIZER = "gpt2"

    def __init__(self) -> None:
        self.pretokenizer = self.DEFAULT_PRETOKENIZER
        self.token_to_id: dict[str, int] = {self.UNKNOWN_TOKEN: 0}
        self.id_to_token: list[str] = [self.UNKNOWN_TOKEN]
        self.merge_rules: list[tuple[str, str, str]] = []
        self.base_tokens: set[str] = set()
        self._merge_index: dict[tuple[str, str], tuple[int, str]] = {}
        self._encode_cache: dict[str, tuple[int, ...]] = {}

    @property
    def vocab(self) -> dict[str, int]:
        return self.token_to_id

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @property
    def fingerprint(self) -> str:
        payload: dict[str, Any] = {
            "format_version": self.FORMAT_VERSION,
            "tokens": self.id_to_token,
            "merges": self.merge_rules,
            "base_tokens": sorted(self.base_tokens),
        }
        if self.pretokenizer != "legacy":
            # Omitido no formato legado para os fingerprints antigos continuarem válidos.
            payload["pretokenizer"] = self.pretokenizer
        canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _reset(self, base_tokens: Iterable[str]) -> None:
        ordered = sorted(set(base_tokens))
        if self.UNKNOWN_TOKEN in ordered:
            ordered.remove(self.UNKNOWN_TOKEN)
        self.base_tokens = set(ordered)
        self.id_to_token = [self.UNKNOWN_TOKEN, *ordered]
        self.token_to_id = {token: index for index, token in enumerate(self.id_to_token)}
        self.merge_rules = []
        self._merge_index = {}
        self._encode_cache = {}

    def _rebuild_indexes(self) -> None:
        self._merge_index = {
            (left, right): (rank, merged)
            for rank, (left, right, merged) in enumerate(self.merge_rules)
        }
        self._encode_cache = {}

    def _chunks(self, text: str) -> Iterable[str]:
        yield from _PRETOKENIZERS[self.pretokenizer].findall(text)

    def _merge_pair(self, symbols: tuple[Any, ...], pair: tuple[Any, Any], merged: Any) -> tuple[Any, ...]:
        result: list[Any] = []
        index = 0
        n = len(symbols)
        while index < n:
            if index + 1 < n and symbols[index] == pair[0] and symbols[index + 1] == pair[1]:
                result.append(merged)
                index += 2
            else:
                result.append(symbols[index])
                index += 1
        return tuple(result)

    def _collect_chunks(self, text: str | Iterable[str], training_chars: Optional[int]) -> Counter:
        """Conta os chunks (palavras pré-tokenizadas) do corpus de treino do tokenizer."""
        counter: Counter = Counter()
        if isinstance(text, str):
            if not text:
                raise TokenizerError("O corpus para treinar o tokenizer está vazio.")
            if training_chars is not None and training_chars > 0:
                text = text[:training_chars]
            counter.update(self._chunks(text))
            return counter
        collected = 0
        for line in text:
            for chunk in self._chunks(line):
                counter[chunk] += 1
                collected += len(chunk)
            if training_chars and collected >= training_chars:
                break
        return counter

    @staticmethod
    def _merge_word(word: list[str], left: str, right: str, merged: str) -> list[str]:
        result: list[str] = []
        index = 0
        n = len(word)
        while index < n:
            if index + 1 < n and word[index] == left and word[index + 1] == right:
                result.append(merged)
                index += 2
            else:
                result.append(word[index])
                index += 1
        return result

    def fit(
        self,
        text: str | Iterable[str],
        vocab_size: Optional[int] = None,
        min_pair_frequency: int = 2,
        training_chars: Optional[int] = None,
        verbose: bool = True,
    ) -> "BPETokenizer":
        """Treina o BPE.

        As frequências de pares são mantidas de forma incremental (só as palavras
        que contêm o par fundido são recalculadas) e o par mais frequente sai de
        um heap, em vez de recontar o corpus inteiro a cada fusão.
        """
        chunk_counter = self._collect_chunks(text, training_chars)
        if not chunk_counter:
            raise TokenizerError("O corpus não contém tokens utilizáveis.")
        if verbose:
            print(
                f"[Tokenizer] {sum(chunk_counter.values()):,} chunks "
                f"({len(chunk_counter):,} únicos) coletados."
            )

        base_tokens = {char for chunk in chunk_counter for char in chunk}
        self._reset(base_tokens)
        if verbose:
            print(f"[Tokenizer] {len(self.id_to_token)} tokens base.")

        if vocab_size is None:
            return self
        if vocab_size < self.vocab_size:
            raise TokenizerError(f"vocab_size={vocab_size} é menor que o vocabulário básico.")
        if min_pair_frequency < 1:
            raise TokenizerError("min_pair_frequency deve ser pelo menos 1.")

        words: list[list[str]] = [list(chunk) for chunk in chunk_counter]
        counts: list[int] = list(chunk_counter.values())
        symbols = set(self.base_tokens)

        pair_freq: dict[tuple[str, str], int] = defaultdict(int)
        where: dict[tuple[str, str], set[int]] = defaultdict(set)
        for index, (word, count) in enumerate(zip(words, counts)):
            for pair in zip(word, word[1:]):
                pair_freq[pair] += count
                where[pair].add(index)
        heap = [(-freq, pair) for pair, freq in pair_freq.items()]
        heapq.heapify(heap)

        report_every = 500
        while self.vocab_size < vocab_size:
            selected: Optional[tuple[str, str]] = None
            while heap:
                negative, pair = heapq.heappop(heap)
                freq = pair_freq.get(pair, 0)
                if freq <= 0:
                    continue
                if -negative != freq:
                    heapq.heappush(heap, (-freq, pair))
                    continue
                if freq < min_pair_frequency:
                    heap.clear()
                    break
                if pair[0] + pair[1] in symbols:
                    continue  # colisão: já existe um símbolo com essa grafia
                selected = pair
                break
            if selected is None:
                break

            left, right = selected
            merged = left + right
            delta: dict[tuple[str, str], int] = defaultdict(int)
            for index in where.pop(selected, ()):
                word = words[index]
                if not any(word[j] == left and word[j + 1] == right for j in range(len(word) - 1)):
                    continue
                count = counts[index]
                for pair in zip(word, word[1:]):
                    delta[pair] -= count
                word = self._merge_word(word, left, right, merged)
                words[index] = word
                for pair in zip(word, word[1:]):
                    delta[pair] += count
                    where[pair].add(index)
            for pair, change in delta.items():
                if change == 0:
                    continue
                pair_freq[pair] += change
                if change > 0 and pair_freq[pair] > 0:
                    heapq.heappush(heap, (-pair_freq[pair], pair))

            symbols.add(merged)
            self.merge_rules.append((left, right, merged))
            self.id_to_token.append(merged)
            self.token_to_id[merged] = len(self.id_to_token) - 1
            if verbose and len(self.merge_rules) % report_every == 0:
                print(f"[Tokenizer] vocabulário {self.vocab_size}/{vocab_size}")

        if verbose:
            print(f"[Tokenizer] treino concluído: vocabulário {self.vocab_size}.")
        self._rebuild_indexes()
        return self

    def _encode_chunk(self, chunk: str) -> list[int]:
        symbols = [char if char in self.base_tokens else self.UNKNOWN_TOKEN for char in chunk]
        while len(symbols) > 1:
            selected_pair = None
            selected_rule = None
            for pair in zip(symbols, symbols[1:]):
                rule = self._merge_index.get(pair)
                if rule is not None and (selected_rule is None or rule[0] < selected_rule[0]):
                    selected_pair = pair
                    selected_rule = rule
            if selected_pair is None or selected_rule is None:
                break
            symbols = list(self._merge_pair(tuple(symbols), selected_pair, selected_rule[1]))
        return [self.token_to_id.get(symbol, 0) for symbol in symbols]

    def encode(self, text: str | Iterable[str], cache_limit: int = 100_000) -> list[int]:
        if not text:
            return []
        if len(self.id_to_token) <= 1:
            raise TokenizerError("Tokenizer ainda não foi treinado ou carregado.")
        encoded: list[int] = []

        if isinstance(text, str):
            chunks_iter = self._chunks(text)
        else:
            # text is an iterable of strings (e.g. lines)
            def combined_chunks():
                for line in text:
                    yield from self._chunks(line)
            chunks_iter = combined_chunks()

        for chunk in chunks_iter:
            cached = self._encode_cache.get(chunk)
            if cached is None:
                cached = tuple(self._encode_chunk(chunk))
                if len(self._encode_cache) < cache_limit:
                    self._encode_cache[chunk] = cached
            encoded.extend(cached)
        return encoded

    def decode(self, ids: Iterable[int]) -> str:
        pieces: list[str] = []
        for token_id in ids:
            try:
                index = int(token_id)
            except (TypeError, ValueError):
                pieces.append(self.UNKNOWN_TOKEN)
                continue
            if 0 <= index < self.vocab_size:
                pieces.append(self.id_to_token[index])
            else:
                pieces.append(self.UNKNOWN_TOKEN)
        return "".join(pieces)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "abrobabpe",
            "format_version": self.FORMAT_VERSION,
            "pretokenizer": self.pretokenizer,
            "unknown_token": self.UNKNOWN_TOKEN,
            "tokens": self.id_to_token,
            "merges": [list(rule) for rule in self.merge_rules],
            "base_tokens": sorted(self.base_tokens),
            "fingerprint": self.fingerprint,
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
            temporary.replace(destination)
        except OSError as exc:
            raise TokenizerError(f"Não foi possível salvar tokenizer em '{destination}': {exc}") from exc

    def _load_new_format(self, data: dict[str, Any]) -> None:
        if data.get("format") != "abrobabpe":
            raise TokenizerError("Formato de tokenizer desconhecido.")
        if int(data.get("format_version", -1)) != self.FORMAT_VERSION:
            raise TokenizerError("Versão de tokenizer não suportada.")
        tokens = data.get("tokens")
        merges = data.get("merges", [])
        base_tokens = data.get("base_tokens")
        if not isinstance(tokens, list) or len(tokens) < 2 or tokens[0] != self.UNKNOWN_TOKEN:
            raise TokenizerError("Arquivo tokenizer.json possui lista de tokens inválida.")
        if len(set(tokens)) != len(tokens) or not all(isinstance(token, str) for token in tokens):
            raise TokenizerError("Arquivo tokenizer.json contém tokens duplicados ou inválidos.")
        if not isinstance(base_tokens, list) or not all(isinstance(token, str) for token in base_tokens) :
            raise TokenizerError("Arquivo tokenizer.json possui base_tokens inválido.")
        parsed_merges: list[tuple[str, str, str]] = []
        for rule in merges:
            if not isinstance(rule, list) or len(rule) != 3 or not all(isinstance(item, str) for item in rule):
                raise TokenizerError("Arquivo tokenizer.json possui regra BPE inválida.")
            parsed_merges.append((rule[0], rule[1], rule[2]))
        pretokenizer = data.get("pretokenizer", "legacy")  # arquivos antigos não têm o campo
        if pretokenizer not in _PRETOKENIZERS:
            raise TokenizerError(f"Pré-tokenizador desconhecido: '{pretokenizer}'.")
        self.pretokenizer = pretokenizer
        self.id_to_token = tokens
        self.token_to_id = {token: index for index, token in enumerate(tokens)}
        self.base_tokens = set(base_tokens)
        self.merge_rules = parsed_merges
        self._rebuild_indexes()
        expected = data.get("fingerprint")
        if expected is not None and expected != self.fingerprint:
            raise TokenizerError("Fingerprint do tokenizer não confere; o arquivo pode estar corrompido.")

    def _load_legacy_vocab(self, data: dict[str, Any]) -> None:
        tokens = data.get("tokens", [])
        if not tokens:
            # Se não tiver tokens, tenta usar as chaves do dict legado
            tokens = sorted(data.keys())
        self.pretokenizer = "legacy"
        self.id_to_token = tokens
        self.token_to_id = {token: index for index, token in enumerate(tokens)}
        self.base_tokens = {token for token in self.id_to_token if len(token) == 1}
        self.merge_rules = []
        self._rebuild_indexes()

    def load(self, path: str | Path) -> "BPETokenizer":
        source = Path(path)
        try:
            with source.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError as exc:
            raise TokenizerError(f"Tokenizer não encontrado: '{source}'.") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise TokenizerError(f"Não foi possível ler tokenizer '{source}': {exc}") from exc
        if not isinstance(data, dict):
            raise TokenizerError("O tokenizer precisa ser um objeto JSON.")
        if data.get("format") == "abrobabpe":
            self._load_new_format(data)
        else:
            self._load_legacy_vocab(data)
        return self

class CharacterTokenizer(BPETokenizer):
    """Alias de compatibilidade; arquivos novos devem usar :class:`BPETokenizer`."""
