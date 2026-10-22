"""Aquisição e preparação de dados para o AbrobaGPT."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np
from colorama import Fore, Style, init

from src.tokenizer import BPETokenizer, TokenizerError
from src.utils import project_path

init(autoreset=True)


DATASET_SPECS: dict[str, dict[str, str]] = {
    "general": {
        "label": "Corpus PT-BR geral",
        "dataset": "Madras1/corpus-ptbr-v1",
        "raw_filename": "input.txt",
    },
    "instruct": {
        "label": "Dataset Q&A / Instruct",
        "dataset": "emdemor/ptbr-question-and-answer",
        "raw_filename": "input_instruct.txt",
    },
    "alpaca": {
        "label": "Alpaca PT-BR (instruções)",
        "dataset": "dominguesm/alpaca-data-pt-br",
        "raw_filename": "input_alpaca.txt",
    },
    "ragqa": {
        "label": "Madras1 RAG QA Fulltext PT-BR",
        "dataset": "Madras1/rag-qa-fulltext-ptbr",
        # O repositório possui também ``documents``; o pipeline de treino espera
        # pares pergunta/resposta, portanto este é o subconjunto compatível.
        "config": "qa_pairs",
        "raw_filename": "raw/rag-qa-fulltext-ptbr/train.jsonl",
    },
}

DATASET_TYPES: tuple[str, ...] = tuple(DATASET_SPECS)

# Valores que o Alpaca usa como "campo vazio" no campo `input`.
_ALPACA_EMPTY_VALUES = {"", "<noinput>", "<no input>", "no input", "noinput", "n/a", "<no output>"}
# Incrementar quando a serialização do corpus mudar invalida automaticamente os
# arquivos processados antigos, evitando treinar com um formato diferente do chat.
_CORPUS_FORMAT_VERSION = 4


class DataError(RuntimeError):
    """Falha de dataset ou preparação de corpus."""


def normalize_dataset_type(dataset_type: str) -> str:
    normalized = str(dataset_type).lower().strip()
    # Alias legado: nunca é emitido pela TUI; mantém compatibilidade com comandos antigos.
    if normalized == "canarim":
        normalized = "ragqa"
    if normalized not in DATASET_SPECS:
        options = ", ".join(f"'{name}'" for name in DATASET_TYPES)
        raise DataError(f"Tipo de dataset deve ser um de: {options}.")
    return normalized


def dataset_spec(dataset_type: str) -> dict[str, str]:
    return DATASET_SPECS[normalize_dataset_type(dataset_type)]


def raw_path_for(dataset_type: str) -> Path:
    return project_path(Path("data") / dataset_spec(dataset_type)["raw_filename"])


def processed_dir() -> Path:
    return project_path("data/processed")


def processed_paths() -> dict[str, Path]:
    directory = processed_dir()
    return {
        "tokenizer": directory / "tokenizer.json",
        "train": directory / "train.bin",
        "val": directory / "val.bin",
        "metadata": directory / "metadata.json",
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(data: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, destination)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise DataError(f"Não foi possível escrever '{destination}': {exc}") from exc


def _atomic_array(array: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        array.tofile(temporary)
        os.replace(temporary, destination)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise DataError(f"Não foi possível escrever '{destination}': {exc}") from exc


def _read_metadata() -> Optional[dict[str, Any]]:
    path = processed_paths()["metadata"]
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        raise DataError(f"Metadados processados inválidos em '{path}': {exc}") from exc
    return value if isinstance(value, dict) else None


def _preparation_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """Opções do config que alteram tokenizer/split e, portanto, os arquivos processados."""
    return {
        "tokenizer_vocab_size": int(config["tokenizer_vocab_size"]),
        "tokenizer_training_chars": int(config["tokenizer_training_chars"]),
        "tokenizer_min_pair_frequency": int(config["tokenizer_min_pair_frequency"]),
        "validation_fraction": float(config["validation_fraction"]),
        "pretokenizer": BPETokenizer.DEFAULT_PRETOKENIZER,
        "corpus_format_version": _CORPUS_FORMAT_VERSION,
    }


def processed_is_current(
    raw_path: str | Path, dataset_type: str, config: Optional[Mapping[str, Any]] = None
) -> bool:
    """True se os arquivos processados ainda correspondem ao corpus e às opções do config.

    Metadados antigos (sem as opções de preparação) contam como desatualizados quando
    um ``config`` é informado, para que a preparação seja refeita uma vez.
    """
    raw = project_path(raw_path)
    paths = processed_paths()
    if not raw.exists() or not all(path.exists() for path in paths.values()):
        return False
    metadata = _read_metadata()
    if metadata is None:
        return False
    try:
        if config is not None:
            wanted = _preparation_settings(config)
            if any(metadata.get(key) != value for key, value in wanted.items()):
                return False
        return (
            metadata.get("raw_sha256") == sha256_file(raw)
            and metadata.get("dataset_type") == normalize_dataset_type(dataset_type)
        )
    except OSError:
        return False


def prepare_corpus(
    raw_path: str | Path,
    config: Mapping[str, Any],
    *,
    dataset_type: str = "general",
) -> dict[str, Any]:
    """Cria tokenizer.json, train.bin, val.bin e metadados consistentes."""
    raw = project_path(raw_path)
    kind = normalize_dataset_type(dataset_type)

    if not raw.exists():
        raise DataError(f"Arquivo local não encontrado: '{raw}'.")

    validation_fraction = float(config["validation_fraction"])
    if not 0.0 < validation_fraction < 1.0:
        raise DataError("validation_fraction deve estar entre 0 e 1.")

    # O perfil RAG QA é JSONL: cada exemplo (instrução + entrada + resposta) é mantido
    # inteiro no mesmo split, e o tokenizer é ajustado somente com exemplos de treino.
    example_stats: dict[str, int] = {}
    if kind == "ragqa":
        import json
        import hashlib

        def rows():
            with raw.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(item, dict):
                        continue
                    instruction = _clean_ragqa_field(_to_text(item.get("instruction")))
                    context = _clean_ragqa_field(_to_text(item.get("input")))
                    answer = _clean_ragqa_field(_to_text(item.get("output")), answer=True)
                    if context.lower() in _ALPACA_EMPTY_VALUES:
                        context = ""
                    if answer == "__FULLTEXT__":
                        if instruction.strip():
                            yield instruction.strip() + "\n\n"
                        continue
                    if not instruction or not answer:
                        continue
                    question = f"{instruction}\n{context}" if context else instruction
                    yield f"Pergunta: {question}\nResposta: {answer}\n\n"

        def split_for(text: str) -> str:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            return "val" if int.from_bytes(digest[:8], "big") / (2**64) < validation_fraction else "train"

        # Estatísticas por exemplo completo, sem manter o corpus em memória.
        total_examples = train_examples = val_examples = 0
        for text in rows():
            total_examples += 1
            if split_for(text) == "val":
                val_examples += 1
            else:
                train_examples += 1
        if total_examples < 2 or not train_examples or not val_examples:
            raise DataError("RAG QA precisa de pelo menos dois exemplos válidos para treino/validação.")
        with raw.open("r", encoding="utf-8") as source_handle:
            raw_examples = sum(1 for _ in source_handle)
        example_stats = {"valid_examples": total_examples, "train_examples": train_examples,
                         "val_examples": val_examples, "discarded_examples": max(0, raw_examples - total_examples)}

        def train_text():
            yield from (text for text in rows() if split_for(text) == "train")
        def val_text():
            yield from (text for text in rows() if split_for(text) == "val")

        tokenizer = BPETokenizer()
        try:
            tokenizer.fit(train_text(), vocab_size=int(config["tokenizer_vocab_size"]),
                          min_pair_frequency=int(config["tokenizer_min_pair_frequency"]),
                          training_chars=int(config["tokenizer_training_chars"]), verbose=False)
            train_encoded = tokenizer.encode(train_text())
            val_encoded = tokenizer.encode(val_text())
        except (TokenizerError, KeyError, ValueError) as exc:
            raise DataError(f"Falha ao preparar tokenizer: {exc}") from exc
        if len(train_encoded) < 2 or len(val_encoded) < 2:
            raise DataError("RAG QA precisa de exemplos suficientes nos splits de treino e validação.")
        train_ids = np.asarray(train_encoded, dtype=np.uint16 if tokenizer.vocab_size <= np.iinfo(np.uint16).max else np.uint32)
        val_ids = np.asarray(val_encoded, dtype=train_ids.dtype)
        train_tokens, val_tokens = len(train_ids), len(val_ids)
        dtype = train_ids.dtype
    else:
        # Usamos um gerador para evitar carregar todo o arquivo na RAM
        def stream_text():
            with raw.open("r", encoding="utf-8") as f:
                for line in f:
                    yield line

        tokenizer = BPETokenizer()
        try:
            tokenizer.fit(stream_text(), vocab_size=int(config["tokenizer_vocab_size"]),
                          min_pair_frequency=int(config["tokenizer_min_pair_frequency"]),
                          training_chars=int(config["tokenizer_training_chars"]))
            encoded = tokenizer.encode(stream_text())
        except (TokenizerError, KeyError, ValueError) as exc:
            raise DataError(f"Falha ao preparar tokenizer: {exc}") from exc
        if len(encoded) < 4:
            raise DataError("O corpus é curto demais; são necessários ao menos quatro tokens.")
        split = int(len(encoded) * (1.0 - validation_fraction))
        split = max(2, min(len(encoded) - 2, split))
        dtype = np.uint16 if tokenizer.vocab_size <= np.iinfo(np.uint16).max else np.uint32
        ids = np.asarray(encoded, dtype=dtype)
        train_ids, val_ids = ids[:split], ids[split:]
        train_tokens, val_tokens = len(train_ids), len(val_ids)

    paths = processed_paths()
    tokenizer.save(paths["tokenizer"])
    _atomic_array(train_ids, paths["train"])
    _atomic_array(val_ids, paths["val"])
    metadata: dict[str, Any] = {
        "format_version": 2,
        "dataset_type": kind,
        "raw_path": str(raw),
        "raw_sha256": sha256_file(raw),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dtype": np.dtype(dtype).name,
        "train_tokens": int(train_tokens),
        "val_tokens": int(val_tokens),
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "train_bytes": int(paths["train"].stat().st_size),
        "val_bytes": int(paths["val"].stat().st_size),
        "raw_bytes": int(raw.stat().st_size),
        **example_stats,
        **_preparation_settings(config),
    }
    _atomic_json(metadata, paths["metadata"])
    return metadata


def prepare_if_needed(
    raw_path: str | Path,
    config: Mapping[str, Any],
    *,
    dataset_type: str = "general",
    force: bool = False,
) -> tuple[bool, dict[str, Any]]:
    if not force and processed_is_current(raw_path, dataset_type, config):
        metadata = _read_metadata()
        assert metadata is not None
        return False, metadata
    return True, prepare_corpus(raw_path, config, dataset_type=dataset_type)


def load_processed_data() -> tuple[BPETokenizer, np.ndarray, np.ndarray, dict[str, Any]]:
    paths = processed_paths()
    if not all(path.exists() for path in paths.values()):
        missing = [name for name, path in paths.items() if not path.exists()]
        raise DataError(
            "Dados processados ausentes (" + ", ".join(missing) + "). Execute a preparação do corpus."
        )
    metadata = _read_metadata()
    if metadata is None:
        raise DataError("metadata.json está ausente ou inválido; reprepare o corpus.")
    dtype_name = metadata.get("dtype")
    if dtype_name not in {"uint16", "uint32"}:
        raise DataError("dtype em metadata.json é inválido; reprepare o corpus.")
    dtype = np.dtype(dtype_name)
    try:
        train = np.fromfile(paths["train"], dtype=dtype)
        val = np.fromfile(paths["val"], dtype=dtype)
        tokenizer = BPETokenizer().load(paths["tokenizer"])
    except (OSError, TokenizerError) as exc:
        raise DataError(f"Falha ao carregar dados processados: {exc}") from exc
    if len(train) < 2 or len(val) < 2:
        raise DataError("train.bin ou val.bin não possui tokens suficientes; reprepare com mais dados.")
    if tokenizer.vocab_size != metadata.get("vocab_size"):
        raise DataError("vocab_size do tokenizer não confere com metadata.json.")
    if tokenizer.fingerprint != metadata.get("tokenizer_fingerprint"):
        raise DataError("Fingerprint do tokenizer não confere com metadata.json.")
    if int(train.max(initial=0)) >= tokenizer.vocab_size or int(val.max(initial=0)) >= tokenizer.vocab_size:
        raise DataError("Dados tokenizados têm IDs fora do vocabulário; reprepare o corpus.")
    return tokenizer, train, val, metadata


def _to_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return " ".join(piece for item in value if (piece := _to_text(item)))
    if isinstance(value, Mapping):
        return " ".join(piece for item in value.values() if (piece := _to_text(item)))
    return ""


def _clean_ragqa_field(value: str, *, answer: bool = False) -> str:
    """Remove marcadores de papel sem apagar o conteúdo da amostra RAG QA."""
    import re
    text = value.strip().replace("\r\n", "\n").replace("\r", "\n")
    # Alguns espelhos/exportações trazem os delimitadores dentro dos próprios campos.
    if answer:
        text = re.sub(r"(?im)^\s*#{2,}\s*(?:RESPOSTA|RESPOSTAS|ANSWER)\s*:?\s*", "", text)
    else:
        text = re.sub(r"(?im)^\s*#{2,}\s*(?:PERGUNTA|QUESTÃO|QUESTAO|INSTRUÇÕES?|ENTRADA|QUESTION|INPUT)\s*:?\s*", "", text)
    return text.strip()


def _first_field(example: Mapping[str, Any], names: Iterable[str]) -> str:
    normalized = {str(key).lower(): value for key, value in example.items()}
    for name in names:
        value = _to_text(normalized.get(name))
        if value:
            return value
    return ""


def example_to_text(example: Mapping[str, Any], dataset_type: str) -> str:
    kind = normalize_dataset_type(dataset_type)
    if kind in {"alpaca", "ragqa"}:
        # Esquema dos datasets de instruções: instruction / input / output.
        instruction = _first_field(example, ("instruction", "instrucao", "instrução", "question", "pergunta", "query", "prompt", "user", "question_text"))
        context = _first_field(example, ("input", "entrada", "context", "contexto", "retrieved_context", "passage"))
        answer = _first_field(example, ("output", "resposta", "response", "answer", "completion", "assistant", "answer_text", "ground_truth"))
        if kind == "ragqa":
            instruction = _clean_ragqa_field(instruction)
            context = _clean_ragqa_field(context)
            answer = _clean_ragqa_field(answer, answer=True)
            # Alguns dumps RAG fornecem somente um campo de texto integral.
            if not instruction or not answer:
                fulltext = _first_field(example, ("text", "full_text", "fulltext", "content", "document"))
                if fulltext:
                    return fulltext.strip() + "\n\n"
        if context.lower() in _ALPACA_EMPTY_VALUES:
            context = ""
        if answer.lower() in _ALPACA_EMPTY_VALUES:
            answer = ""
        if not instruction or not answer:
            return ""
        question = f"{instruction}\n{context}" if context else instruction
        # Mesmo formato do tipo 'instruct', para o chat/geração continuarem compatíveis.
        return f"Pergunta: {question}\nResposta: {answer}\n\n"
    if kind == "instruct":
        question = _first_field(example, ("question", "pergunta", "instruction", "prompt", "input"))
        answer = _first_field(example, ("answer", "resposta", "response", "output", "completion"))
        if question and answer:
            return f"Pergunta: {question}\nResposta: {answer}\n\n"
        if answer:
            return f"Resposta: {answer}\n\n"
        return ""

    text = _first_field(example, ("text", "content", "document", "body", "article", "sentence"))
    if text:
        return text + "\n"
    for value in example.values():
        text = _to_text(value)
        if text:
            return text + "\n"
    return ""


def _utf8_prefix(data: bytes, maximum: int) -> bytes:
    prefix = data[:maximum]
    while prefix:
        try:
            prefix.decode("utf-8")
            return prefix
        except UnicodeDecodeError as exc:
            prefix = prefix[: exc.start]
    return b""


def _stream_ragqa(dataset_id: str, destination: Path, temporary: Path, byte_limit: int,
                    split: str, dataset_config: Optional[str]) -> dict[str, Any]:
    """Baixa exemplos do perfil RAG QA para JSONL, com deduplicação SQLite e troca atômica."""
    import json
    import sqlite3
    try:
        from datasets import load_dataset
        from tqdm import tqdm
    except ImportError as exc:
        raise DataError("A dependência 'datasets' ou 'tqdm' não está instalada.") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Arquivos temporários exclusivos por execução evitam colisões/WinError 32
    # quando duas instâncias tentam preparar o mesmo dataset simultaneamente.
    db_path = temporary.with_name(temporary.name + ".dedupe.sqlite3")
    records = skipped = written = duplicates = 0
    db: Optional[sqlite3.Connection] = None
    try:
        # O context manager da conexão SQLite confirma/reverte, mas não garante
        # close(). No Windows o handle precisa ser fechado antes do unlink abaixo.
        db = sqlite3.connect(db_path)
        with db, temporary.open("w", encoding="utf-8", newline="\n") as out:
            db.execute("CREATE TABLE IF NOT EXISTS seen (digest TEXT PRIMARY KEY)")
            db.execute("DELETE FROM seen")
            stream = load_dataset(dataset_id, name=dataset_config, split=split, streaming=True)
            bar = tqdm(desc=f"Baixando {dataset_id}", unit="ex", unit_scale=False)
            try:
                for example in stream:
                    if not isinstance(example, Mapping):
                        skipped += 1; continue
                    instruction = _first_field(example, ("instruction", "instrucao", "instrução", "question", "pergunta", "query", "prompt", "user", "question_text"))
                    context = _first_field(example, ("input", "entrada", "context", "contexto", "retrieved_context", "passage"))
                    answer = _first_field(example, ("output", "resposta", "response", "answer", "completion", "assistant", "answer_text", "ground_truth"))
                    if not instruction or not answer:
                        fulltext = _first_field(example, ("text", "full_text", "fulltext", "content", "document"))
                        if not fulltext:
                            skipped += 1; continue
                        row = {"instruction": fulltext, "input": "", "output": "__FULLTEXT__"}
                    else:
                        row = {"instruction": instruction, "input": context, "output": answer}
                    canonical = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                    try:
                        db.execute("INSERT INTO seen(digest) VALUES (?)", (digest,))
                    except sqlite3.IntegrityError:
                        duplicates += 1; continue
                    encoded = (canonical + "\n").encode("utf-8")
                    # Nunca trunca um registro: encerra antes de exceder o limite.
                    if written + len(encoded) > byte_limit:
                        db.rollback()
                        break
                    out.write(canonical + "\n")
                    written += len(encoded); records += 1; bar.update(1)
                out.flush(); os.fsync(out.fileno())
            finally:
                bar.close()
            db.commit()
        if records == 0:
            raise DataError("Nenhum exemplo válido do Madras1 RAG QA coube no limite; aumente Limite MB ou verifique a conexão.")
        try:
            os.replace(temporary, destination)
        except PermissionError as exc:
            raise DataError(
                f"O Windows bloqueou a substituição de '{destination}'. Feche outras instâncias do AbrobaGPT, "
                "terminais/processos Python que estejam usando esse arquivo e tente novamente. "
                f"O download temporário foi preservado apenas durante a operação. Detalhe: {exc}"
            ) from exc
    except DataError:
        temporary.unlink(missing_ok=True); raise
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise DataError(f"Falha ao baixar dataset RAG QA '{dataset_id}': {exc}") from exc
    finally:
        # O índice de duplicatas é temporário; não deixa lixo se a operação terminou.
        if db is not None:
            db.close()
        db_path.unlink(missing_ok=True)
    source_marker = destination.with_suffix(destination.suffix + ".source.json")
    try:
        source_marker.write_text(json.dumps({"dataset": dataset_id, "split": split, "dataset_config": dataset_config}, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        raise DataError(f"Download terminou, mas não foi possível gravar o marcador de origem: {exc}") from exc
    return {"dataset": dataset_id, "output": str(destination), "size_bytes": written,
            "records": records, "skipped": skipped, "duplicates": duplicates,
            "dataset_type": "ragqa"}


def stream_dataset_to_file(
    dataset_id: str,
    output_path: str | Path,
    *,
    dataset_type: str,
    size_mb: float,
    split: str = "train",
    dataset_config: Optional[str] = None,
) -> dict[str, Any]:
    """Baixa em streaming com barra de progresso TUI."""
    if size_mb <= 0:
        raise DataError("O limite de tamanho deve ser maior que zero MB.")
    destination = project_path(output_path)
    temporary = destination.with_name(destination.name + f".{os.getpid()}.{uuid.uuid4().hex}.download")
    byte_limit = max(1, int(float(size_mb) * 1024 * 1024))
    if normalize_dataset_type(dataset_type) == "ragqa":
        # Reutiliza um JSONL local já íntegro; a substituição de downloads é atômica.
        source_marker = destination.with_suffix(destination.suffix + ".source.json")
        if destination.is_file() and destination.stat().st_size > 0 and source_marker.is_file():
            try:
                import json
                if source_marker.is_file():
                    marker = json.loads(source_marker.read_text(encoding="utf-8"))
                    if marker.get("dataset") != dataset_id or marker.get("split") != split or marker.get("dataset_config") != dataset_config:
                        raise ValueError("cache pertence a outro dataset/split")
                count = 0
                with destination.open("r", encoding="utf-8") as cached:
                    for line in cached:
                        row = json.loads(line)
                        if not isinstance(row, dict) or not row.get("instruction") or not row.get("output"):
                            raise ValueError("registro inválido")
                        count += 1
                if count:
                    return {"dataset": dataset_id, "output": str(destination),
                            "size_bytes": destination.stat().st_size, "records": count,
                            "skipped": 0, "duplicates": 0, "cached": True,
                            "dataset_type": "ragqa"}
            except (OSError, ValueError, json.JSONDecodeError):
                pass  # O arquivo existente fica intacto até o novo download terminar.
        return _stream_ragqa(dataset_id, destination, temporary, byte_limit, split, dataset_config)
    try:
        from datasets import load_dataset
        from tqdm import tqdm
    except ImportError as exc:
        raise DataError("A dependência 'datasets' ou 'tqdm' não está instalada.") from exc
    records = 0
    skipped = 0
    written = 0
    try:
        stream = load_dataset(dataset_id, name=dataset_config, split=split, streaming=True)
        destination.parent.mkdir(parents=True, exist_ok=True)

        with temporary.open("wb") as handle:
            with tqdm(total=byte_limit, unit="B", unit_scale=True, desc=f"Baixando {dataset_id}", colour="cyan") as pbar:
                for example in stream:
                    if not isinstance(example, Mapping):
                        skipped += 1
                        continue
                    text = example_to_text(example, dataset_type)
                    if not text:
                        skipped += 1
                        continue
                    encoded = text.encode("utf-8")
                    remaining = byte_limit - written
                    if remaining <= 0:
                        break
                    if len(encoded) > remaining:
                        encoded = _utf8_prefix(encoded, remaining)
                    if not encoded:
                        break
                    handle.write(encoded)
                    written += len(encoded)
                    records += 1
                    pbar.update(len(encoded))
                    if written >= byte_limit:
                        break
        if records == 0:
            raise DataError(
                "O streaming não forneceu texto utilizável; verifique o dataset, split e esquema."
            )
        os.replace(temporary, destination)
    except DataError:
        temporary.unlink(missing_ok=True)
        raise
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise DataError(f"Falha no streaming de '{dataset_id}': {exc}") from exc

    return {
        "dataset": dataset_id,
        "output": str(destination),
        "size_bytes": written,
        "records": records,
        "skipped": skipped,
        "dataset_type": normalize_dataset_type(dataset_type),
    }
