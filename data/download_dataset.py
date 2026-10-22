"""Baixa um limite de texto de datasets Hugging Face por streaming."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import (
    DATASET_TYPES,
    DataError,
    dataset_spec,
    prepare_corpus,
    raw_path_for,
    stream_dataset_to_file,
)
from src.utils import ConfigError, load_config, project_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Streaming de dataset Hugging Face para o AbrobaGPT")
    parser.add_argument("--dataset-type", choices=(*DATASET_TYPES, "canarim"), default="general")
    parser.add_argument("--dataset", type=str, default=None, help="Sobrescreve o repositório padrão")
    parser.add_argument("--size-mb", type=float, default=None, help="Limite aproximado em MB")
    parser.add_argument("--output", type=str, default=None, help="Arquivo de texto de destino")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--config", type=str, default="config/base_config.json")
    parser.add_argument("--no-prepare", action="store_true", help="Não executar a preparação após baixar")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        spec = dataset_spec(args.dataset_type)
        dataset_id = args.dataset or spec["dataset"]
        size_mb = args.size_mb if args.size_mb is not None else float(config["hf_default_size_mb"])
        output = project_path(args.output) if args.output else raw_path_for(args.dataset_type)
        result = stream_dataset_to_file(
            dataset_id,
            output,
            dataset_type=args.dataset_type,
            size_mb=size_mb,
            split=args.split,
            dataset_config=args.dataset_config,
        )
        print(
            f"Download concluído: {result['records']} exemplos, "
            f"{result['size_bytes'] / (1024 * 1024):.2f} MB em {result['output']}"
        )
        if not args.no_prepare:
            metadata = prepare_corpus(output, config, dataset_type=args.dataset_type)
            print(
                f"Preparação concluída: vocabulário {metadata['vocab_size']}, "
                f"treino {metadata['train_tokens']} tokens, validação {metadata['val_tokens']} tokens."
            )
    except (ConfigError, DataError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
