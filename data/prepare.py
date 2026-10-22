"""Prepara um corpus local em tokenizer.json, train.bin e val.bin."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import DATASET_TYPES, DataError, prepare_corpus, raw_path_for
from src.utils import ConfigError, load_config, project_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Preparar corpus para o AbrobaGPT")
    parser.add_argument("--input", type=str, default=None, help="Arquivo UTF-8 de entrada")
    parser.add_argument(
        "--dataset-type", choices=(*DATASET_TYPES, "canarim"), default="general", help="Tipo do corpus"
    )
    parser.add_argument("--config", type=str, default="config/base_config.json")
    parser.add_argument("--vocab-size", type=int, default=None, help="Limite de vocabulário BPE")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        if args.vocab_size is not None:
            if args.vocab_size < 2:
                raise DataError("--vocab-size deve ser pelo menos 2.")
            config["tokenizer_vocab_size"] = args.vocab_size
        raw_path = project_path(args.input) if args.input else raw_path_for(args.dataset_type)
        metadata = prepare_corpus(raw_path, config, dataset_type=args.dataset_type)
    except (ConfigError, DataError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1

    print("Preparação concluída.")
    print(f"Tokenizer: data/processed/tokenizer.json ({metadata['vocab_size']} tokens)")
    print(f"Treino: {metadata['train_tokens']} tokens | validação: {metadata['val_tokens']} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
