"""Inspeciona se o corpus preparado contém várias perguntas/respostas concatenadas."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from src.data import processed_paths
from src.tokenizer import BPETokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens', type=int, default=250000, help='Máximo de tokens a inspecionar')
    parser.add_argument('--split', choices=('train', 'val'), default='train')
    args = parser.parse_args()
    paths = processed_paths()
    if not paths['tokenizer'].exists() or not paths[args.split].exists():
        parser.error('Corpus processado não encontrado; baixe/prepara o dataset primeiro.')
    tokenizer = BPETokenizer().load(paths['tokenizer'])
    ids = np.fromfile(paths[args.split], dtype=np.uint16)[:max(0, args.tokens)].astype(np.int64).tolist()
    text = tokenizer.decode(ids)
    q = text.count('Pergunta:')
    a = text.count('Resposta:')
    pairs = min(q, a)
    repeated_qas = max(0, pairs - 1)
    print(json.dumps({
        'split': args.split, 'tokens_inspected': len(ids), 'characters_decoded': len(text),
        'question_markers': q, 'answer_markers': a,
        'multiple_qa_examples_in_window': repeated_qas > 0,
        'estimated_qa_pairs_in_window': pairs,
        'warning': ('O modelo é treinado em janelas contínuas do corpus; se há vários marcadores, pode aprender a continuar com outra pergunta após responder.' if repeated_qas else 'Nenhum par Pergunta/Resposta completo detectado nesta janela; isso não prova ausência no corpus inteiro.')
    }, ensure_ascii=False, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
