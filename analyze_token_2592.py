import numpy as np
import json
from src.tokenizer import BPETokenizer

tokenizer = BPETokenizer().load("data/processed/tokenizer.json")
data = np.fromfile("data/processed/train.bin", dtype=np.uint16)

token_id = 2592
try:
    token_text = tokenizer.id_to_token[token_id]
except IndexError:
    token_text = "DESCONHECIDO"

print(f"=== ANÁLISE DO TOKEN {token_id} ===")
print(f"Representação textual: '{token_text}'")

# Contagem total
total_occurrences = np.sum(data == token_id)
print(f"Ocorrências totais: {total_occurrences:,}")

# Busca por blocos de repetição (ex: 10 ou mais vezes seguidas)
threshold = 10
blocks = 0
max_streak = 0
current_streak = 0

for t in data:
    if t == token_id:
        current_streak += 1
    else:
        if current_streak >= threshold:
            blocks += 1
            max_streak = max(max_streak, current_streak)
        current_streak = 0

if current_streak >= threshold:
    blocks += 1
    max_streak = max(max_streak, current_streak)

print(f"Blocos de repetição (>= {threshold} vezes): {blocks}")
print(f"Maior sequência contínua encontrada: {max_streak}")

# Vamos ver o contexto de um desses blocos
if blocks > 0:
    print("\nExemplo de contexto de repetição:")
    for i in range(len(data)-100):
        if data[i] == token_id and data[i+1] == token_id:
            # Pegamos um pedaço ao redor
            start = max(0, i-50)
            end = min(len(data), i+50)
            print(tokenizer.decode(data[start:end]))
            break

