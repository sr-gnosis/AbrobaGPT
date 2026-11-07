import numpy as np
import json
from src.tokenizer import BPETokenizer

# Carrega o tokenizer
tokenizer = BPETokenizer().load("data/processed/tokenizer.json")

# Carrega os dados de treino
try:
    # Tentamos ler como uint16 primeiro (comum para vocabulários < 65k)
    data = np.fromfile("data/processed/train.bin", dtype=np.uint16)
except Exception:
    data = np.fromfile("data/processed/train.bin", dtype=np.uint32)

print("=== INSPECAO DE DADOS PROCESSADOS ===")
print(f"Total de tokens no treino: {len(data):,}")

# Amostras de diferentes partes do arquivo
indices = [0, len(data)//4, len(data)//2, 3*len(data)//4, len(data)-512]
block_size = 300

for i, start in enumerate(indices):
    end = min(start + block_size, len(data))
    chunk = data[start:end]
    text = tokenizer.decode(chunk)
    print(f"\n--- Amostra {i+1} (Posicao {start}) ---")
    print(text)
    print("-" * 40)

# Verificacao de repeticoes sequenciais
def find_repetitions(data, threshold=10):
    for i in range(len(data)-threshold):
        if len(set(data[i:i+threshold])) == 1:
            return data[i]
    return None

rep = find_repetitions(data)
if rep is not None:
    print(f"\nALERTA: Encontrada sequencia de tokens repetidos: Token ID {rep}")
else:
    print("\nNenhuma repeticao sequencial simples encontrada.")
