import numpy as np
import json
from src.tokenizer import BPETokenizer

# Carrega o tokenizer
tokenizer = BPETokenizer().load("data/processed/tokenizer.json")

# Carrega os dados de treino
dtype = np.uint16 # Baseado no metadata.json geralmente
try:
    data = np.fromfile("data/processed/train.bin", dtype=dtype)
except:
    # Tenta uint32 se falhar
    data = np.fromfile("data/processed/train.bin", dtype=np.uint32)

print("=== INSPEÇÃO DE DADOS PROCESSADOS ===")
print(f"Total de tokens no treino: {len(data):,}")

# Vamos pegar 5 amostras de diferentes partes do arquivo
indices = [0, len(data)//4, len(data)//2, 3*len(data)//4, len(data)-512]
block_size = 256

for i, start in enumerate(indices):
    end = min(start + block_size, len(data))
    chunk = data[start:end]
    text = tokenizer.decode(chunk)
    print(f"\n--- Amostra {i+1} (Posição {start}) ---")
    print(text)
    print("-" * 40)

# Verificar se há sequências repetitivas óbvias (ex: o mesmo token 10x seguidas)
def find_repetitions(data, threshold=5):
    for i in range(len(data)-threshold):
        if len(set(data[i:i+threshold])) == 1:
            return data[i]
    return None

rep = find_repetitions(data)
if rep is not None:
    print(f"\n⚠️ ALERTA: Encontrada sequência de tokens repetidos: Token ID {rep}")
else:
    print("\n✅ Nenhuma repetição sequencial simples encontrada.")
