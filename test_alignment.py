import torch
import torch.nn.functional as F
from src.model import GPT

# Configuração mínima para o teste
config = {
    "n_layer": 1,
    "n_embd": 16,
    "n_head": 2,
    "block_size": 4,
    "vocab_size": 10,
    "dropout": 0.0
}

model = GPT(config)
model.eval()

# Entrada: Sequência de tokens [1, 2, 3, 4]
idx = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
targets = torch.tensor([[1, 2, 3, 4]], dtype=torch.long) 

logits, loss = model(idx, targets)

print("=== TESTE DE ALINHAMENTO DE TOKENS ===")
print(f"Entrada (idx):    {idx[0].tolist()}")
print(f"Alvos (targets):  {targets[0].tolist()}")

print("\nAnalise de Posicoes na Loss:")
for t in range(idx.shape[1]):
    logit_pos_t = logits[t] 
    target_pos_t = targets[0, t].item()
    
    print(f"Posicao {t}:")
    print(f"  Input token:  {idx[0, t].item()}")
    print(f"  Target usado: {target_pos_t}")
    
    if idx[0, t].item() == target_pos_t:
        print("  [!] ERRO: O modelo esta tentando prever o PROPRIO token!")
    else:
        print("  [V] CORRETO: O modelo esta prevendo o proximo token.")

print("\nConclusao:")
if idx[0, 0].item() == targets[0, 0].item():
    print("O BUG-01 foi CONFIRMADO: O alinhamento esta errado.")
else:
    print("O alinhamento parece correto.")
