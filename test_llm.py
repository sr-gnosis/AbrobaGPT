import torch
import numpy as np
import os
from src.tokenizer import CharacterTokenizer
from src.model import GPT
from src.utils import save_checkpoint, load_checkpoint
from colorama import init, Fore, Style

init(autoreset=True)

def test_tokenizer():
    print(f"{Fore.YELLOW}--- [1/5] Testing Tokenizer ---", flush=True)
    tokenizer = CharacterTokenizer()
    text = "Olá Mundo! 123"
    tokenizer.fit(text)
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text, f"Tokenizer failed: {decoded} != {text}"
    print(f"{Fore.GREEN}[OK] Tokenizer PASSED\n", flush=True)

def test_model_shape():
    print(f"{Fore.YELLOW}--- [2/5] Testing Model Shape ---", flush=True)
    config = {
        "n_layer": 2,
        "n_embd": 32,
        "n_head": 2,
        "block_size": 16,
        "vocab_size": 50
    }
    model = GPT(config)
    idx = torch.randint(0, 50, (1, 16))
    logits, _ = model(idx)
    assert logits.shape == (1 * 16, 50), f"Wrong shape: {logits.shape}"
    print(f"{Fore.GREEN}[OK] Model Shape PASSED\n", flush=True)

def test_causal_mask():
    print(f"{Fore.YELLOW}--- [3/5] Testing Causal Mask ---", flush=True)
    config = {
        "n_layer": 2,
        "n_embd": 32,
        "n_head": 2,
        "block_size": 16,
        "vocab_size": 50
    }
    model = GPT(config)
    idx1 = torch.randint(0, 50, (1, 16))
    idx2 = idx1.clone()
    idx2[0, -1] = (idx1[0, -1] + 1) % 50

    with torch.no_grad():
        logits1, _ = model(idx1)
        logits2, _ = model(idx2)

    diff = torch.abs(logits1[:-1] - logits2[:-1]).max()
    assert diff < 1e-5, f"Causal mask failed: max diff {diff}"
    print(f"{Fore.GREEN}[OK] Causal Mask PASSED\n", flush=True)

def test_checkpointing():
    print(f"{Fore.YELLOW}--- [4/5] Testing Checkpointing ---", flush=True)
    config = {
        "n_layer": 1,
        "n_embd": 16,
        "n_head": 1,
        "block_size": 8,
        "vocab_size": 10
    }
    model = GPT(config)
    optimizer = torch.optim.AdamW(model.parameters())

    path = "checkpoints/test_ckpt.pt"
    os.makedirs("checkpoints", exist_ok=True)

    save_checkpoint(model, optimizer, 100, path)

    with torch.no_grad():
        model.lm_head.weight.fill_(0)

    load_checkpoint(model, optimizer, path)

    assert torch.any(model.lm_head.weight != 0), "Checkpoint failed to restore weights"
    print(f"{Fore.GREEN}[OK] Checkpointing PASSED\n", flush=True)

def test_overfitting():
    print(f"{Fore.YELLOW}--- [5/5] Testing Overfitting (This may take a minute) ---", flush=True)
    config = {
        "n_layer": 2,
        "n_embd": 32,
        "n_head": 2,
        "block_size": 16,
        "vocab_size": 0,
        "learning_rate": 0.01,
        "device": "cpu"
    }
    text = "O gato subiu no telhado."
    tokenizer = CharacterTokenizer()
    tokenizer.fit(text)
    config['vocab_size'] = tokenizer.vocab_size

    model = GPT(config).to("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

    tokens = torch.tensor([tokenizer.encode(text)], dtype=torch.long)

    model.train()
    for i in range(200):
        x = tokens[:, :-1]
        y = tokens[:, 1:]
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            print(f"   Step {i}/200 - Loss: {loss.item():.4f}", flush=True)

    logits, loss = model(tokens[:, :-1], tokens[:, 1:])
    assert loss.item() < 0.5, f"Loss too high: {loss.item()}"
    print(f"{Fore.GREEN}[OK] Overfitting PASSED\n", flush=True)

if __name__ == "__main__":
    print("Starting LLM System Tests...\n", flush=True)
    try:
        test_tokenizer()
        test_model_shape()
        test_causal_mask()
        test_checkpointing()
        test_overfitting()
        print("====================================")
        print(f"{Fore.GREEN}ALL BASIC UNIT TESTS PASSED!")
        print("====================================")
    except Exception as e:
        print(f"\n{Fore.RED}TEST FAILED: {e}", flush=True)
        import traceback
        traceback.print_exc()
        exit(1)
