# Local LLM Project Documentation

## Overview
AbrobaGPT is a from-scratch implementation of a small, educational Large Language Model based on the GPT
(Generative Pre-trained Transformer) architecture. It is a decoder-only Transformer designed to be trained and
run locally on common hardware. Everything can be driven from the terminal UI (`python main.py`) or from the
command-line scripts below.

## Project Structure
```text
.
├── config/
│   └── base_config.json       # Hyperparameters for the model, training and sampling
├── data/
│   ├── input*.txt             # Raw training text (general / instruct / alpaca)
│   ├── prepare.py             # Trains the tokenizer and writes train.bin / val.bin
│   ├── download_dataset.py    # Streams a Hugging Face dataset to a text file
│   └── processed/             # tokenizer.json, train.bin, val.bin, metadata.json
├── checkpoints/               # Saved model weights (.pt files)
├── src/
│   ├── model.py               # PyTorch implementation of the Transformer
│   ├── tokenizer.py           # BPE tokenizer (stdlib only)
│   ├── data.py                # Dataset download / preparation
│   ├── prompting.py           # Chat prompt format + stop sequences (no torch)
│   ├── inference.py           # Safe checkpoint/tokenizer loading for chat & generate
│   ├── lr_presets.py          # Learning-rate presets used by train.py and the TUI
│   ├── defaults.py            # Defaults for the optional config keys
│   └── utils.py               # Config, device and checkpoint helpers
├── requirements.txt
├── train.py                   # Train the model
├── generate.py                # One-shot text generation
├── chat.py                    # Interactive terminal chat
├── tui.py / main.py           # Terminal UI
└── test_*.py                  # Sanity tests
```

## Getting Started

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Data preparation
Put the raw text in `data/` (or stream it from Hugging Face) and prepare it:
```bash
python data/prepare.py --dataset-type alpaca            # uses data/input_alpaca.txt
python data/download_dataset.py --dataset-type alpaca --size-mb 20
```
Preparation trains the BPE tokenizer and tokenizes the corpus. The processed files are rebuilt automatically
whenever the raw file **or** any of `tokenizer_vocab_size`, `tokenizer_training_chars`,
`tokenizer_min_pair_frequency`, `validation_fraction` changes.

### 3. Training
```bash
python train.py                  # interactive menus
python train.py --resume         # resume the latest compatible checkpoint
```
The training loss shown in the log is the average since the previous evaluation.
Only the newest `keep_checkpoints` periodic checkpoints created by the current run are kept (0 keeps all);
`best_model.pt` is never deleted.

### 4. Text generation
```bash
python generate.py --prompt "Pergunta: O que é um átomo?\nResposta:" --stop "\n\n"
```

### 5. Chat
```bash
python chat.py
```
The chat uses the same `Pergunta: ... / Resposta: ...` format the model was trained on and stops generating at the
end of the answer.

## Configuration (`config/base_config.json`)
- Architecture: `n_layer`, `n_embd`, `n_head` (must divide `n_embd`), `block_size`, `dropout`.
- Training: `batch_size`, `max_iters`, `learning_rate`, `min_learning_rate`, `warmup_iters`, `weight_decay`,
  `grad_clip`, `eval_interval`, `eval_iters`, `save_interval`, early-stopping keys.
- Tokenizer / data: `tokenizer_vocab_size`, `tokenizer_training_chars`, `tokenizer_min_pair_frequency`,
  `validation_fraction`, `hf_default_size_mb`.
- Optional keys (missing = default): `temperature` (0.75) and `keep_checkpoints` (5).

Changing the architecture or the tokenizer makes older checkpoints incompatible; `--resume` will ignore them and
start from scratch.

## Architecture Details
- **Tokenizer**: BPE with a GPT-2 style pre-tokenizer (the leading space is part of the word). `tokenizer.json`
  files created before this change (no `pretokenizer` field) still load with the old behaviour.
- **Model**: decoder-only Transformer with Pre-LayerNorm, causal multi-head attention
  (`scaled_dot_product_attention`), GELU, tied input/output embeddings and GPT-2 style scaled residual init.
- **Optimizer**: AdamW (no weight decay on LayerNorm, biases and embeddings) with linear warmup and cosine decay.
- **Sampling**: temperature scaling, optional top-k and optional stop sequences.

## Madras1 RAG QA Fulltext PT-BR na TUI

- Dataset: `Madras1/rag-qa-fulltext-ptbr`.
- Esquema confirmado: `instruction`, `input` (opcional/vazio) e `output`.
- Na tela **Preparar Dataset**, selecione **Madras1 RAG QA Fulltext PT-BR** e a fonte Hugging Face. O limite de MB limita o número de exemplos baixados; o último exemplo nunca é truncado. Um JSONL local íntegro é reutilizado.
- Brutos: `data/raw/rag-qa-fulltext-ptbr/train.jsonl`; processados: `data/processed/{tokenizer.json,train.bin,val.bin,metadata.json}`.
- A validação é dividida por exemplo completo, com hash determinístico. Metadados registram exemplos válidos/descartados, bytes, tokens e fingerprint do tokenizer.
- Checkpoints anteriores não são sobrescritos pela preparação. Um checkpoint incompatível com o tokenizer novo não pode ser retomado com segurança.
- O dataset está indicado como CC BY-NC 4.0 no cartão Hugging Face; uso comercial não é permitido pela licença indicada.
