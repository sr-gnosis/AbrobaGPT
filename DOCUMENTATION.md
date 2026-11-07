# AbrobaGPT Documentation

## 🌟 Overview
AbrobaGPT is a from-scratch implementation of a Large Language Model (LLM) based on the GPT (Generative Pre-trained Transformer) architecture. It is a decoder-only Transformer designed to be trained and run locally. 

The project features a sophisticated Terminal User Interface (TUI) that orchestrates the entire pipeline: from data preparation to training and final evaluation.

## 🛠️ Project Structure
```text
.
├── config/                  # Hyperparameters (base_config.json)
├── data/                   # Dataset scripts and processed binaries
│   ├── prepare.py          # Tokenizer training and corpus binarization
│   ├── download_dataset.py   # Hugging Face streaming utilities
│   └── processed/          # Saved tokenizer.json, train.bin, val.bin
├── checkpoints/             # Model weights (.pt files)
├── src/                    # Core logic
│   ├── model.py            # Transformer architecture (PyTorch)
│   ├── tokenizer.py        # BPE Tokenizer implementation
│   ├── data.py             # Data loading and processing
│   ├── prompting.py        # Chat templates and stop sequences
│   ├── inference.py        # Model loading and generation logic
│   ├── evaluation.py       # Quantitative benchmarking and scoring
│   ├── lr_presets.py       # Learning rate schedules
│   ├── defaults.py         # Default configuration values
│   └── utils.py            # General helpers
├── evaluation_presets/     # JSON files for different test scenarios
├── training_logs/          # Training metrics (CSV/JSONL) and logs
├── main.py                 # Entry point for the TUI
├── tui.py                  # TUI Implementation (Curses)
├── train.py                # Training loop script
├── generate.py             # Text generation CLI
├── evaluate.py             # Model evaluation CLI
├── chat.py                 # Interactive chat CLI
└── tests/                  # Unit and regression tests
```

## 🚀 Getting Started

### 1. Installation
```bash
pip install -r requirements.txt
```
*Note: Windows users will automatically install `windows-curses` via requirements.*

### 2. Data Preparation
Use the TUI (`python main.py` -> Prepare Dataset) or the CLI:
```bash
python train.py --prepare-only --dataset-type ragqa --source hf
```
This trains the BPE tokenizer and converts the raw text into binary files for fast loading during training.

### 3. Training
Start the training process via the TUI or CLI:
```bash
python train.py --resume
```
The system supports **AMP (Automatic Mixed Precision)** and **Early Stopping** based on validation loss.

### 4. Evaluation
Compare checkpoints using defined presets:
```bash
python evaluate.py --preset evaluation_presets/conhecimento_geral.json --checkpoint checkpoints/best_model.pt
```

### 5. Generation & Chat
```bash
# One-shot generation
python generate.py --prompt "O futuro da IA é" --ckpt checkpoints/model.pt

# Interactive chat
python chat.py
```

## ⚙️ Configuration (`config/base_config.json`)
- **Architecture**: `n_layer`, `n_embd`, `n_head`, `block_size`, `dropout`.
- **Training**: `batch_size`, `max_iters`, `learning_rate`, `min_learning_rate`, `warmup_iters`, `weight_decay`, `grad_clip`, `eval_interval`, `eval_iters`, `save_interval`.
- **Early Stopping**: `early_stopping` (bool) and `patience` (int).
- **Tokenizer**: `tokenizer_vocab_size`, `tokenizer_training_chars`, `tokenizer_min_pair_frequency`.

## 🧠 Architecture Details
- **Tokenizer**: BPE with GPT-2 style pre-tokenization.
- **Model**: Decoder-only Transformer with Pre-LayerNorm, Causal Multi-Head Attention, GELU activation, and tied embeddings.
- **Optimizer**: AdamW with linear warmup and cosine decay.
- **Sampling**: Temperature scaling and Top-K filtering.

## 📊 Madras1 RAG QA Fulltext PT-BR
The project includes native support for the `Madras1/rag-qa-fulltext-ptbr` dataset, utilizing a deterministic hash-based validation split to ensure consistency across different training runs.
