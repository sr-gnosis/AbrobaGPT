"""Treinamento interativo e reproduzível do AbrobaGPT."""

from __future__ import annotations

import argparse
from datetime import datetime
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Optional, Callable

import gc
import os
import torch
from colorama import Fore, Style, init
from tqdm import tqdm

from monitoring import RunLogger

from src.data import (
    DataError,
    DATASET_TYPES,
    dataset_spec,
    normalize_dataset_type,
    load_processed_data,
    prepare_if_needed,
    raw_path_for,
    stream_dataset_to_file,
)
from src.lr_presets import LR_PRESETS, describe_preset, resolve_lr_preset
from src.model import GPT, ModelConfigError
from src.utils import (
    CheckpointError,
    ConfigError,
    checkpoint_compatibility,
    find_latest_compatible_checkpoint,
    get_device,
    inspect_checkpoint,
    load_checkpoint,
    load_config,
    project_path,
    safe_checkpoint_path,
    save_checkpoint,
)


init(autoreset=True)


def log(message: str, color: str = "", callback: Optional[Callable[[str], None]] = None) -> None:
    timestamp = datetime.now().strftime("[%H:%M:%S]")
    formatted = f"{color}{timestamp} {message}{Style.RESET_ALL}"
    if callback:
        callback(formatted)
    tqdm.write(formatted)


class WarmupCosineScheduler:
    """Scheduler serializável: warmup linear seguido de decaimento cosseno."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        peak_lr: float,
        min_lr: float,
        warmup_iters: int,
        max_iters: int,
        cosine_decay: bool = True,
    ) -> None:
        self.optimizer = optimizer
        self.peak_lr = float(peak_lr)
        self.min_lr = float(min_lr)
        self.warmup_iters = max(0, int(warmup_iters))
        self.max_iters = max(1, int(max_iters))
        self.cosine_decay = bool(cosine_decay)
        self.last_iter = -1

    def learning_rate(self, iteration: int) -> float:
        iteration = max(0, int(iteration))
        if self.warmup_iters and iteration < self.warmup_iters:
            return self.peak_lr * (iteration + 1) / self.warmup_iters
        if not self.cosine_decay:
            return self.peak_lr
        if self.max_iters <= self.warmup_iters or iteration >= self.max_iters:
            return self.min_lr
        ratio = (iteration - self.warmup_iters) / (self.max_iters - self.warmup_iters)
        ratio = min(max(ratio, 0.0), 1.0)
        coefficient = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return self.min_lr + coefficient * (self.peak_lr - self.min_lr)

    def step(self, iteration: int) -> float:
        self.last_iter = int(iteration)
        lr = self.learning_rate(iteration)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def state_dict(self) -> dict[str, Any]:
        return {
            "peak_lr": self.peak_lr,
            "min_lr": self.min_lr,
            "warmup_iters": self.warmup_iters,
            "max_iters": self.max_iters,
            "cosine_decay": self.cosine_decay,
            "last_iter": self.last_iter,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        # A duração total da execução atual pode mudar ao retomar; a posição fica.
        self.last_iter = int(state.get("last_iter", -1))
        if self.last_iter >= 0:
            self.step(self.last_iter)


def _menu_choice(title: str, options: Mapping[str, str], default: str) -> str:
    print(f"\n{Fore.YELLOW}{title}{Style.RESET_ALL}")
    for key, label in options.items():
        print(f"  [{key}] {label}")
    raw = input(f"Escolha [{default}]: ").strip()
    if not raw:
        return default
    if raw not in options:
        log(f"Opção inválida '{raw}'. Usando {default}.", Fore.YELLOW)
        return default
    return raw


def _choose_dataset_type(provided: Optional[str]) -> str:
    if provided:
        return normalize_dataset_type(provided)
    if not sys.stdin.isatty():
        return "general"
    choice = _menu_choice(
        "Tipo de corpus:",
        {
            "1": "Corpus PT-BR geral (Madras1/corpus-ptbr-v1)",
            "2": "Q&A / Instruct (emdemor/ptbr-question-and-answer)",
            "3": "Alpaca PT-BR (dominguesm/alpaca-data-pt-br)",
            "4": "Madras1 RAG QA Fulltext PT-BR (Madras1/rag-qa-fulltext-ptbr)",
        },
        "1",
    )
    return {"1": "general", "2": "instruct", "3": "alpaca", "4": "ragqa"}[choice]


def _choose_iterations(args: argparse.Namespace, config: Mapping[str, Any]) -> int:
    if args.max_iters is not None:
        if args.max_iters > 0:
            return args.max_iters
        log("--max-iters inválido; usando valor padrão da configuração.", Fore.YELLOW)
        return int(config["max_iters"])
    raw = args.iterations_thousands
    if raw is None and sys.stdin.isatty():
        raw = input(
            f"\nQuantos milhares de iterações deseja treinar? (ex.: 5 = 5.000; padrão {int(config['max_iters']) // 1000}): "
        ).strip()
    if raw is None or not str(raw).strip():
        return int(config["max_iters"])
    try:
        total = int(round(float(str(raw).replace(",", ".")) * 1000))
        if total <= 0:
            raise ValueError
        return total
    except ValueError:
        log("Quantidade inválida; usando max_iters da configuração.", Fore.YELLOW)
        return int(config["max_iters"])


def _select_and_prepare_data(args: argparse.Namespace, config: Mapping[str, Any], log_callback: Optional[Callable[[str], None]] = None) -> None:
    if args.use_existing_data:
        return
    source = args.source
    if source is None:
        if not sys.stdin.isatty():
            raise DataError(
                "Nenhuma fonte de dados informada em modo não interativo. "
                "Use --source local/hf ou --use-existing-data."
            )
        choice = _menu_choice(
            "Fonte de dados:", {"1": "Arquivo local", "2": "Streaming do Hugging Face"}, "1"
        )
        source = "local" if choice == "1" else "hf"
    dataset_type = _choose_dataset_type(args.dataset_type)
    spec = dataset_spec(dataset_type)

    if source == "local":
        raw_path = project_path(args.local_file) if args.local_file else raw_path_for(dataset_type)
        log(f"Fonte local: {raw_path} ({spec['label']})", Fore.CYAN, callback=log_callback)
        rebuilt, metadata = prepare_if_needed(raw_path, config, dataset_type=dataset_type)
    else:
        size_mb = args.size_mb
        if size_mb is None and sys.stdin.isatty():
            raw_size = input(f"Limite de download em MB [padrão {config['hf_default_size_mb']}]: ").strip()
            if raw_size:
                try:
                    size_mb = float(raw_size.replace(",", "."))
                except ValueError:
                    log("Tamanho inválido; usando o padrão da configuração.", Fore.YELLOW, callback=log_callback)
        size_mb = float(config["hf_default_size_mb"]) if size_mb is None else float(size_mb)
        raw_path = raw_path_for(dataset_type)
        dataset_id = args.dataset or spec["dataset"]
        # A configuração explícita vence; o perfil oficial RAG QA usa qa_pairs.
        # Não aplicamos esse padrão quando o usuário troca o id do dataset.
        dataset_config = args.dataset_config if args.dataset_config is not None else (
            spec.get("config") if dataset_id == spec["dataset"] else None
        )
        config_detail = f" (configuração: {dataset_config})" if dataset_config else ""
        log(f"Streaming de {dataset_id}{config_detail} até {size_mb:g} MB...", Fore.CYAN, callback=log_callback)
        result = stream_dataset_to_file(
            dataset_id,
            raw_path,
            dataset_type=dataset_type,
            size_mb=size_mb,
            split=args.split,
            dataset_config=dataset_config,
        )
        cache_label = " (reutilizado do disco)" if result.get("cached") else ""
        detail = f", {result.get('duplicates', 0)} duplicatas removidas, {result.get('skipped', 0)} inválidos ignorados" if dataset_type == "ragqa" else ""
        log(
            f"Download concluído{cache_label}: {result['records']} exemplos, "
            f"{result['size_bytes'] / (1024 * 1024):.2f} MB{detail}.",
            Fore.GREEN, callback=log_callback
        )
        rebuilt, metadata = prepare_if_needed(raw_path, config, dataset_type=dataset_type, force=True)

    action = "preparado" if rebuilt else "já estava atualizado"
    log(
        f"Corpus {action}: vocab={metadata['vocab_size']}, "
        f"treino={metadata['train_tokens']} tokens, validação={metadata['val_tokens']} tokens.",
        Fore.GREEN, callback=log_callback,
    )
    if dataset_type == "ragqa":
        log(
            f"Madras1 RAG QA: {metadata.get('valid_examples', result.get('records', 0) if source == 'hf' else '—')} exemplos válidos; "
            f"{metadata.get('discarded_examples', 0)} descartados; "
            f"{metadata.get('raw_bytes', 0):,} bytes brutos.",
            Fore.CYAN, callback=log_callback,
        )


def _make_batch(data: torch.Tensor, sequence_length: int, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    maximum_start = len(data) - sequence_length - 1
    if maximum_start < 0:
        raise DataError("Não há tokens suficientes para formar um batch.")

    # Geramos todos os índices de início de uma vez
    starts = torch.randint(0, maximum_start + 1, (batch_size,))

    # Usamos indexação avançada para extrair os batches de forma mais eficiente
    # Criamos uma matriz de índices (batch_size, sequence_length)
    offsets = torch.arange(sequence_length)
    indices_x = starts.unsqueeze(1) + offsets
    indices_y = indices_x + 1

    x = data[indices_x].to(device, non_blocking=device.type == "cuda")
    y = data[indices_y].to(device, non_blocking=device.type == "cuda")

    return x, y


@torch.no_grad()
def _estimate_validation_loss(
    model: GPT,
    val_data: torch.Tensor,
    sequence_length: int,
    batch_size: int,
    eval_iters: int,
    device: torch.device,
    amp_enabled: bool,
) -> float:
    model.eval()
    losses: list[float] = []
    for _ in range(max(1, eval_iters)):
        xb, yb = _make_batch(val_data, sequence_length, batch_size, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            _, loss = model(xb, yb)
        assert loss is not None
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)


def _save_training_checkpoint(
    desired: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    scaler: Optional[torch.amp.GradScaler],
    iteration: int,
    config: Mapping[str, Any],
    tokenizer: Any,
    best_val_loss: float,
    patience_counter: int,
) -> Path:
    destination = safe_checkpoint_path(desired, config, tokenizer)
    save_checkpoint(
        model,
        optimizer,
        iteration,
        destination,
        scheduler=scheduler,
        scaler=scaler,
        config=config,
        tokenizer=tokenizer,
        best_val_loss=best_val_loss,
        patience_counter=patience_counter,
    )
    return destination


def _apply_explicit_training_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    """Aplica somente argumentos que foram informados para esta execução."""
    if args.early_stopping is not None:
        config["early_stopping"] = args.early_stopping
    if args.patience is not None:
        if args.patience < 0:
            raise ConfigError("'patience' não pode ser negativa.")
        config["patience"] = args.patience


def _apply_lr_preset(
    config: dict[str, Any], preset: Optional[str], max_iters: int, log_callback: Optional[Callable[[str], None]] = None
) -> None:
    """Sobrescreve LR de pico/mínimo/warmup quando um preset foi escolhido."""
    if not preset:
        log(
            f"Learning rate (config): pico {float(config['learning_rate']):.0e} | "
            f"mínimo {float(config['min_learning_rate']):.0e} | warmup {int(config['warmup_iters']):,} iterações.",
            Fore.CYAN, callback=log_callback,
        )
        return
    config.update(resolve_lr_preset(preset, max_iters))
    log(
        f"Learning rate: {LR_PRESETS[preset]['label']} ({describe_preset(preset)}) -> "
        f"pico {config['learning_rate']:.0e} | mínimo {config['min_learning_rate']:.0e} | "
        f"warmup {config['warmup_iters']:,} iterações.",
        Fore.CYAN, callback=log_callback,
    )


def train(log_callback: Optional[Callable[[str], None]] = None) -> int:
    parser = argparse.ArgumentParser(description="Treinar o AbrobaGPT")
    parser.add_argument("--config", default="config/base_config.json")
    parser.add_argument("--resume", action="store_true", help="Retoma o checkpoint compatível mais recente")
    parser.add_argument("--allow-legacy-checkpoint", action="store_true", help="Permite checkpoint sem fingerprint")
    parser.add_argument("--source", choices=("local", "hf"), default=None)
    parser.add_argument("--dataset-type", choices=(*DATASET_TYPES, "canarim"), default=None)
    parser.add_argument("--local-file", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--size-mb", type=float, default=None)
    parser.add_argument("--use-existing-data", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--iterations-thousands", type=str, default=None)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--nerd-mode", action="store_true", help="Exibe métricas detalhadas de treinamento")
    parser.add_argument(
        "--lr-preset", choices=tuple(LR_PRESETS), default=None,
        help="Preset de learning rate: baixo, medio ou alto (padrão: valores do config)",
    )
    early_stopping_group = parser.add_mutually_exclusive_group()
    early_stopping_group.add_argument(
        "--early-stopping", dest="early_stopping", action="store_true", help="Ativa o Early Stopping"
    )
    early_stopping_group.add_argument(
        "--no-early-stopping", dest="early_stopping", action="store_false", help="Desativa o Early Stopping"
    )
    parser.set_defaults(early_stopping=None)
    parser.add_argument("--patience", type=int, default=None, help="Paciência para Early Stopping")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        _apply_explicit_training_overrides(config, args)

        if args.use_existing_data:
            log("Usando dados processados existentes...", Fore.CYAN, callback=log_callback)
            tokenizer, train_array, val_array, _ = load_processed_data()
        else:
            _select_and_prepare_data(args, config, log_callback=log_callback)
            if args.prepare_only:
                log("Preparação concluída; treinamento não iniciado (--prepare-only).", Fore.GREEN, callback=log_callback)
                return 0
            tokenizer, train_array, val_array, _ = load_processed_data()
        config["vocab_size"] = tokenizer.vocab_size
        max_iters = _choose_iterations(args, config)
        config["max_iters"] = max_iters
        _apply_lr_preset(config, args.lr_preset, max_iters, log_callback)
        early_stopping_enabled = bool(config["early_stopping"])
        device = get_device(config["device"])
        amp_enabled = bool(config["amp"]) and not args.no_amp and device.type == "cuda"
        log(f"AbrobaGPT | dispositivo: {device} | AMP: {'ativo' if amp_enabled else 'desativado'}", Fore.CYAN, callback=log_callback)
        log(
            f"Tokenizer BPE: {tokenizer.vocab_size} tokens | fingerprint {tokenizer.fingerprint[:12]}...",
            Fore.CYAN, callback=log_callback
        )
        log(f"Total escolhido: {max_iters:,} iterações.", Fore.CYAN, callback=log_callback)
        if early_stopping_enabled:
            log("Early Stopping: ON", Fore.CYAN, callback=log_callback)
            log(f"Patience: {int(config['patience'])}", Fore.CYAN, callback=log_callback)
        else:
            log("Early Stopping: OFF", Fore.CYAN, callback=log_callback)

        torch.manual_seed(int(config["seed"]))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(config["seed"]))
        train_data = torch.from_numpy(train_array).to(torch.long)
        val_data = torch.from_numpy(val_array).to(torch.long)
        train_sequence = min(int(config["block_size"]), len(train_data) - 1)
        val_sequence = min(int(config["block_size"]), len(val_data) - 1)
        if train_sequence < 1 or val_sequence < 1:
            raise DataError("Corpus insuficiente para batches de treino e validação.")
        if train_sequence < int(config["block_size"]) or val_sequence < int(config["block_size"]):
            log(
                f"Corpus curto: usando sequências de treino={train_sequence} e validação={val_sequence} tokens.",
                Fore.YELLOW,
            )




        model = GPT(config).to(device)
        optimizer = model.configure_optimizer(
            learning_rate=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
        )
        scheduler = WarmupCosineScheduler(
            optimizer,
            peak_lr=float(config["learning_rate"]),
            min_lr=float(config["min_learning_rate"]),
            warmup_iters=int(config["warmup_iters"]),
            max_iters=max_iters,
            cosine_decay=bool(config.get("cosine_decay", True)),
        )
        scaler: Optional[torch.amp.GradScaler] = (
            torch.amp.GradScaler("cuda", enabled=True) if amp_enabled else None
        )

        checkpoint_dir = project_path(args.checkpoint_dir)
        start_iter = 0
        best_val_loss = float("inf")
        best_val_iteration = 0
        patience_counter = 0
        if args.resume:
            latest, rejected = find_latest_compatible_checkpoint(
                checkpoint_dir,
                config,
                tokenizer,
                allow_legacy=args.allow_legacy_checkpoint,
            )
            for reason in rejected:
                log(f"Checkpoint ignorado: {reason}", Fore.YELLOW)
            if latest is None:
                log("Nenhum checkpoint compatível encontrado; iniciando do zero.", Fore.YELLOW)
            else:
                summary = inspect_checkpoint(latest)
                compatible, reason = checkpoint_compatibility(
                    summary,
                    config,
                    tokenizer,
                    allow_legacy=args.allow_legacy_checkpoint,
                )
                if not compatible:
                    raise CheckpointError(reason)
                start_iter = load_checkpoint(
                    model, optimizer, latest, scheduler=scheduler, scaler=scaler, map_location=device
                )
                payload = summary["payload"]
                best_val_loss = float(payload.get("best_val_loss", best_val_loss))
                best_val_iteration = start_iter if math.isfinite(best_val_loss) else 0
                patience_counter = int(payload.get("patience_counter", 0))
                log(f"Retomado de {latest.name} na iteração {start_iter:,}. {reason}", Fore.GREEN)

        if start_iter >= max_iters:
            log("O checkpoint já atingiu o total solicitado; nada a treinar.", Fore.YELLOW)
            return 0

        run_logger = RunLogger(project_path("training_logs"), start_iteration=start_iter)
        run_logger.write(f"Início do treinamento; total={max_iters}; resume={bool(args.resume)}")
        log(f"Logs desta execução: {run_logger.directory}", Fore.CYAN, callback=log_callback)
        log("Iniciando treinamento...", Fore.GREEN)
        keep_checkpoints = int(config.get("keep_checkpoints", 0))
        periodic_saved: list[Path] = []  # só os checkpoints criados por esta execução são rotacionados
        loss_sum = torch.zeros((), device=device)
        loss_count = 0
        saved_iteration = -1
        completed = start_iter
        stopped_early = False
        nerd_started_at = time.perf_counter()
        last_grad_norm: Optional[float] = None
        if device.type == "cuda" and args.nerd_mode:
            torch.cuda.reset_peak_memory_stats(device)
        with tqdm(total=max_iters, initial=start_iter, unit="it", dynamic_ncols=True, desc="Treino") as progress:
            for iteration in range(start_iter, max_iters):
                lr = scheduler.step(iteration)
                xb, yb = _make_batch(train_data, train_sequence, int(config["batch_size"]), device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    _, loss = model(xb, yb)
                assert loss is not None
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if float(config["grad_clip"]) > 0:
                        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"]))
                        last_grad_norm = float(grad_norm_tensor.detach().item())
                    elif args.nerd_mode:
                        last_grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")).detach().item())
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if float(config["grad_clip"]) > 0:
                        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"]))
                        last_grad_norm = float(grad_norm_tensor.detach().item())
                    elif args.nerd_mode:
                        last_grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")).detach().item())
                    optimizer.step()

                loss_sum += loss.detach().float()
                loss_count += 1
                completed = iteration + 1
                progress.update(1)
                if completed % int(config["eval_interval"]) == 0 or completed == 1:
                    val_loss = _estimate_validation_loss(
                        model,
                        val_data,
                        val_sequence,
                        int(config["batch_size"]),
                        int(config["eval_iters"]),
                        device,
                        amp_enabled,
                    )
                    perplexity = math.exp(min(val_loss, 20.0))
                    # Média do loss de treino desde a última avaliação (um batch só é muito ruidoso).
                    train_loss = float(loss_sum.item()) / max(1, loss_count)
                    loss_sum.zero_()
                    loss_count = 0
                    if val_loss < best_val_loss:
                        prospective_best = val_loss
                        prospective_best_iter = completed
                    else:
                        prospective_best = best_val_loss
                        prospective_best_iter = best_val_iteration
                    run_logger.metric(iteration=completed, total_iterations=max_iters, train_loss=train_loss, validation_loss=val_loss, learning_rate=lr, perplexity=perplexity, best_validation_loss=prospective_best, best_iteration=prospective_best_iter, early_stopping=False)
                    run_logger.write(f"Avaliação iteração={completed}/{max_iters} train_loss={train_loss:.8g} val_loss={val_loss:.8g} lr={lr:.8g} perplexity={perplexity:.8g}")
                    progress.set_postfix(loss=f"{train_loss:.4f}", val=f"{val_loss:.4f}", lr=f"{lr:.2e}")
                    log(
                        f"Iter {completed:,}/{max_iters:,} | loss {train_loss:.4f} | "
                        f"val {val_loss:.4f} | perplexidade {perplexity:.2f} | lr {lr:.3e}"
                    )
                    if args.nerd_mode:
                        elapsed = max(0.001, time.perf_counter() - nerd_started_at)
                        tokens_per_second = (completed - start_iter) * int(config["batch_size"]) * train_sequence / elapsed
                        remaining_seconds = max(0, max_iters - completed) * elapsed / max(1, completed - start_iter)
                        memory = "métricas de memória GPU indisponíveis (dispositivo CPU)"
                        if device.type == "cuda":
                            allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
                            peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                            memory = f"GPU alocada={allocated:.0f} MiB; pico={peak:.0f} MiB"
                        log(
                            "[PARA NERDS] "
                            f"elapsed={elapsed:.1f}s | ETA≈{remaining_seconds:.1f}s | "
                            f"tokens/s≈{tokens_per_second:.1f} | grad_norm={last_grad_norm if last_grad_norm is not None else 'n/a'} | "
                            f"device={device} | {memory} | checkpoint_dir={checkpoint_dir}",
                            Fore.MAGENTA, callback=log_callback,
                        )
                    if val_loss < best_val_loss:
                        previous = best_val_loss
                        best_val_loss = val_loss
                        best_val_iteration = completed
                        patience_counter = 0
                        best_path = _save_training_checkpoint(
                            checkpoint_dir / "best_model.pt", model, optimizer, scheduler, scaler, completed,
                            config, tokenizer, best_val_loss, patience_counter,
                        )
                        detail = "primeira validação" if math.isinf(previous) else f"{previous:.4f} -> {val_loss:.4f}"
                        log(f"Validation loss melhorou ({detail}); melhor modelo salvo em {best_path.name}.", Fore.GREEN)
                        run_logger.write(f"Novo melhor validation loss={best_val_loss:.8g} na iteração {completed}; checkpoint={best_path.name}")
                    elif early_stopping_enabled:
                        if completed < int(config["early_stopping_warmup_iters"]):
                            log(
                                "Validation loss não melhorou; early stopping ainda em warmup "
                                f"até a iteração {config['early_stopping_warmup_iters']}.", Fore.YELLOW,
                            )
                        else:
                            patience_counter += 1
                            log(
                                "Validation loss piorou/não melhorou; "
                                f"contador {patience_counter}/{config['patience']}.", Fore.YELLOW,
                            )
                            if patience_counter >= int(config["patience"]):
                                log(
                                    "Early stopping acionado: não houve melhora de validation loss "
                                    f"por {patience_counter} avaliações após o warmup.", Fore.RED,
                                )
                                stopped_early = True
                                run_logger.metric(iteration=completed, total_iterations=max_iters, best_validation_loss=best_val_loss, early_stopping=True)
                                run_logger.write(f"EARLY STOPPING na iteração {completed}: patience={patience_counter}; melhor val loss={best_val_loss:.8g}")

                if completed % int(config["save_interval"]) == 0:
                    periodic_path = _save_training_checkpoint(
                        checkpoint_dir / f"ckpt_{completed:06d}.pt", model, optimizer, scheduler, scaler,
                        completed, config, tokenizer, best_val_loss, patience_counter,
                    )
                    saved_iteration = completed
                    log(f"Checkpoint periódico salvo: {periodic_path.name}", Fore.GREEN)
                    run_logger.write(f"Checkpoint periódico salvo: {periodic_path.name} na iteração {completed}")
                    periodic_saved.append(periodic_path)
                    while keep_checkpoints > 0 and len(periodic_saved) > keep_checkpoints:
                        obsolete = periodic_saved.pop(0)
                        try:
                            obsolete.unlink(missing_ok=True)
                            log(f"Checkpoint antigo removido (mantendo {keep_checkpoints}): {obsolete.name}", Fore.CYAN)
                        except OSError as exc:
                            log(f"Não foi possível remover {obsolete.name}: {exc}", Fore.YELLOW)
                if stopped_early:
                    break

        if completed != saved_iteration:
            final_path = _save_training_checkpoint(
                checkpoint_dir / f"ckpt_{completed:06d}.pt", model, optimizer, scheduler, scaler,
                completed, config, tokenizer, best_val_loss, patience_counter,
            )
            log(f"Checkpoint final salvo: {final_path.name}", Fore.GREEN)
            run_logger.write(f"Checkpoint final salvo: {final_path.name} na iteração {completed}")
        log("Treinamento concluído.", Fore.GREEN)
        run_logger.close("early_stopping" if stopped_early else "concluído")
        return 0
    except (ConfigError, DataError, ModelConfigError, CheckpointError, OSError) as exc:
        if "run_logger" in locals(): run_logger.close(f"erro: {exc}")
        log(f"Erro: {exc}", Fore.RED)
        return 1
    except KeyboardInterrupt:
        if "run_logger" in locals(): run_logger.close("interrompido por KeyboardInterrupt")
        log("Treinamento interrompido pelo usuário.", Fore.YELLOW)
        return 130
    except Exception as exc:
        if "run_logger" in locals(): run_logger.close(f"erro inesperado: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(train())
