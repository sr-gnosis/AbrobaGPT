"""Chat interativo com um checkpoint compatível do AbrobaGPT."""

from __future__ import annotations

import argparse
import sys

import torch
from colorama import Fore, Style, init

from src.inference import InferenceError, load_inference_components, resolve_temperature
from src.prompting import CHAT_STOP_SEQUENCES, build_prompt, clean_response, make_stop_fn


init(autoreset=True)


def chat() -> int:
    parser = argparse.ArgumentParser(description="Conversar com o AbrobaGPT")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint; por padrão procura o melhor/mais recente")
    parser.add_argument("--config", type=str, default="config/base_config.json")
    parser.add_argument("--tokenizer", type=str, default="data/processed/tokenizer.json")
    parser.add_argument("--max-tokens", "--max_tokens", dest="max_tokens", type=int, default=80)
    parser.add_argument(
        "--temp", type=float, default=None,
        help="Temperatura de geração (padrão: 'temperature' do base_config.json, 0.75 se ausente)",
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--allow-legacy-checkpoint", action="store_true")
    args = parser.parse_args()

    if args.max_tokens < 1:
        print(f"{Fore.RED}Erro: --max-tokens deve ser positivo.")
        return 1
    temperature = resolve_temperature(args.config, args.temp)
    if temperature <= 0:
        print(f"{Fore.RED}Erro: --temp deve ser maior que zero.")
        return 1
    try:
        model, tokenizer, device, checkpoint = load_inference_components(
            config_path=args.config,
            tokenizer_path=args.tokenizer,
            checkpoint_path=args.ckpt,
            allow_legacy_checkpoint=args.allow_legacy_checkpoint,
        )
    except InferenceError as exc:
        print(f"{Fore.RED}Erro ao carregar AbrobaGPT: {exc}")
        return 1

    print(f"\n{Fore.YELLOW}--- AbrobaGPT ---{Style.RESET_ALL}")
    print(f"Checkpoint: {checkpoint.name} | Dispositivo: {device} | Temperatura: {temperature:g}")
    print("Digite 'sair', 'exit' ou 'quit' para encerrar.\n")
    turns: list[tuple[str, str]] = []  # (pergunta, resposta) já trocadas
    # Orçamento de tokens para o histórico: o modelo só enxerga block_size tokens.
    prompt_budget = max(model.block_size - args.max_tokens, model.block_size // 2)
    stop_fn = make_stop_fn(tokenizer.decode, CHAT_STOP_SEQUENCES)
    try:
        while True:
            user_input = input(f"{Fore.CYAN}Você: {Style.RESET_ALL}").strip()
            if user_input.lower() in {"sair", "exit", "quit"}:
                break
            if not user_input:
                print(f"{Fore.YELLOW}AbrobaGPT: diga algo para eu responder.")
                continue
            # Mesmo formato do treino: "Pergunta: ...\nResposta: ..."
            prompt = build_prompt(
                turns, user_input, lambda text: len(tokenizer.encode(text)), prompt_budget
            )
            encoded = tokenizer.encode(prompt)
            if not encoded:
                print(f"{Fore.RED}AbrobaGPT: não consegui tokenizar a mensagem.")
                continue
            prompt_tokens = torch.tensor([encoded], dtype=torch.long, device=device)
            with torch.no_grad():
                generated = model.generate(
                    prompt_tokens,
                    max_new_tokens=args.max_tokens,
                    temperature=temperature,
                    top_k=args.top_k,
                    stop_fn=stop_fn,
                )
            response_ids = generated[0, len(encoded) :].tolist()
            response = clean_response(tokenizer.decode(response_ids)) or "…"
            print(f"{Fore.GREEN}AbrobaGPT: {Style.RESET_ALL}{response}")
            turns.append((user_input, response))
    except (EOFError, KeyboardInterrupt):
        print()
    except (ValueError, RuntimeError) as exc:
        print(f"{Fore.RED}Erro durante a geração: {exc}")
        return 1
    print(f"{Fore.YELLOW}Até mais!")
    return 0


if __name__ == "__main__":
    raise SystemExit(chat())
