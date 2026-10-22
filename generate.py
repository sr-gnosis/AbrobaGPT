"""Geração única de texto com AbrobaGPT."""

from __future__ import annotations

import argparse
import json

import torch

from src.inference import InferenceError, load_inference_components, resolve_temperature
from src.prompting import cut_at_stop, make_stop_fn


def generate() -> int:
    parser = argparse.ArgumentParser(description="Gerar texto com AbrobaGPT")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--config", type=str, default="config/base_config.json")
    parser.add_argument("--tokenizer", type=str, default="data/processed/tokenizer.json")
    parser.add_argument("--max-tokens", "--max_tokens", dest="max_tokens", type=int, default=100)
    parser.add_argument(
        "--temp", type=float, default=None,
        help="Temperatura (padrão: 'temperature' do base_config.json, 0.75 se ausente)",
    )
    parser.add_argument("--top-k", "--top_k", dest="top_k", type=int, default=None)
    parser.add_argument("--greedy", action="store_true", help="Escolhe sempre o token de maior logit (determinístico)")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Penalidade de repetição (1.0 desativa)")
    parser.add_argument("--repetition-window", type=int, default=64, help="Janela recente de tokens para penalidade")
    parser.add_argument(
        "--stop", action="append", default=None, metavar="TEXTO",
        help="Interrompe a geração ao produzir este texto (repita para várias). Ex.: --stop $'\\n\\n'",
    )
    parser.add_argument("--allow-legacy-checkpoint", action="store_true")
    parser.add_argument("--stream", action="store_true", help="Emite tokens progressivamente (protocolo da TUI)")
    parser.add_argument("--nerd-mode", action="store_true", help="Emite diagnóstico de top-5 tokens por passo")
    args = parser.parse_args()
    if not args.prompt:
        parser.error("--prompt não pode estar vazio")
    if args.max_tokens < 0:
        parser.error("--max-tokens não pode ser negativo")
    if args.top_k is not None and args.top_k < 1:
        parser.error("--top-k deve ser >= 1")
    if args.repetition_penalty < 1.0:
        parser.error("--repetition-penalty deve ser >= 1.0")
    if args.repetition_window < 1:
        parser.error("--repetition-window deve ser >= 1")

    try:
        model, tokenizer, device, _ = load_inference_components(
            config_path=args.config,
            tokenizer_path=args.tokenizer,
            checkpoint_path=args.ckpt,
            allow_legacy_checkpoint=args.allow_legacy_checkpoint,
        )
        encoded = tokenizer.encode(args.prompt)
        if not encoded:
            raise ValueError("Prompt não contém tokens.")
        temperature = resolve_temperature(args.config, args.temp)
        stops = [stop.replace("\\n", "\n") for stop in (args.stop or [])]
        stop_fn = make_stop_fn(tokenizer.decode, stops) if stops else None
        prompt_tokens = torch.tensor([encoded], dtype=torch.long, device=device)
        def on_token(generated_ids: list[int]) -> None:
            if args.stream and generated_ids:
                piece = tokenizer.decode([generated_ids[-1]])
                print("@@ABROBA_TOKEN@@" + json.dumps(piece, ensure_ascii=False), flush=True)

        def on_diagnostics(info: dict) -> None:
            if args.nerd_mode:
                def decorate(entries):
                    return [{"id": token_id, "text": tokenizer.decode([token_id]), "probability": prob}
                            for token_id, prob in entries]
                payload = dict(info)
                for key in ("top5_raw_logits", "top5_after_temperature", "top5_after_top_k"):
                    if key in payload:
                        payload[key] = decorate(payload[key])
                print("@@ABROBA_NERD@@" + json.dumps(payload, ensure_ascii=False), flush=True)

        with torch.no_grad():
            generated = model.generate(
                prompt_tokens,
                max_new_tokens=args.max_tokens,
                temperature=temperature,
                top_k=args.top_k,
                stop_fn=stop_fn,
                token_callback=on_token if args.stream else None,
                diagnostics_callback=on_diagnostics if args.nerd_mode else None,
                greedy=args.greedy,
                repetition_penalty=args.repetition_penalty,
                repetition_window=args.repetition_window,
            )
        new_text = cut_at_stop(tokenizer.decode(generated[0, len(encoded):].tolist()), stops)
        if args.stream:
            print("@@ABROBA_DONE@@" + json.dumps({"prompt_tokens": len(encoded), "generated_tokens": len(generated[0, len(encoded):]), "temperature": temperature, "top_k": args.top_k, "greedy": args.greedy, "repetition_penalty": args.repetition_penalty, "device": str(device)}, ensure_ascii=False), flush=True)
        else:
            print(tokenizer.decode(encoded) + new_text)
        return 0
    except (InferenceError, ValueError, RuntimeError) as exc:
        print(f"Erro: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(generate())
