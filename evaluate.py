"""CLI da avaliação de checkpoints; a TUI consome suas linhas @@ABROBA_EVAL@@."""
from __future__ import annotations
import argparse, json, signal
from threading import Event
from src.evaluation import EvaluationCancelled, EvaluationError, evaluate_checkpoints, export_preset, save_report

def main() -> int:
    parser = argparse.ArgumentParser(description="Avaliar checkpoints AbrobaGPT")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--questions", type=int, default=10); parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=80); parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--temperature", type=float, default=.75); parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=1.0); parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--validation-loss", action="store_true"); parser.add_argument("--allow-legacy-checkpoint", action="store_true")
    parser.add_argument("--report", default=None); parser.add_argument("--export-preset", default=None)
    args = parser.parse_args(); cancelled = Event()
    signal.signal(signal.SIGINT, lambda *_: cancelled.set())
    if args.export_preset:
        export_preset(args.preset, args.export_preset); return 0
    settings = vars(args).copy(); settings.pop("checkpoint"); settings.pop("preset"); settings.pop("report"); settings.pop("export_preset")
    def progress(value): print("@@ABROBA_EVAL@@" + json.dumps(value, ensure_ascii=False), flush=True)
    try:
        report = evaluate_checkpoints(args.checkpoint, args.preset, settings, cancel_event=cancelled, progress=progress)
        path = save_report(report, args.report); print("@@ABROBA_EVAL_DONE@@" + json.dumps({"report": str(path), "recommended": report["recommended_checkpoint"]}, ensure_ascii=False), flush=True); return 0
    except EvaluationCancelled: print("@@ABROBA_EVAL_CANCELLED@@", flush=True); return 130
    except EvaluationError as exc: print(f"Erro: {exc}", flush=True); return 1
if __name__ == "__main__": raise SystemExit(main())
