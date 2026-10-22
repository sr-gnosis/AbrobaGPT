#!/usr/bin/env python3
"""Servidor local do AbrobaGPT: site + menu Dev (chat, treino e checkpoints).

Uso:  python server.py            (abre em http://localhost:8000)
      python server.py --open     (também abre o navegador)
      python server.py --port 9000

Só escuta em 127.0.0.1: nada fica acessível fora do seu computador. Usa apenas a
biblioteca padrão; o torch só é importado quando você conversa ou treina.
O tui.py continua funcionando do jeito de sempre (python main.py).
"""
from __future__ import annotations

import argparse
import json
import math
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

CONFIG = "config/base_config.json"
TOKENIZER = "data/processed/tokenizer.json"
CKPT_RE = re.compile(r"^[\w.\-]+\.pt$")


class ApiError(Exception):
    def __init__(self, message: str, code: int = 400):
        super().__init__(message)
        self.code = code


# ───────────────────────────── modelo (chat) ─────────────────────────────
class Model:
    lock = threading.Lock()
    key = None
    parts = None

    @staticmethod
    def stamp(ckpt):
        d = ROOT / "checkpoints"
        files = [d / ckpt] if ckpt else list(d.glob("*.pt"))
        return tuple(sorted((f.name, f.stat().st_mtime_ns) for f in files if f.exists()))

    @classmethod
    def load(cls, ckpt):
        key = (ckpt, cls.stamp(ckpt))
        if cls.key != key:
            from src.inference import load_inference_components

            cls.parts = load_inference_components(
                config_path=CONFIG,
                tokenizer_path=TOKENIZER,
                checkpoint_path=f"checkpoints/{ckpt}" if ckpt else None,
            )
            cls.key = key
        return cls.parts


def valid_ckpt(name):
    if not name:
        return None
    name = str(name)
    if not CKPT_RE.match(name) or not (ROOT / "checkpoints" / name).is_file():
        raise ApiError("Checkpoint inválido ou inexistente.")
    return name


def default_temperature() -> float:
    try:
        from src.inference import resolve_temperature

        return resolve_temperature(CONFIG, None)
    except Exception:
        return 0.75


def _chat_params(body: dict) -> tuple[str, str | None, int, float, int | None, list[tuple[str, str]]]:
    message = body.get("message", "")
    if not isinstance(message, str) or not message.strip():
        raise ApiError("Digite uma mensagem.")
    message = message.strip()
    ckpt = valid_ckpt(body.get("ckpt"))

    try:
        raw_max = body.get("max_tokens", 80)
        if isinstance(raw_max, bool) or not isinstance(raw_max, (int, float)) or int(raw_max) != raw_max:
            raise ValueError
        max_tokens = int(raw_max)
        raw_temp = body.get("temperature")
        if raw_temp is None or raw_temp == "":
            raw_temp = default_temperature()
        if isinstance(raw_temp, bool) or not isinstance(raw_temp, (int, float)):
            raise ValueError
        temperature = float(raw_temp)
        raw_top_k = body.get("top_k")
        if raw_top_k is None or raw_top_k == "":
            top_k = None
        else:
            if isinstance(raw_top_k, bool) or not isinstance(raw_top_k, (int, float)) or int(raw_top_k) != raw_top_k:
                raise ValueError
            top_k = int(raw_top_k)
    except (TypeError, ValueError, OverflowError):
        raise ApiError("Parâmetros de geração inválidos: max_tokens e top_k devem ser inteiros; temperature deve ser numérica.")

    if not 1 <= max_tokens <= 400:
        raise ApiError("max_tokens deve estar entre 1 e 400.")
    if not math.isfinite(temperature) or not 0 < temperature <= 5:
        raise ApiError("temperature deve ser finita, maior que zero e no máximo 5.")
    if top_k is not None and not 1 <= top_k <= 4000:
        raise ApiError("top_k deve estar entre 1 e 4000 ou ficar vazio.")

    raw_history = body.get("history", [])
    if raw_history is None:
        raw_history = []
    if not isinstance(raw_history, list):
        raise ApiError("history deve ser uma lista de pares pergunta/resposta.")
    history: list[tuple[str, str]] = []
    for pair in raw_history[-8:]:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ApiError("Cada item de history deve conter pergunta e resposta.")
        if not all(isinstance(value, str) for value in pair):
            raise ApiError("As perguntas e respostas do histórico devem ser textos.")
        history.append((pair[0], pair[1]))
    return message, ckpt, max_tokens, temperature, top_k, history


def do_chat(body: dict, on_delta=None) -> dict:
    message, ckpt, requested_tokens, temperature, top_k, history = _chat_params(body)
    try:
        import torch
        from src.inference import InferenceError
        from src.prompting import CHAT_STOP_SEQUENCES, build_prompt, clean_response, make_stop_fn
    except ImportError as exc:
        raise ApiError(f"Dependência ausente ({exc}). Rode: pip install -r requirements.txt", 500)

    started = time.time()
    try:
        with Model.lock:
            model, tokenizer, device, checkpoint = Model.load(ckpt)
            # Reserva inicialmente espaço para a saída; o limite efetivo será
            # recalculado com o tamanho real do prompt tokenizado.
            requested_tokens = min(requested_tokens, max(1, model.block_size - 1))
            budget = max(1, model.block_size - requested_tokens)
            prompt = build_prompt(history, message, lambda t: len(tokenizer.encode(t)), budget)
            encoded = tokenizer.encode(prompt)
            if not encoded:
                raise ApiError("Não consegui tokenizar a mensagem.")
            if len(encoded) > model.block_size:
                raise ApiError(
                    f"A mensagem atual ocupa {len(encoded)} tokens, mas o limite do modelo é {model.block_size}. "
                    "Encurte a mensagem e tente novamente."
                )
            effective_tokens = min(requested_tokens, model.block_size - len(encoded))
            if effective_tokens < 1:
                raise ApiError("A mensagem ocupa todo o contexto do modelo; não há espaço para gerar uma resposta.")
            tokens = torch.tensor([encoded], dtype=torch.long, device=device)
            def stream_token(generated_ids: list[int]) -> None:
                if on_delta is None or not generated_ids:
                    return
                # BPETokenizer.decode concatena as peças; decodificar apenas o último
                # token evita refazer a decodificação do prefixo em cada passo.
                delta = tokenizer.decode([generated_ids[-1]])
                if delta:
                    on_delta(delta)

            out = model.generate(
                tokens,
                max_new_tokens=effective_tokens,
                temperature=temperature,
                top_k=top_k,
                stop_fn=make_stop_fn(tokenizer.decode, CHAT_STOP_SEQUENCES),
                token_callback=stream_token if on_delta is not None else None,
            )
            response = clean_response(tokenizer.decode(out[0, len(encoded):].tolist())) or "…"
    except ApiError:
        raise
    except InferenceError as exc:
        raise ApiError(f"Erro ao carregar o AbrobaGPT: {exc}")
    except (ValueError, RuntimeError) as exc:
        raise ApiError(f"Erro durante a geração: {exc}", 500)
    return {
        "response": response,
        "checkpoint": checkpoint.name,
        "temperature": temperature,
        "device": str(device),
        "seconds": round(time.time() - started, 2),
        "requested_max_tokens": requested_tokens,
        "generated_token_limit": effective_tokens,
    }


# ───────────────────────────── treino ─────────────────────────────
class Trainer:
    proc = None
    lines: deque = deque(maxlen=400)
    progress = ""
    cmd = ""
    returncode = None
    lock = threading.Lock()

    @classmethod
    def running(cls) -> bool:
        return cls.proc is not None and cls.proc.poll() is None

    @classmethod
    def _reader(cls, proc):
        buf = b""
        while True:
            chunk = proc.stdout.read(512)
            if not chunk:
                break
            buf += chunk
            while True:
                m = re.search(rb"[\r\n]", buf)
                if not m:
                    break
                piece, term, buf = buf[: m.start()], buf[m.start(): m.end()], buf[m.end():]
                text = piece.decode("utf-8", "replace").rstrip()
                if not text:
                    continue
                if term == b"\r":
                    cls.progress = text
                else:
                    cls.lines.append(text)
        if buf.strip():
            cls.lines.append(buf.decode("utf-8", "replace").strip())
        proc.wait()
        cls.returncode = proc.returncode
        cls.lines.append(f"[processo encerrado com código {proc.returncode}]")

    @classmethod
    def start(cls, o: dict):
        with cls.lock:
            if cls.running():
                raise ApiError("Já existe um treino em andamento.")
            cmd = [sys.executable, "train.py"]
            source = o.get("source", "local")
            if source not in ("local", "hf"):
                raise ApiError("Fonte inválida.")
            cmd += ["--source", source]
            if o.get("dataset_type") in dataset_types():
                cmd += ["--dataset-type", o["dataset_type"]]
            if o.get("lr_preset") in lr_presets():
                cmd += ["--lr-preset", o["lr_preset"]]
            try:
                iters = int(o.get("iterations_thousands") or 0)
                patience = int(o.get("patience") or 0)
            except (TypeError, ValueError):
                raise ApiError("Iterações e paciência devem ser números inteiros.")
            if iters > 0:
                cmd += ["--max-iters", str(iters * 1000)]
            if o.get("resume"):
                cmd += ["--resume"]
            if o.get("use_existing_data"):
                cmd += ["--use-existing-data"]
            es = o.get("early_stopping")
            if es == "on":
                cmd += ["--early-stopping"]
            elif es == "off":
                cmd += ["--no-early-stopping"]
            if patience > 0 and es != "off":
                cmd += ["--patience", str(patience)]
            env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
            cls.lines.clear()
            cls.progress = ""
            cls.returncode = None
            cls.cmd = " ".join(cmd)
            cls.lines.append("$ " + cls.cmd)
            cls.proc = subprocess.Popen(
                cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
            threading.Thread(target=cls._reader, args=(cls.proc,), daemon=True).start()

    @classmethod
    def stop(cls):
        if cls.running():
            cls.proc.terminate()
            try:
                cls.proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                cls.proc.kill()
            cls.lines.append("[treino interrompido pelo menu Dev]")


def dataset_types() -> list:
    try:
        from src.data import DATASET_TYPES

        return list(DATASET_TYPES)
    except Exception:
        return []


def lr_presets() -> dict:
    try:
        from src.lr_presets import LR_PRESETS, describe_preset

        return {k: f"{v['label']} ({describe_preset(k)})" for k, v in LR_PRESETS.items()}
    except Exception:
        return {}


def metrics() -> list:
    try:
        from monitoring import load_metrics

        _, rows = load_metrics(ROOT / "training_logs")
    except Exception:
        return []
    keep = ("iteration", "total_iterations", "train_loss", "validation_loss", "learning_rate", "perplexity")
    return [{k: r.get(k) for k in keep} for r in rows[-600:]]


def checkpoints() -> list:
    d = ROOT / "checkpoints"
    out = []
    for f in sorted(d.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True):
        st = f.stat()
        out.append({"name": f.name, "mb": round(st.st_size / 1e6, 1), "mtime": int(st.st_mtime)})
    return out


def status() -> dict:
    return {
        "training": {
            "running": Trainer.running(),
            "returncode": Trainer.returncode,
            "progress": Trainer.progress,
            "cmd": Trainer.cmd,
            "lines": list(Trainer.lines)[-80:],
        },
        "metrics": metrics(),
        "checkpoints": checkpoints(),
        "model_loaded": Model.parts is not None,
    }


def options() -> dict:
    return {
        "dataset_types": dataset_types(),
        "lr_presets": lr_presets(),
        "temperature": default_temperature(),
    }


# ───────────────────────────── HTTP ─────────────────────────────
GET_API = {"status": status, "options": options}
POST_API = {
    "chat": do_chat,
    "train/start": lambda b: (Trainer.start(b), {"ok": True})[1],
    "train/stop": lambda b: (Trainer.stop(), {"ok": True})[1],
}


class Handler(BaseHTTPRequestHandler):
    server_version = "AbrobaGPT-local"

    def log_message(self, *args):
        pass

    def _send(self, code, payload, ctype="application/json; charset=utf-8"):
        data = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _guard(self) -> bool:
        """Bloqueia DNS rebinding e requisições vindas de outros sites."""
        host = self.headers.get("Host", "").rsplit(":", 1)[0].strip("[]")
        if host not in ("localhost", "127.0.0.1"):
            self._send(403, {"error": "Host não permitido."})
            return False
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).hostname not in ("localhost", "127.0.0.1"):
            self._send(403, {"error": "Origem não permitida."})
            return False
        return True

    def do_GET(self):
        if not self._guard():
            return
        path = unquote(urlparse(self.path).path)
        if path.startswith("/api/"):
            fn = GET_API.get(path[5:])
            return self._send(200, fn()) if fn else self._send(404, {"error": "Rota inexistente."})
        rel = "index.html" if path == "/" else path.lstrip("/")
        if rel == "dev":
            rel = "dev.html"
        target = (WEB / rel).resolve()
        if WEB not in target.parents or not target.is_file():
            return self._send(404, b"Nao encontrado", "text/plain; charset=utf-8")
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "image/svg+xml"):
            ctype += "; charset=utf-8" if "charset" not in ctype else ""
        self._send(200, target.read_bytes(), ctype)

    def _stream_chat(self):
        """Emite eventos NDJSON: delta incremental e um evento final com metadados."""
        try:
            size = int(self.headers.get("Content-Length") or 0)
            if size > 1_000_000:
                raise ApiError("Requisição grande demais.", 413)
            body = json.loads(self.rfile.read(size) or b"{}")
            if not isinstance(body, dict):
                raise ApiError("Corpo inválido.")
        except ApiError as exc:
            return self._send(exc.code, {"error": str(exc)})
        except (ValueError, json.JSONDecodeError):
            return self._send(400, {"error": "JSON inválido."})

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-store, no-transform")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit(payload):
            data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
            self.wfile.write(data)
            self.wfile.flush()

        try:
            result = do_chat(body, on_delta=lambda delta: emit({"type": "delta", "text": delta}))
            emit({"type": "done", **result})
        except ApiError as exc:
            try:
                emit({"type": "error", "error": str(exc), "status": exc.code})
            except (BrokenPipeError, ConnectionResetError):
                pass
        except Exception as exc:
            try:
                emit({"type": "error", "error": f"Falha na geração: {type(exc).__name__}: {exc}", "status": 500})
            except (BrokenPipeError, ConnectionResetError):
                pass

    def do_POST(self):
        if not self._guard():
            return
        path = unquote(urlparse(self.path).path)
        if path == "/api/chat/stream":
            self._stream_chat()
            return
        fn = POST_API.get(path[5:]) if path.startswith("/api/") else None
        if not fn:
            return self._send(404, {"error": "Rota inexistente."})
        try:
            size = int(self.headers.get("Content-Length") or 0)
            if size > 1_000_000:
                raise ApiError("Requisição grande demais.", 413)
            body = json.loads(self.rfile.read(size) or b"{}")
            if not isinstance(body, dict):
                raise ApiError("Corpo inválido.")
            self._send(200, fn(body))
        except ApiError as exc:
            self._send(exc.code, {"error": str(exc)})
        except json.JSONDecodeError:
            self._send(400, {"error": "JSON inválido."})
        except Exception as exc:  # não derruba o servidor por um erro de rota
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})


def main() -> int:
    parser = argparse.ArgumentParser(description="Site local + menu Dev do AbrobaGPT")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open", action="store_true", help="Abre o navegador ao iniciar")
    args = parser.parse_args()
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as exc:
        print(f"Não consegui usar a porta {args.port}: {exc}\nTente: python server.py --port 8001")
        return 1
    url = f"http://localhost:{args.port}"
    print(f"AbrobaGPT local em {url}  (Dev: {url}/dev.html)  —  Ctrl+C para sair")
    if args.open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        if Trainer.running():
            print("Encerrando o treino em andamento…")
            Trainer.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
