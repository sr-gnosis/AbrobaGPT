#!/usr/bin/env python3
"""TUI (Text User Interface) completa para o AbrobaGPT."""

from __future__ import annotations

import argparse
import base64
import curses
import json
import locale
import math
import os
import re
import subprocess
import sys
import signal
import textwrap
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

# ─────────────────────────── constantes de cor ───────────────────────────────
# Pares de cor (serão inicializados em init_colors)
C_NORMAL    = 0
C_TITLE     = 1   # Ciano brilhante em preto
C_SELECTED  = 2   # Preto em ciano
C_BORDER    = 3   # Azul em preto
C_SUCCESS   = 4   # Verde em preto
C_WARNING   = 5   # Amarelo em preto
C_ERROR     = 6   # Vermelho em preto
C_DIM       = 7   # Branco escuro em preto
C_HEADER    = 8   # Preto em azul
C_PROMPT    = 9   # Magenta em preto
C_INPUT     = 10  # Branco brilhante em azul escuro
C_LOG       = 11  # Ciano escuro em preto

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.defaults import DEFAULT_TEMPERATURE, OPTIONAL_CONFIG_DEFAULTS
from src.lr_presets import LR_PRESETS, describe_preset
from src.prompting import CHAT_STOP_SEQUENCES, build_prompt, clean_response, make_stop_fn
from src.evaluation import copy_selected_checkpoint, list_presets, load_examples
from monitoring import load_metrics, braille_plot

# Opções do campo "Learning rate": None = usar o que está no config/base_config.json
LR_KEYS   = [None, *LR_PRESETS]
LR_LABELS = ["Padrão (config)", *[LR_PRESETS[k]["label"] for k in LR_PRESETS]]

# Tipos de dataset (mesmos nomes de DATASET_SPECS em src/data.py)
DATASET_TYPES  = ["general", "instruct", "alpaca", "ragqa"]
DATASET_LABELS = ["Corpus PT-BR geral", "Q&A / Instruct", "Alpaca PT-BR (dominguesm)", "Madras1 RAG QA Fulltext PT-BR"]


def cycle_index(idx: int, key: int, size: int) -> int:
    """← volta uma opção; →/Espaço avança (circular)."""
    step = -1 if key == curses.KEY_LEFT else 1
    return (idx + step) % size


# ──────────────────────────── utilitários ────────────────────────────────────
def init_colors() -> None:
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(C_TITLE,    curses.COLOR_CYAN,    -1)
    curses.init_pair(C_SELECTED, curses.COLOR_BLACK,   curses.COLOR_CYAN)
    curses.init_pair(C_BORDER,   curses.COLOR_BLUE,    -1)
    curses.init_pair(C_SUCCESS,  curses.COLOR_GREEN,   -1)
    curses.init_pair(C_WARNING,  curses.COLOR_YELLOW,  -1)
    curses.init_pair(C_ERROR,    curses.COLOR_RED,     -1)
    curses.init_pair(C_DIM,      curses.COLOR_WHITE,   -1)
    curses.init_pair(C_HEADER,   curses.COLOR_BLACK,   curses.COLOR_BLUE)
    curses.init_pair(C_PROMPT,   curses.COLOR_MAGENTA, -1)
    curses.init_pair(C_INPUT,    curses.COLOR_WHITE,   curses.COLOR_BLUE)
    curses.init_pair(C_LOG,      curses.COLOR_CYAN,    -1)
    # Orange for the logo
    # We use a mix of red and yellow or just a bright red if true orange isn't available
    # Most terminals map COLOR_YELLOW to a golden/yellow.
    # COLOR_RED is clearly red.
    # To get something closer to "orange" in standard 8-color curses,
    # we often use COLOR_YELLOW or a custom color if the terminal supports it.
    # Let's try a more vibrant approach or stick to YELLOW if that's the best 'orange'.
    # Since the user says YELLOW is "weird", let's try COLOR_RED or a custom color.
    try:
        curses.init_pair(100, 208, -1) # 208 is a common xterm orange
    except:
        curses.init_pair(100, curses.COLOR_RED, -1) # Fallback to Red



def cp(pair: int, bold: bool = False) -> int:
    attr = curses.color_pair(pair)
    if bold:
        attr |= curses.A_BOLD
    return attr


def draw_box(win: "curses._CursesWindow", y: int, x: int, h: int, w: int,
             title: str = "", color: int = 0) -> None:
    """Desenha uma caixa com bordas Unicode."""
    attr = cp(C_BORDER) if color == 0 else color
    try:
        # No Windows/curses, hline()/addch() podem tentar converter
        # caracteres Unicode para um chtype de 1 byte, causando:
        # OverflowError: byte doesn't fit in chtype.
        # addstr(), por outro lado, passa pela camada wide-character e
        # preserva os caracteres Unicode da interface.
        top = "═" * max(0, w - 2)
        bottom = top
        win.addstr(y, x, "╔" + top + "╗", attr)
        win.addstr(y+h-1, x, "╚" + bottom + "╝", attr)
        side = "║"
        for row in range(y+1, y+h-1):
            win.addstr(row, x, side, attr)
            win.addstr(row, x+w-1, side, attr)
        if title:
            label = f" {title} "
            tx = x + (w - len(label)) // 2
            win.addstr(y, tx, label, cp(C_TITLE, bold=True))
    except curses.error:
        pass


def safe_addstr(win: "curses._CursesWindow", y: int, x: int, text: str,
                attr: int = 0, max_width: int = 0) -> None:
    """addstr sem explodir no canto inferior direito."""
    if max_width:
        text = text[:max_width]
    try:
        win.addstr(y, x, text, attr)
    except curses.error:
        pass


def clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def draw_image(win: "curses._CursesWindow", y: int, x: int, image_path: Path,
               width: Optional[str] = None, height: Optional[str] = None) -> None:
    """
    Renders an image in WezTerm/iTerm2 within a curses window.
    Uses os.write to bypass all Python and curses buffering.
    """
    try:
        if not image_path.exists():
            win.addstr(y, x, f"[Image Not Found: {image_path.name}]")
            return

        with open(image_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")

        # 1. Position curses cursor
        win.move(y, x)
        win.refresh()

        # 2. Construct basic iTerm2 sequence
        # Using the most compatible basic version: \x1b]1337;File=inline=1:DATA\x07
        args = "inline=1"
        if width:
            args += f";width={width}"
        if height:
            args += f";height={height}"

        seq = f"\x1b]1337;File={args}:{img_data}\x07"
        seq_bytes = seq.encode("utf-8")

        # 3. OS-level write to bypass ALL buffering
        # This is the most reliable way to send escape sequences while curses is active
        os.write(sys.stdout.fileno(), seq_bytes)

    except Exception as e:
        try:
            win.addstr(y, x, f"[Image Error: {e}]")
        except curses.error:
            pass


def load_config_safe() -> dict[str, Any]:
    path = PROJECT_ROOT / "config" / "base_config.json"
    config: dict[str, Any] = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            config = json.load(f)
    # Chaves opcionais ausentes no arquivo aparecem com o valor padrão.
    for key, default in OPTIONAL_CONFIG_DEFAULTS.items():
        config.setdefault(key, default)
    return config


def save_config_safe(cfg: dict[str, Any]) -> bool:
    path = PROJECT_ROOT / "config" / "base_config.json"
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        return True
    except OSError:
        return False


def list_checkpoints() -> list[Path]:
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    if not ckpt_dir.exists():
        return []
    return sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)


def human_size(b: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024:
            return f"{b:.0f} {unit}"
        b /= 1024  # type: ignore[assignment]
    return f"{b:.1f} TB"


# ──────────────────────────── LOGO ASCII ─────────────────────────────────────
LOGO_TEXT = r"""
     █████╗ ██████╗ ██████╗  ██████╗ ██████╗  █████╗        ██████╗ ██████╗ ████████╗
    ██╔══██╗██╔══██╗██╔══██╗██╔═══██╗██╔══██╗██╔══██╗      ██╔════╝ ██╔══██╗╚══██╔══╝
███████║██████╔╝██████╔╝██║   ██║██████╔╝███████║█████╗██║  ███╗██████╔╝   ██║
██╔══██║██╔══██╗██╔══██╗██║   ██║██╔══██╗██╔══██║╚════╝██║   ██║██╔═══╝    ██║
██║  ██║██████╔╝██║  ██║╚██████╔╝██████╔╝██║  ██║      ╚██████╔╝██║        ██║
╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝       ╚═════╝ ╚═╝        ╚═╝
"""

LOGO_BRAILLE = r"""
⠀⠀⠀⠀⠀⠀⣠⣴⡾⠿⠿⠿⣶⣤⣀⣀⣀⣀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⢀⣼⠟⠁⠀⠀⢀⣠⣼⡿⠟⠛⠛⠛⠿⣶⣄⠀⠀⠀
⠀⠀⢀⣠⣼⡟⠀⠀⣤⣶⠿⠋⠁⠀⢀⣤⣀⠀⠀⠈⢿⣧⠀⠀
⠀⣴⡿⠋⣿⡇⠀⠀⣿⠁⠀⣀⣴⡾⠟⠛⠻⢷⣦⣄⢘⣿⠀⠀
⣼⡟⠀⠀⣿⡇⠀⠀⣿⣶⠿⠛⠛⠿⣶⣄⡀⠀⠈⠛⠿⣿⡄⠀
⣿⡇⠀⠀⣿⣇⠀⠀⣿⠁⠀⠀⠀⠀⠈⣿⠿⣷⣤⡀⠀⠘⣿⡆
⠸⣿⡄⠀⠈⠛⢿⣶⣿⡀⠀⠀⠀⠀⢀⣿⠀⠀⢹⣿⠀⠀⢸⣿
⠀⠘⣿⣶⣤⡀⠀⠈⠙⠿⣶⣤⣤⣶⠿⣿⠀⠀⢸⣿⠀⠀⣼⡟
⠀⠀⣿⡅⠙⠻⢷⣦⣤⣴⡾⠟⠉⠀⢀⣿⠀⠀⢸⣿⣠⣾⠟⠀
⠀⠀⢻⣷⡀⠀⠀⠉⠛⠁⠀⢀⣠⣶⡿⠛⠀⠀⣸⡟⠋⠁⠀⠀
⠀⠀⠀⠙⠿⣶⣤⣤⣤⣤⣾⡟⠋⠁⠀⠀⢀⣴⡿⠁⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠉⠉⠉⠉⠛⠿⣶⣶⣶⡾⠟⠋⠀⠀⠀⠀⠀⠀
"""

LOGO_LINES = [l for l in LOGO_BRAILLE.split("\n") if l.strip()]


# ─────────────────────── widget de menu navegável ────────────────────────────
class Menu:
    def __init__(self, items: list[tuple[str, str, Callable]]) -> None:
        # items: [(tecla, label, callback)]
        self.items = items
        self.cursor = 0

    def move_up(self) -> None:
        self.cursor = (self.cursor - 1) % len(self.items)

    def move_down(self) -> None:
        self.cursor = (self.cursor + 1) % len(self.items)

    def select(self) -> Any:
        return self.items[self.cursor][2]()

    def draw(self, win: "curses._CursesWindow", y: int, x: int,
             width: int = 40) -> None:
        for i, (key, label, _) in enumerate(self.items):
            text = f"  [{key}]  {label}"
            text = text.ljust(width)[:width]
            if i == self.cursor:
                safe_addstr(win, y+i, x, text, cp(C_SELECTED, bold=True))
            else:
                safe_addstr(win, y+i, x, text, cp(C_DIM))


# ─────────────────────── widget de log em tempo real ─────────────────────────
class LogPanel:
    def __init__(self, max_lines: int = 200) -> None:
        self._lines: list[tuple[str, int]] = []   # (texto, attr)
        self._lock = threading.Lock()
        self.max_lines = max_lines
        self.scroll = 0

    def add(self, text: str, attr: int = 0) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        with self._lock:
            for line in (f"[{ts}] {text}").splitlines():
                self._lines.append((line, attr))
                if len(self._lines) > self.max_lines:
                    self._lines.pop(0)
            self.scroll = 0   # auto-scroll para o fim

    def draw(self, win: "curses._CursesWindow", y: int, x: int,
             h: int, w: int) -> None:
        with self._lock:
            visible = self._lines[-(h - self.scroll):][:h] if h > 0 else []
            for i, (text, attr) in enumerate(visible[-h:]):
                safe_addstr(win, y+i, x, text[:w-1].ljust(w-1), attr or cp(C_DIM))


# ─────────────────────── barra de progresso ──────────────────────────────────
class ProgressBar:
    def __init__(self) -> None:
        self.current = 0
        self.total = 0
        self.label = ""
        self.extra = ""
        self._lock = threading.Lock()

    def update(self, current: int, total: int, label: str = "", extra: str = "") -> None:
        with self._lock:
            self.current = current
            self.total = total
            self.label = label
            self.extra = extra

    def draw(self, win: "curses._CursesWindow", y: int, x: int, w: int, status_msg: str = "", status_attr: int = 0) -> None:
        with self._lock:
            cur, tot = self.current, self.total
            lbl, ext = self.label, self.extra
        pct = cur / tot if tot else 0

        # Line 1: Status
        safe_addstr(win, y, x, f"Status: {status_msg}"[:w], status_attr or cp(C_DIM))

        # Line 2: Space

        # Line 3: Bar
        bar_w = w - 10 # Approximate room for pct and cur/tot
        filled = int(bar_w * pct)
        bar = "█" * filled + "░" * (bar_w - filled)
        pct_str = f"{pct*100:5.1f}%"
        safe_addstr(win, y+2, x, f"│{bar}│ {pct_str} {cur:,}/{tot:,}", cp(C_BORDER))

        # Line 4: Extra Metrics
        safe_addstr(win, y+3, x, f"{ext}"[:w], cp(C_DIM))



# ─────────────────────── estado global da TUI ────────────────────────────────
class AppState:
    def __init__(self) -> None:
        self.log = LogPanel()
        self.progress = ProgressBar()
        self.running_process: Optional[subprocess.Popen] = None  # type: ignore[type-arg]
        self.process_thread: Optional[threading.Thread] = None
        self.config: dict[str, Any] = load_config_safe()
        self.status_msg = "Pronto"
        self.status_attr = 0   # será atualizado após init_colors
        self._lock = threading.Lock()
        self._stop_requested = False

    def set_status(self, msg: str, kind: str = "ok") -> None:
        attr_map = {"ok": cp(C_SUCCESS), "warn": cp(C_WARNING), "err": cp(C_ERROR)}
        with self._lock:
            self.status_msg = msg
            self.status_attr = attr_map.get(kind, cp(C_DIM))

    def get_status(self) -> tuple[str, int]:
        with self._lock:
            return self.status_msg, self.status_attr

    def is_busy(self) -> bool:
        with self._lock:
            return self.running_process is not None or self.process_thread is not None

    def launch(self, cmd: list[str], cwd: Path, on_line: Optional[Callable[[str], None]] = None) -> bool:
        """Inicia um processo filho sem bloquear a TUI e captura a saída em tempo real.

        O train.py usa tqdm, que atualiza a mesma linha com ``\\r``. Por isso a
        leitura trata tanto ``\\n`` quanto ``\\r`` como atualizações de saída.
        """
        if self.is_busy():
            return False

        def _emit_line(line: str) -> None:
            line = line.strip("\\r\\n")
            if not line:
                return
            if not line.startswith(("@@ABROBA_TOKEN@@", "@@ABROBA_NERD@@", "@@ABROBA_DONE@@")):
                self.log.add(line)
            if on_line:
                on_line(line)

        def _run() -> None:
            proc: Optional[subprocess.Popen[bytes]] = None
            try:
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                env["PYTHONIOENCODING"] = "utf-8"
                env["PYTHONUTF8"] = "1"

                popen_kwargs: dict[str, Any] = {}
                if os.name == "nt":
                    popen_kwargs["creationflags"] = getattr(
                        subprocess, "CREATE_NEW_PROCESS_GROUP", 0
                    )
                else:
                    popen_kwargs["start_new_session"] = True

                # -u garante que o Python filho não retenha os logs no pipe.
                launch_cmd = (
                    [sys.executable, "-u", *cmd[1:]]
                    if cmd and cmd[0] == sys.executable
                    else cmd
                )
                proc = subprocess.Popen(
                    launch_cmd,
                    cwd=str(cwd),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=0,
                    env=env,
                    **popen_kwargs,
                )
                with self._lock:
                    self.running_process = proc

                import codecs
                decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
                pending = ""
                stream = proc.stdout
                if stream is not None:
                    while True:
                        chunk = stream.read(4096)
                        if not chunk:
                            break
                        pending += decoder.decode(chunk)

                        # tqdm usa CR para redesenhar a linha; logs normais
                        # usam LF. Ambos chegam ao parser imediatamente.
                        parts = re.split(r"[\r\n]", pending)
                        pending = parts.pop()
                        for part in parts:
                            _emit_line(part)

                pending += decoder.decode(b"", final=True)
                if pending:
                    _emit_line(pending)

                proc.wait()
                code = proc.returncode
                with self._lock:
                    stopped = self._stop_requested
                    self.running_process = None
                    self._stop_requested = False

                if stopped:
                    self.set_status("Processo interrompido.", "warn")
                elif code == 0:
                    self.set_status("Concluído com sucesso ✓", "ok")
                else:
                    self.log.add(
                        f"Processo encerrado com código {code}.",
                        cp(C_ERROR, bold=True),
                    )
                    self.set_status(f"Processo encerrado (código {code})", "err")
            except Exception as exc:
                with self._lock:
                    self.running_process = None
                    self._stop_requested = False
                self.log.add(
                    f"Erro ao executar processo: {exc}",
                    cp(C_ERROR, bold=True),
                )
                self.set_status(f"Erro: {exc}", "err")
            finally:
                with self._lock:
                    self.process_thread = None

        t = threading.Thread(target=_run, daemon=True)
        with self._lock:
            self.process_thread = t
            self._stop_requested = False
        t.start()
        return True

    def kill(self) -> None:
        with self._lock:
            proc = self.running_process
            self._stop_requested = proc is not None

        if proc is None:
            return

        self.log.add("Solicitando interrupção do processo…", cp(C_WARNING, bold=True))
        try:
            if os.name == "nt":
                try:
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                except (AttributeError, OSError, ValueError):
                    proc.terminate()
            else:
                os.killpg(proc.pid, signal.SIGINT)
        except Exception as exc:
            self.log.add(f"Falha ao interromper: {exc}", cp(C_ERROR, bold=True))
            try:
                proc.terminate()
            except Exception:
                pass


TRAIN_STATUS_MAP = {
    "Streaming de": ("[ DATASET ] Baixando dados do HF...", "ok"),
    "Download concluído": ("[ DATASET ] Download concluído!", "ok"),
    "Corpus preparado": ("[ DADOS ] Corpus preparado com sucesso!", "ok"),
    "Corpus já estava atualizado": ("[ DADOS ] Corpus já estava atualizado.", "ok"),
    "dispositivo:": ("[ GPU ] Hardware detectado e configurado.", "ok"),
    "Tokenizer BPE:": ("[ TOKENIZER ] Vocabulário carregado.", "ok"),
    "Retomado de": ("[ CHECKPOINT ] Treino retomado de checkpoint.", "ok"),
    "Iniciando treinamento": ("[ TREINAMENTO ] Iniciando loop de treino...", "ok"),
    "Validation loss melhorou": ("[ AVALIAÇÃO ] Nova melhoria de loss! Modelo salvo.", "ok"),
    "Validation loss não melhorou": ("[ AVALIAÇÃO ] Sem melhora na validação.", "warn"),
    "early stopping ainda em warmup": ("[ EARLY STOPPING ] Warmup ativo, ignorando loss.", "warn"),
    "contador": ("[ EARLY STOPPING ] Paciência: contador incrementado.", "warn"),
    "Early stopping acionado": ("[ EARLY STOPPING ] Treino interrompido prematuramente.", "err"),
    "Checkpoint periódico salvo": ("[ CHECKPOINT ] Backup periódico realizado.", "ok"),
    "Checkpoint final salvo": ("[ CHECKPOINT ] Modelo final salvo.", "ok"),
    "Treinamento concluído": ("[ CONCLUÍDO ] Treinamento finalizado com sucesso!", "ok"),
    "Erro:": ("[ ERRO ] Ocorreu uma falha crítica.", "err"),
}

def parse_train_line(line: str, state: AppState) -> None:
    """Extrai total/iteração do output real do train.py e atualiza a barra."""
    # 1. Update Status from Mapping
    for key, (msg, kind) in TRAIN_STATUS_MAP.items():
        if key in line:
            state.set_status(msg, kind)
            break

    # 2. Total Iterations
    total_match = re.search(
        r"Total escolhido:\s*([\d,]+)\s*iterações", line, re.IGNORECASE
    )
    if total_match:
        total = int(total_match.group(1).replace(",", ""))
        state.progress.update(
            min(state.progress.current, total), total, "Iniciando…"
        )
        return

    # 3. Resuming
    resume_match = re.search(r"na iteração\s*([\d,]+)", line, re.IGNORECASE)
    if resume_match:
        current = int(resume_match.group(1).replace(",", ""))
        total = state.progress.total
        if total:
            state.progress.update(current, total, "Retomando treinamento…")
        return

    # 4. Training Metrics (Iter X/Y | loss ... | val ... | lr ...)
    m = re.search(
        r"Iter\s+([\d,]+)/([\d,]+)\s*\|\s*loss\s+([\d.]+)\s*\|\s*val\s+([\d.]+).*?lr\s+([\d.eE+-]+)",
        line,
    )
    if m:
        cur = int(m.group(1).replace(",", ""))
        tot = int(m.group(2).replace(",", ""))
        loss, val, lr = m.group(3), m.group(4), m.group(5)
        state.progress.update(
            cur, tot, "Treinando AbrobaGPT…",
            f"Loss: {loss} | Val: {val} | LR: {lr}",
        )
        return

    # 5. tqdm Fallback: | 124/1000 [ ... ]
    m2 = re.search(r"\|\s*([\d,]+)\s*/\s*([\d,]+)\s*\[", line)
    if m2:
        cur = int(m2.group(1).replace(",", ""))
        tot = int(m2.group(2).replace(",", ""))
        state.progress.update(cur, tot, "Treinando…")


# ──────────────────────── telas da TUI ───────────────────────────────────────
class Screen:
    """Classe base para todas as telas."""
    def __init__(self, app: "TUIApp") -> None:
        self.app = app

    def draw(self, win: "curses._CursesWindow") -> None:
        pass

    def handle_key(self, key: int) -> Optional[str]:
        """Retorna nome da próxima tela ou None para ficar."""
        return None


class GraphScreen(Screen):
    """Visualiza o histórico mais recente sem interferir no processo de treino."""
    def __init__(self, app: "TUIApp") -> None:
        super().__init__(app)
        self.kind = 0
        self.kinds = ["Loss", "Learning rate", "Perplexidade"]

    def draw(self, win: "curses._CursesWindow") -> None:
        h, w = win.getmaxyx(); win.erase()
        directory, rows = load_metrics(PROJECT_ROOT / "training_logs")
        safe_addstr(win, 0, 2, "GRÁFICO · histórico de treinamento", cp(C_TITLE, bold=True))
        if directory:
            safe_addstr(win, 1, 2, f"Execução: {directory.name} | avaliações: {len(rows)}", cp(C_DIM))
        else:
            safe_addstr(win, 2, 2, "Nenhum log encontrado. Inicie um treino para criar o histórico.", cp(C_WARNING))
        safe_addstr(win, 3, 2, "←/→ ou 1-3 trocar gráfico | atualização automática | ESC/Q voltar", cp(C_DIM))
        numeric=[]
        for row in rows:
            try:
                it=float(row.get('iteration') or 0)
                if self.kind == 0:
                    numeric.append((it, row.get('train_loss'), row.get('validation_loss')))
                elif self.kind == 1: numeric.append((it, row.get('learning_rate'), None))
                else: numeric.append((it, row.get('perplexity'), None))
            except (TypeError, ValueError): continue
        if self.kind == 0:
            series=[("train loss", [(x,float(a)) for x,a,b in numeric if _valid(a)], "⠿"), ("validation loss", [(x,float(b)) for x,a,b in numeric if _valid(b)], "⣿")]
        elif self.kind == 1: series=[("learning rate",[(x,float(a)) for x,a,b in numeric if _valid(a)],"⣿")]
        else: series=[("perplexidade",[(x,float(a)) for x,a,b in numeric if _valid(a)],"⣿")]
        plot=braille_plot(series, max(12,w-4), max(5,h-8), self.kinds[self.kind])
        for i,line in enumerate(plot[:max(0,h-5)]): safe_addstr(win, 5+i, 2, line[:max(0,w-4)], cp(C_SUCCESS if i>0 else C_TITLE))
        if rows and self.kind==0:
            best=[]
            for r in rows:
                if _valid(r.get('validation_loss')): best.append((float(r['validation_loss']), int(float(r.get('iteration') or 0))))
            if best:
                value,it=min(best)
                safe_addstr(win,h-3,2,f"Melhor validation loss: {value:.6g} na iteração {it}",cp(C_SUCCESS,bold=True))
        stopped = next((r for r in reversed(rows) if str(r.get('early_stopping')).lower() in ('true','1')), None)
        if stopped:
            safe_addstr(win,h-2,2,f"EARLY STOPPING: iteração {stopped.get('iteration','?')}",cp(C_ERROR,bold=True))

    def handle_key(self, key: int) -> Optional[str]:
        if key in (27, ord('q'), ord('Q')): return 'home'
        if key in (curses.KEY_RIGHT, ord('l')): self.kind=(self.kind+1)%3
        elif key in (curses.KEY_LEFT, ord('h')): self.kind=(self.kind-1)%3
        elif key in (ord('1'),ord('2'),ord('3')): self.kind=int(chr(key))-1
        return None

def _valid(value: Any) -> bool:
    try: return math.isfinite(float(value))
    except (TypeError, ValueError): return False


# ────────────────────────────── tela principal ───────────────────────────────
class HomeScreen(Screen):
    def __init__(self, app: "TUIApp") -> None:
        super().__init__(app)
        self.menu = Menu([
            ("1", "Treinar Modelo",             lambda: "train"),
            ("2", "Gerar Texto",                lambda: "generate"),
            ("3", "Chat Interativo",            lambda: "chat"),
            ("4", "Preparar Dataset",           lambda: "prepare"),
            ("5", "Gerenciar Checkpoints",      lambda: "checkpoints"),
            ("6", "Configurações",              lambda: "config"),
            ("7", "Executar Testes",            lambda: "tests"),
            ("8", "Monitor de Processo",        lambda: "monitor"),
            ("9", "Gráfico",                    lambda: "graphs"),
            ("0", "Testar Checkpoints",         lambda: "evaluation"),
            ("Q", "Sair",                       lambda: "quit"),
        ])

    def draw(self, win: "curses._CursesWindow") -> None:
        h, w = win.getmaxyx()
        win.clear()

        # Proporcionalidade baseada em uma largura ideal de 120 e altura de 40
        # Isso evita que os elementos fiquem "grudados" em telas pequenas ou muito longe em telas gigantes
        scale_w = w / 120
        scale_h = h / 40

        # Stylized "AbrobaGPT" Title
        title_text = "  A B R O B A G P T  "
        title_x = max(0, (w - len(title_text)) // 2)
        title_y = int(2 * scale_h)
        try:
            win.addstr(title_y, title_x, title_text, cp(C_TITLE, bold=True))
        except curses.error:
            pass

        # Logo ASCII Original (Texto)
        for i, line in enumerate(LOGO_TEXT.split("\n")):
            if not line.strip(): continue
            try:
                line_x = max(0, (w - len(line)) // 2)
                line_y = title_y + 1 + i
                win.addstr(line_y, line_x, line, cp(C_TITLE))
            except curses.error:
                pass

        # Logo Braille (Embaixo do texto)
        for i, line in enumerate(LOGO_LINES):
            try:
                line_x = max(0, (w - len(line)) // 2)
                line_y = title_y + 9 + i
                win.addstr(line_y, line_x, line, curses.color_pair(100))
            except curses.error:
                pass

        # Offset dinâmico para a altura total (Texto + Braille)
        img_h_offset = int(20 * scale_h) + 9



        # Subtítulo
        sub = "─── AbrobaGPT · LLM do Zero em PyTorch ───"
        sub_y = int(-2 * scale_h) + img_h_offset
        safe_addstr(win, sub_y, (w-len(sub))//2, sub, cp(C_BORDER, bold=True))

        # Menu
        menu_y = int(3 * scale_h) + img_h_offset
        menu_w = int(42 * scale_w)
        menu_w = max(30, min(menu_w, 60)) # Limites para não quebrar a estética
        menu_x = (w - menu_w) // 2
        draw_box(win, menu_y-1, menu_x-2, len(self.menu.items)+3, menu_w+4, "MENU PRINCIPAL")
        self.menu.draw(win, menu_y, menu_x, menu_w)

        # Atalhos
        hint_y = menu_y + len(self.menu.items) + 2
        hint = "↑↓ navegar   Enter/número selecionar   Q sair"
        safe_addstr(win, hint_y, (w-len(hint))//2, hint, cp(C_DIM))

        # Status
        msg, attr = self.app.state.get_status()
        safe_addstr(win, h-2, 2, f"● {msg}"[:w-4], attr | curses.A_BOLD)

        # Relógio
        clock = datetime.now().strftime("%H:%M:%S")
        safe_addstr(win, h-2, w-12, f"[{clock}]", cp(C_DIM))

        # Busy indicator
        if self.app.state.is_busy():
            frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
            spin = frames[int(time.time()*8) % len(frames)]
            safe_addstr(win, h-2, w-20, f"{spin} processo ativo", cp(C_WARNING, bold=True))

    def handle_key(self, key: int) -> Optional[str]:
        if key in (curses.KEY_UP,    ord("k")): self.menu.move_up()
        elif key in (curses.KEY_DOWN, ord("j")): self.menu.move_down()
        elif key in (curses.KEY_ENTER, 10, 13):
            return self.menu.select()
        else:
            for k, (shortcut, _, cb) in enumerate(self.menu.items):
                if key == ord(shortcut.lower()) or key == ord(shortcut.upper()):
                    return cb()
        return None


# ────────────────────────────── tela de treino ───────────────────────────────
class TrainScreen(Screen):
    def __init__(self, app: "TUIApp") -> None:
        super().__init__(app)
        self.source_idx  = 0   # 0=local, 1=hf
        self.dataset_idx = 0   # índice em DATASET_TYPES
        self.iters_str   = ""
        self.resume      = False
        self.early_stop    = bool(app.state.config.get("early_stopping", False))
        self.nerd_mode = False
        self.patience_str  = str(app.state.config.get("patience", ""))
        self._early_stop_overridden = False
        self._patience_overridden = False
        self.size_mb_str   = "10"
        self.field         = 0   # campo ativo
        self.lr_idx        = 0   # índice em LR_KEYS
        self._fields      = ["source", "dataset", "iters", "lr", "resume", "early_stop", "patience", "nerds", "advanced", "run"]
        self.cursor_pos    = len(self.iters_str)
        self.log_scroll    = 0
        self.advanced_open = False
        self.advanced_cursor = 0
        self.advanced_editing = False
        self.advanced_buffer = ""
        self.advanced_message = ""
        self._advanced_fields = [("cosine_decay", "Cosine decay", "bool"), ("warmup_iters", "Warmup (iterações)", "int"), ("min_learning_rate", "Learning rate mínimo", "float"), ("weight_decay", "Weight decay", "float"), ("grad_clip", "Gradient clipping (0=off)", "float"), ("dropout", "Dropout", "float"), ("batch_size", "Batch size", "int"), ("eval_interval", "Intervalo de avaliação", "int"), ("eval_iters", "Batches por avaliação", "int"), ("amp", "Mixed precision (AMP)", "bool"), ("validation_fraction", "Fração de validação", "float"), ("seed", "Seed aleatória", "int")]

    def _field_count(self) -> int:
        return len(self._fields)

    def draw(self, win: "curses._CursesWindow") -> None:
        h, w = win.getmaxyx()
        win.clear()
        draw_box(win, 0, 0, h-1, w, "TREINAMENTO · AbrobaGPT")

        col1 = 4
        # Título
        safe_addstr(win, 2, col1, "Configure e inicie o treinamento:", cp(C_TITLE, bold=True))

        if self.advanced_open:
            safe_addstr(win, 3, col1, "▼ AVANÇADO · salvo em config/base_config.json", cp(C_TITLE, bold=True))
            safe_addstr(win, 4, col1, "↑↓ navegar · Enter editar/alternar · ESC voltar", cp(C_DIM))
            visible = max(1, min(len(self._advanced_fields), h - 9))
            start = max(0, min(self.advanced_cursor - visible + 1, len(self._advanced_fields) - visible))
            for j, i in enumerate(range(start, start + visible)):
                k, label, kind = self._advanced_fields[i]; val = self.app.state.config.get(k, "")
                selected = i == self.advanced_cursor
                safe_addstr(win, 6+j, col1, ("› " if selected else "  ") + label, cp(C_SELECTED if selected else C_DIM, bold=selected))
                shown = self.advanced_buffer if selected and self.advanced_editing else (("ON" if val else "OFF") if kind == "bool" else str(val))
                safe_addstr(win, 6+j, min(42, w//2), str(shown)[:max(1,w//2-2)], cp(C_INPUT if selected and self.advanced_editing else C_WARNING))
            if self.advanced_message: safe_addstr(win, h-3, col1, self.advanced_message[:w-col1-2], cp(C_SUCCESS))
            safe_addstr(win, h-2, col1, "ESC: fechar Avançado", cp(C_DIM))
            return

        sel = self._fields[self.field]

        # fonte
        src_lbl = ["Arquivo local", "Streaming HuggingFace"][self.source_idx]
        self._draw_field(win, 4, col1, "Fonte de dados", f"[ {src_lbl} ]",
                         sel == "source", w)

        # dataset
        ds_lbl = DATASET_LABELS[self.dataset_idx]
        self._draw_field(win, 5, col1, "Dataset        ", f"[ {ds_lbl} ]",
                         sel == "dataset", w)

        # iterações
        iterations_value = self.iters_str or f"(config: {self.app.state.config.get('max_iters', '')})"
        self._draw_field(win, 6, col1, "Mil iterações  ", iterations_value,
                         sel == "iters", w)

        # learning rate
        lr_key = LR_KEYS[self.lr_idx]
        if lr_key is None:
            cfg = self.app.state.config
            lr_extra = f"pico {float(cfg.get('learning_rate', 0)):.0e} · warmup {cfg.get('warmup_iters', '')}".replace("e-0", "e-")
        else:
            lr_extra = describe_preset(lr_key)
        self._draw_field(win, 7, col1, "Learning rate  ", f"[ {LR_LABELS[self.lr_idx]} ]  {lr_extra}",
                         sel == "lr", w)

        # retomar
        resume_lbl = "[X] Retomar checkpoint" if self.resume else "[ ] Retomar checkpoint"
        self._draw_field(win, 8, col1, "               ", resume_lbl,
                         sel == "resume", w)

        # early stopping
        es_lbl = "[X] Early Stopping" if self.early_stop else "[ ] Early Stopping"
        self._draw_field(win, 9, col1, "               ", es_lbl,
                         sel == "early_stop", w)

        # paciência
        patience_val = f"[ {self.patience_str} ]" if self.early_stop else "(desativado)"
        self._draw_field(win, 10, col1, "Paciência       ", patience_val,
                         sel == "patience", w)
        if DATASET_TYPES[self.dataset_idx] == "ragqa":
            safe_addstr(win, 11, col1, "HF: Madras1/rag-qa-fulltext-ptbr"[:w-col1-2], cp(C_DIM))

        nerd_label = "[X] Para nerds: métricas detalhadas" if self.nerd_mode else "[ ] Para nerds: métricas detalhadas"
        self._draw_field(win, 12, col1, "               ", nerd_label, sel == "nerds", w)
        # Menu retrátil de parâmetros avançados
        self._draw_field(win, 13, col1, "               ", "▶ Avançado (abrir parâmetros)", sel == "advanced", w)

        # botão TREINAR
        btn_attr = cp(C_SELECTED, bold=True) if sel == "run" else cp(C_SUCCESS, bold=True)
        busy = self.app.state.is_busy()
        btn_text = "  [PROCESSO EM ANDAMENTO...]  " if busy else "  [INICIAR TREINAMENTO]  "
        safe_addstr(win, 14, col1, btn_text, btn_attr)

        if busy:
            draw_box(win, 15, 2, 5, w-4, "PROGRESSO")
            self.app.state.progress.draw(win, 16, 4, w-8)
        else:
            safe_addstr(win, 15, col1,
                        "ESC voltar  ↑↓ campos  ←→/Espaço mudar valor  Enter confirmar",
                        cp(C_DIM))

        # Log
        log_y = 20
        log_h = h - log_y - 2
        if log_h > 2:
            draw_box(win, log_y-1, 1, log_h+2, w-2, "LOG DE SAÍDA")
            self.app.state.log.draw(win, log_y, 2, log_h, w-4)

        # Botão matar
        if busy:
            safe_addstr(win, h-2, col1, "  [K] INTERROMPER PROCESSO  ", cp(C_ERROR, bold=True))

    def _draw_field(self, win: "curses._CursesWindow", y: int, x: int,
                    label: str, value: str, selected: bool, w: int) -> None:
        lbl_attr  = cp(C_PROMPT, bold=True) if selected else cp(C_DIM)
        val_attr  = cp(C_SELECTED, bold=True) if selected else cp(C_WARNING)
        safe_addstr(win, y, x,   f"{label}: ", lbl_attr)
        safe_addstr(win, y, x+17, value[:w-x-20], val_attr)

    def handle_key(self, key: int) -> Optional[str]:
        if self.advanced_open:
            if key == 27:
                if self.advanced_editing: self.advanced_editing = False; self.advanced_buffer = ""
                else: self.advanced_open = False
                return None
            if self.advanced_editing:
                k, _, kind = self._advanced_fields[self.advanced_cursor]
                if key in (curses.KEY_ENTER, 10, 13):
                    try:
                        raw=self.advanced_buffer.strip(); value=int(raw) if kind == "int" else float(raw)
                        if not math.isfinite(value) or (kind == "int" and value < 0): raise ValueError
                        if k in ("dropout", "validation_fraction") and not 0 <= value < 1: raise ValueError
                        if k in ("min_learning_rate", "weight_decay", "grad_clip", "warmup_iters") and value < 0: raise ValueError
                        if k in ("batch_size", "eval_interval", "eval_iters") and value < 1: raise ValueError
                        self.app.state.config[k]=value
                        self.advanced_message = f"{k} salvo" if save_config_safe(self.app.state.config) else "Erro ao salvar"
                    except ValueError: self.advanced_message="Valor inválido; não foi salvo"
                    self.advanced_editing=False; self.advanced_buffer=""
                elif key in (curses.KEY_BACKSPACE,127,8): self.advanced_buffer=self.advanced_buffer[:-1]
                elif 0 <= key < 256 and chr(key) in "0123456789.-": self.advanced_buffer+=chr(key)
                return None
            if key == curses.KEY_UP: self.advanced_cursor=(self.advanced_cursor-1)%len(self._advanced_fields)
            elif key == curses.KEY_DOWN: self.advanced_cursor=(self.advanced_cursor+1)%len(self._advanced_fields)
            elif key in (curses.KEY_LEFT,curses.KEY_RIGHT,ord(" "),curses.KEY_ENTER,10,13):
                k,_,kind=self._advanced_fields[self.advanced_cursor]
                if kind == "bool":
                    self.app.state.config[k]=not bool(self.app.state.config.get(k,False))
                    self.advanced_message=f"{k}: salvo" if save_config_safe(self.app.state.config) else "Erro ao salvar"
                elif key in (curses.KEY_ENTER,10,13): self.advanced_editing=True; self.advanced_buffer=str(self.app.state.config.get(k,""))
            return None
        if key == 27:   # ESC
            return "home"
        busy = self.app.state.is_busy()
        if key == ord("k") and busy:
            self.app.state.kill()
            return None
        if busy:
            return None
        name = self._fields[self.field]
        if key == curses.KEY_UP:
            self.field = (self.field - 1) % len(self._fields)
        elif key == curses.KEY_DOWN:
            self.field = (self.field + 1) % len(self._fields)
        elif key in (curses.KEY_LEFT, curses.KEY_RIGHT, ord(" ")):
            if name == "source":
                self.source_idx = 1 - self.source_idx
            elif name == "dataset":
                self.dataset_idx = cycle_index(self.dataset_idx, key, len(DATASET_TYPES))
            elif name == "lr":
                self.lr_idx = cycle_index(self.lr_idx, key, len(LR_KEYS))
            elif name == "resume":
                self.resume = not self.resume
            elif name == "early_stop":
                self.early_stop = not self.early_stop
                self._early_stop_overridden = True
            elif name == "nerds":
                self.nerd_mode = not self.nerd_mode
        elif key in (curses.KEY_ENTER, 10, 13):
            if name == "run": self._start_training()
            elif name == "advanced": self.advanced_open = True; self.advanced_cursor = 0; self.advanced_message = ""
            elif name == "nerds": self.nerd_mode = not self.nerd_mode
            else: self.field = (self.field + 1) % len(self._fields)
        elif name == "iters":
            # Edição de texto
            if key in (curses.KEY_BACKSPACE, 127, 8):
                self.iters_str = self.iters_str[:-1]
            elif 0 <= key < 256 and chr(key).isdigit():
                self.iters_str += chr(key)
        elif name == "patience" and self.early_stop:
            # Edição de paciência
            if key in (curses.KEY_BACKSPACE, 127, 8):
                self.patience_str = self.patience_str[:-1]
                self._patience_overridden = True
            elif 0 <= key < 256 and chr(key).isdigit():
                self.patience_str += chr(key)
                self._patience_overridden = True
        return None

    def _start_training(self) -> None:
        if self.app.state.is_busy():
            return
        cmd = [sys.executable, "train.py"]
        source = ["local", "hf"][self.source_idx]
        cmd += ["--source", source]
        cmd += ["--dataset-type", DATASET_TYPES[self.dataset_idx]]
        if LR_KEYS[self.lr_idx]:
            cmd += ["--lr-preset", LR_KEYS[self.lr_idx]]
        if self.iters_str:
            try:
                cmd += ["--max-iters", str(int(self.iters_str) * 1000)]
            except ValueError:
                pass
        if self.resume:
            cmd += ["--resume"]
        if self.nerd_mode:
            cmd += ["--nerd-mode"]
        if self._early_stop_overridden:
            cmd += ["--early-stopping" if self.early_stop else "--no-early-stopping"]
        if self.early_stop and self._patience_overridden and self.patience_str:
            try:
                cmd += ["--patience", self.patience_str]
            except ValueError:
                pass
        self.app.state.log.add(f"$ {' '.join(cmd)}", cp(C_PROMPT, bold=True))
        self.app.state.set_status("Treinamento iniciado…", "warn")
        total_iters = int(self.app.state.config.get("max_iters", 0))
        if self.iters_str:
            try:
                total_iters = int(self.iters_str) * 1000
            except ValueError:
                pass
        self.app.state.progress.update(0, total_iters, "Iniciando…")
        self.app.state.launch(
            cmd, PROJECT_ROOT,
            on_line=lambda l: parse_train_line(l, self.app.state)
        )


# ──────────────────────────── tela de geração ────────────────────────────────
class GenerateScreen(Screen):
    def __init__(self, app: "TUIApp") -> None:
        super().__init__(app)
        self.prompt      = "O futuro da inteligência artificial"
        self.max_tokens  = "100"
        self.temperature = str(app.state.config.get("temperature", DEFAULT_TEMPERATURE))
        self.top_k       = ""
        self.ckpt_idx    = 0
        self.field       = 0
        self.output      = ""
        self.cursor_pos  = len(self.prompt)
        self.nerd_mode = False
        self.nerd_details = ""

    def _checkpoints(self) -> list[Path]:
        return list_checkpoints()

    def draw(self, win: "curses._CursesWindow") -> None:
        h, w = win.getmaxyx()
        win.clear()
        draw_box(win, 0, 0, h-1, w, "GERAÇÃO DE TEXTO")
        col = 4

        safe_addstr(win, 2, col, "Gere texto a partir de um prompt:", cp(C_TITLE, bold=True))

        ckpts = self._checkpoints()
        ckpt_name = ckpts[self.ckpt_idx].name if ckpts else "(nenhum)"
        self._draw_field(win, 4, col, "Checkpoint  ", f"[ {ckpt_name} ]", self.field == 0, w)
        self._draw_field(win, 5, col, "Max tokens  ", self.max_tokens,    self.field == 1, w)
        self._draw_field(win, 6, col, "Temperatura ", self.temperature,   self.field == 2, w)
        self._draw_field(win, 7, col, "Top-K       ", self.top_k or "(auto)", self.field == 3, w)
        self._draw_field(win, 8, col, "               ", "[X] Para nerds" if self.nerd_mode else "[ ] Para nerds", self.field == 5, w)

        # Prompt multilinha
        prompt_y = 10
        draw_box(win, prompt_y-1, 2, 4, w-4, "Prompt")
        attr = cp(C_SELECTED, bold=True) if self.field == 4 else cp(C_WARNING)
        prompt_display = self.prompt[-(w-8):]
        safe_addstr(win, prompt_y, col, prompt_display, attr)
        if self.field == 4:
            # cursor piscante
            cx = col + min(len(prompt_display), w-8)
            safe_addstr(win, prompt_y, cx, "▌", cp(C_PROMPT, bold=True))

        # Botão
        btn_attr = cp(C_SELECTED, bold=True) if self.field == 6 else cp(C_SUCCESS, bold=True)
        busy = self.app.state.is_busy()
        btn = "  [GERANDO...]  " if busy else "  [GERAR TEXTO]  "
        safe_addstr(win, prompt_y+3, col, btn, btn_attr)
        if self.nerd_mode and self.nerd_details:
            detail_lines = textwrap.wrap(self.nerd_details, width=max(10, w-col-2))
            for detail_idx, detail_line in enumerate(detail_lines[:2]):
                safe_addstr(win, prompt_y+4+detail_idx, col, detail_line, cp(C_DIM))

        # Output
        out_y = prompt_y + (7 if self.nerd_mode else 6)
        out_h = h - out_y - 3
        if out_h > 2:
            draw_box(win, out_y-1, 1, out_h+2, w-2, "SAÍDA DO MODELO")
            lines = textwrap.wrap(self.output or "(aguardando geração…)", w-6)
            for i, line in enumerate(lines[:out_h]):
                attr = cp(C_SUCCESS) if self.output else cp(C_DIM)
                safe_addstr(win, out_y+i, 3, line[:w-6], attr)

        safe_addstr(win, h-2, col,
                    "ESC voltar  ↑↓ campos  ←→ checkpoint  Enter confirmar",
                    cp(C_DIM))

    def _draw_field(self, win: "curses._CursesWindow", y: int, x: int,
                    label: str, value: str, selected: bool, w: int) -> None:
        lbl_attr = cp(C_PROMPT, bold=True) if selected else cp(C_DIM)
        val_attr = cp(C_SELECTED, bold=True) if selected else cp(C_WARNING)
        safe_addstr(win, y, x, f"{label}: ", lbl_attr)
        safe_addstr(win, y, x+14, value[:w-x-18], val_attr)

    def handle_key(self, key: int) -> Optional[str]:
        if key == 27:
            return "home"
        ckpts = self._checkpoints()
        busy = self.app.state.is_busy()
        if busy:
            return None
        if key == curses.KEY_UP:
            self.field = (self.field - 1) % 7
        elif key == curses.KEY_DOWN:
            self.field = (self.field + 1) % 7
        elif key in (curses.KEY_LEFT,):
            if self.field == 0 and ckpts:
                self.ckpt_idx = (self.ckpt_idx - 1) % len(ckpts)
        elif key in (curses.KEY_RIGHT,):
            if self.field == 0 and ckpts:
                self.ckpt_idx = (self.ckpt_idx + 1) % len(ckpts)
        elif key in (curses.KEY_ENTER, 10, 13):
            if self.field == 5:
                self.nerd_mode = not self.nerd_mode
            elif self.field == 6:
                self._start_generate(ckpts)
            else:
                self.field = min(self.field + 1, 5)
        elif self.field in (1, 2, 3):
            targets = {1: "max_tokens", 2: "temperature", 3: "top_k"}
            attr = targets[self.field]
            cur = getattr(self, attr)
            if key in (curses.KEY_BACKSPACE, 127, 8):
                setattr(self, attr, cur[:-1])
            elif chr(key) in "0123456789.":
                setattr(self, attr, cur + chr(key))
        elif self.field == 4:
            if key in (curses.KEY_BACKSPACE, 127, 8):
                self.prompt = self.prompt[:-1]
            elif 32 <= key < 127:
                self.prompt += chr(key)
        return None

    def _start_generate(self, ckpts: list[Path]) -> None:
        if not ckpts:
            self.app.state.set_status("Nenhum checkpoint disponível", "err")
            return
        cmd = [sys.executable, "generate.py",
               "--prompt", self.prompt or "Era uma vez",
               "--ckpt", str(ckpts[self.ckpt_idx]), "--stream"]
        if self.nerd_mode:
            cmd.append("--nerd-mode")
        try:
            cmd += ["--max-tokens", str(int(self.max_tokens))]
        except ValueError:
            pass
        try:
            cmd += ["--temp", str(float(self.temperature))]
        except ValueError:
            pass
        if self.top_k.strip():
            try:
                cmd += ["--top-k", str(int(self.top_k))]
            except ValueError:
                pass
        self.output = ""
        self.nerd_details = ""
        self.nerd_top5 = ""
        output_acc: list[str] = []

        def on_line(line: str) -> None:
            if line.startswith("@@ABROBA_TOKEN@@"):
                try:
                    output_acc.append(json.loads(line[len("@@ABROBA_TOKEN@@"):]))
                    self.output = "".join(output_acc)
                except (ValueError, TypeError):
                    pass
            elif line.startswith("@@ABROBA_NERD@@"):
                try:
                    info = json.loads(line[len("@@ABROBA_NERD@@"):])
                    def fmt(key):
                        entries = info.get(key, [])
                        return ", ".join(f"{x.get('text', '?')!r} {x.get('probability', 0):.1%}" for x in entries)
                    self.nerd_top5 = "Raw: " + fmt("top5_raw_logits") + " · Temp: " + fmt("top5_after_temperature") + " · Top-K: " + fmt("top5_after_top_k")
                    self.nerd_details = (f"Escolhido: {info.get('sampled_token_id')} ({info.get('sampled_probability', 0):.1%}) · "
                                         f"repetições n-gramas: {info.get('repeated_ngram_counts', {})} · " + self.nerd_top5)
                except (ValueError, TypeError):
                    pass
            elif line.startswith("@@ABROBA_DONE@@"):
                try:
                    info = json.loads(line[len("@@ABROBA_DONE@@"):])
                    if self.nerd_mode:
                        summary = f"{info.get('generated_tokens', 0)} tokens · prompt={info.get('prompt_tokens')} · temp={info.get('temperature')} · top_k={info.get('top_k')} · device={info.get('device')}"
                        self.nerd_details = summary + (" · " + self.nerd_top5 if self.nerd_top5 else "")
                except (ValueError, TypeError):
                    pass
            else:
                output_acc.append(line)
                self.output = "\n".join(output_acc)

        self.app.state.log.add(f"$ {' '.join(cmd[:4])}…", cp(C_PROMPT, bold=True))
        self.app.state.launch(cmd, PROJECT_ROOT, on_line=on_line)


# ──────────────────────────── tela de chat ───────────────────────────────────
class ChatScreen(Screen):
    def __init__(self, app: "TUIApp") -> None:
        super().__init__(app)
        self.history: list[tuple[str, str]] = []   # (quem, texto); quem: user | bot | sys
        self.turns: list[tuple[str, str]] = []     # (pergunta, resposta) usados no prompt
        self.input_buf = ""
        self.ckpt_idx  = 0
        self.loaded    = False
        self.model_obj = None
        self.tokenizer = None
        self.device    = None
        self._load_lock = threading.Lock()
        self._history_lock = threading.RLock()
        self._load_err  = ""
        self.generating = False
        self.nerd_mode = False
        self.nerd_details = ""
        self._generation_started = 0.0
        self._generated_token_count = 0

    def _checkpoints(self) -> list[Path]:
        return list_checkpoints()

    def draw(self, win: "curses._CursesWindow") -> None:
        h, w = win.getmaxyx()
        win.clear()
        draw_box(win, 0, 0, h-1, w, "CHAT INTERATIVO · AbrobaGPT")

        # Header
        ckpts = self._checkpoints()
        ckpt_name = ckpts[self.ckpt_idx].name if ckpts else "(nenhum)"
        status = ("⏳ Gerando resposta…" if self.generating else
                  "✓ Modelo carregado" if self.loaded else
                  ("✗ " + self._load_err if self._load_err else "Não carregado"))
        stat_attr = cp(C_SUCCESS, bold=True) if self.loaded else cp(C_ERROR, bold=True)
        safe_addstr(win, 1, 3, f"Checkpoint: {ckpt_name}  ←/→ trocar  L carregar", cp(C_DIM))
        safe_addstr(win, 2, 3, f"Status: {status} · N: Para nerds {'ON' if self.nerd_mode else 'OFF'}", stat_attr)
        if self.nerd_mode and self.nerd_details:
            detail_lines = textwrap.wrap(self.nerd_details, width=max(10, w-6))
            for detail_idx, detail_line in enumerate(detail_lines[:2]):
                safe_addstr(win, 3+detail_idx, 3, detail_line, cp(C_DIM))

        # Área de chat
        chat_y = 6 if self.nerd_mode else 4
        chat_h = h - (11 if self.nerd_mode else 9)
        draw_box(win, chat_y-1, 1, chat_h+2, w-2)
        with self._history_lock:
            history_snapshot = list(self.history)
        # Renderiza mensagens em linhas quebradas, para que o texto que chega
        # token a token continue visível também quando ultrapassa a largura.
        rendered: list[tuple[str, int]] = []
        content_width = max(10, w - 8)
        for who, text in history_snapshot:
            prefix = {"user": "Você: ", "bot": "AbrobaGPT: "}.get(who, "· ")
            attr = (cp(C_PROMPT, bold=True) if who == "user"
                    else cp(C_SUCCESS) if who == "bot" else cp(C_DIM))
            wrapped = textwrap.wrap(str(text), width=max(1, content_width - len(prefix)),
                                    replace_whitespace=False, drop_whitespace=False) or [""]
            rendered.append((prefix + wrapped[0], attr))
            rendered.extend(("  " + line, attr) for line in wrapped[1:])
        visible = rendered[-max(1, chat_h):]
        for i, (line, attr) in enumerate(visible):
            safe_addstr(win, chat_y+i, 3, line[:w-5], attr)

        # Input
        inp_y = h - 4
        draw_box(win, inp_y-1, 1, 3, w-2, "Mensagem")
        cursor_indicator = "▌"
        safe_addstr(win, inp_y, 3, (self.input_buf + cursor_indicator)[-(w-6):], cp(C_INPUT, bold=True))

        # Hints
        hints = "Enter enviar  Ctrl+C limpar  ESC voltar  ←/→ checkpoint  L carregar modelo  F2 Para nerds"
        if self.generating:
            hints += "  ·  Gerando token a token…"
        safe_addstr(win, h-2, 2, hints[:w-4], cp(C_DIM))

    def handle_key(self, key: int) -> Optional[str]:
        if key == 27:
            return "home"
        ckpts = self._checkpoints()
        if key == curses.KEY_LEFT and ckpts and not self.generating:
            self.ckpt_idx = (self.ckpt_idx - 1) % len(ckpts)
            self.loaded = False
            self.model_obj = None
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        elif key == curses.KEY_RIGHT and ckpts and not self.generating:
            self.ckpt_idx = (self.ckpt_idx + 1) % len(ckpts)
            self.loaded = False
            self.model_obj = None
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        elif key == curses.KEY_F2 and not self.generating:
            self.nerd_mode = not self.nerd_mode
            self.nerd_details = "Modo nerd ativado: top-5 probabilidades do próximo token." if self.nerd_mode else ""
        elif key == ord("l") or key == ord("L"):
            if ckpts and not self.loaded:
                self._load_model(ckpts)
        elif key in (curses.KEY_ENTER, 10, 13):
            if self.input_buf.strip():
                self._send_message()
        elif key in (3,):   # Ctrl+C limpa input
            self.input_buf = ""
        elif key in (curses.KEY_BACKSPACE, 127, 8):
            self.input_buf = self.input_buf[:-1]
        elif 32 <= key < 127:
            self.input_buf += chr(key)
        return None

    def _load_model(self, ckpts: list[Path]) -> None:
        def _do_load() -> None:
            try:
                import importlib, sys as _sys
                _sys.path.insert(0, str(PROJECT_ROOT))
                from src.inference import load_inference_components
                model, tok, dev, ckpt = load_inference_components(
                    checkpoint_path=str(ckpts[self.ckpt_idx])
                )
                with self._load_lock:
                    self.model_obj = model
                    self.tokenizer = tok
                    self.device    = dev
                    self.loaded    = True
                    self._load_err = ""
                self.turns.clear()
                self.history.append(("sys", f"Modelo '{ckpts[self.ckpt_idx].name}' pronto!"))
            except Exception as exc:
                with self._load_lock:
                    self._load_err = str(exc)[:60]
        t = threading.Thread(target=_do_load, daemon=True)
        t.start()
        self.history.append(("sys", f"Carregando {ckpts[self.ckpt_idx].name}…"))

    def _generation_params(self) -> tuple[int, float, Optional[int]]:
        """Lê max tokens / temperatura / top-k da tela de geração (com valores padrão)."""
        gen = self.app.screens.get("generate")
        try:
            max_tokens = max(1, min(400, int(getattr(gen, "max_tokens", "80"))))
        except (TypeError, ValueError):
            max_tokens = 80
        try:
            temperature = float(getattr(gen, "temperature", ""))
            if temperature <= 0:
                raise ValueError
        except (TypeError, ValueError):
            temperature = float(self.app.state.config.get("temperature", DEFAULT_TEMPERATURE))
        try:
            top_k: Optional[int] = int(getattr(gen, "top_k", "")) or None
        except (TypeError, ValueError):
            top_k = None
        return max_tokens, temperature, top_k

    def _send_message(self) -> None:
        msg = self.input_buf.strip()
        if not msg:
            return
        if self.generating:
            with self._history_lock:
                self.history.append(("sys", "Aguarde a resposta atual terminar antes de enviar outra mensagem."))
            self.input_buf = ""
            return
        self.input_buf = ""
        with self._history_lock:
            self.history.append(("user", msg))
        if not self.loaded or self.model_obj is None:
            with self._history_lock:
                self.history.append(("sys", "Modelo não carregado. Pressione L para carregar."))
            return
        max_tokens, temperature, top_k = self._generation_params()
        turns = list(self.turns)
        self.generating = True
        self._generation_started = time.perf_counter()
        self._generated_token_count = 0
        self.nerd_details = "Preparando geração…" if self.nerd_mode else ""
        with self._history_lock:
            self.history.append(("bot", "▌"))
            bot_history_index = len(self.history) - 1

        def _do_gen() -> None:
            import torch
            streamed_parts: list[str] = []
            try:
                tokenizer = self.tokenizer
                block_size = int(self.model_obj.block_size)
                # Reserva espaço de saída sem ultrapassar o contexto do modelo.
                requested = min(max_tokens, max(1, block_size - 1))
                budget = max(1, block_size - requested)
                prompt = build_prompt(turns, msg, lambda text: len(tokenizer.encode(text)), budget)
                enc = tokenizer.encode(prompt)
                if not enc:
                    raise ValueError("Não consegui tokenizar a mensagem.")
                if len(enc) > block_size:
                    raise ValueError(
                        f"A mensagem atual ocupa {len(enc)} tokens, mas o limite do modelo é {block_size}. "
                        "Encurte a mensagem e tente novamente."
                    )
                effective_tokens = min(requested, block_size - len(enc))
                if effective_tokens < 1:
                    raise ValueError("A mensagem ocupa todo o contexto do modelo; não há espaço para gerar.")
                tokens = torch.tensor([enc], dtype=torch.long, device=self.device)

                def on_diagnostics(info: dict[str, Any]) -> None:
                    if not self.nerd_mode:
                        return
                    def fmt(key: str) -> str:
                        parts = []
                        for token_id, probability in info.get(key, []):
                            token_text = tokenizer.decode([token_id]).replace("\n", "↵").replace("\t", "⇥")
                            parts.append(f"{token_text!r} {probability:.1%}")
                        return " | ".join(parts)
                    elapsed = max(0.001, time.perf_counter() - self._generation_started)
                    self.nerd_details = (f"Passo {info.get('step', 0)} · escolhido=#{info.get('sampled_token_id')} "
                                         f"({info.get('sampled_probability', 0):.1%}) · repetição={info.get('repeated_ngram_counts', {})} · "
                                         f"raw: {fmt('top5_raw_logits')} · temp: {fmt('top5_after_temperature')} · top-k: {fmt('top5_after_top_k')} · "
                                         f"{self._generated_token_count / elapsed:.1f} tok/s")

                def on_token(generated_ids: list[int]) -> None:
                    if not generated_ids:
                        return
                    # generate() passa a sequência acumulada; decodifica-se só o token novo.
                    piece = tokenizer.decode([generated_ids[-1]])
                    if not piece:
                        return
                    streamed_parts.append(piece)
                    self._generated_token_count += 1
                    with self._history_lock:
                        if bot_history_index < len(self.history):
                            self.history[bot_history_index] = ("bot", "".join(streamed_parts) + "▌")

                with torch.no_grad():
                    generated = self.model_obj.generate(
                        tokens,
                        max_new_tokens=effective_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        stop_fn=make_stop_fn(tokenizer.decode, CHAT_STOP_SEQUENCES),
                        token_callback=on_token,
                        diagnostics_callback=on_diagnostics if self.nerd_mode else None,
                    )
                ids = generated[0, len(enc):].tolist()
                response = clean_response(tokenizer.decode(ids)) or "…"
                self.turns.append((msg, response))
                if self.nerd_mode:
                    elapsed = max(0.001, time.perf_counter() - self._generation_started)
                    self.nerd_details = (f"Concluído · prompt={len(enc)} tokens · gerados={self._generated_token_count} "
                                         f"· {self._generated_token_count / elapsed:.1f} tok/s · temp={temperature} · top_k={top_k} · device={self.device}")
            except Exception as exc:
                response = f"[Erro: {exc}]"
            finally:
                with self._history_lock:
                    if bot_history_index < len(self.history):
                        self.history[bot_history_index] = ("bot", response)
                self.generating = False

        t = threading.Thread(target=_do_gen, daemon=True)
        t.start()


# ─────────────────────── tela de preparação de dados ─────────────────────────
class PrepareScreen(Screen):
    def __init__(self, app: "TUIApp") -> None:
        super().__init__(app)
        self.dataset_idx = 0
        self.source_idx  = 0
        self.size_mb     = "10"
        self.field       = 0

    def draw(self, win: "curses._CursesWindow") -> None:
        h, w = win.getmaxyx()
        win.clear()
        draw_box(win, 0, 0, h-1, w, "PREPARAÇÃO DE DATASET")
        col = 4

        safe_addstr(win, 2, col, "Baixe e prepare o corpus de treinamento:", cp(C_TITLE, bold=True))

        ds_lbl = DATASET_LABELS[self.dataset_idx]
        src_lbl = ["Arquivo local", "HuggingFace Streaming"][self.source_idx]

        self._draw_field(win, 4, col, "Dataset  ", f"[ {ds_lbl} ]",   self.field == 0, w)
        self._draw_field(win, 5, col, "Fonte    ", f"[ {src_lbl} ]",  self.field == 1, w)
        self._draw_field(win, 6, col, "Limite MB", self.size_mb,       self.field == 2, w)

        btn_attr = cp(C_SELECTED, bold=True) if self.field == 3 else cp(C_SUCCESS, bold=True)
        busy = self.app.state.is_busy()
        safe_addstr(win, 8, col,
                    "  [BAIXANDO / PREPARANDO...]  " if busy else "  [PREPARAR DADOS]  ",
                    btn_attr)
        status = "Em andamento…" if busy else "Aguardando ação"
        if DATASET_TYPES[self.dataset_idx] == "ragqa":
            raw_ragqa = PROJECT_ROOT / "data" / "raw" / "rag-qa-fulltext-ptbr" / "train.jsonl"
            metadata_path = PROJECT_ROOT / "data" / "processed" / "metadata.json"
            prepared = False
            try:
                with metadata_path.open("r", encoding="utf-8") as metadata_file:
                    prepared = json.load(metadata_file).get("dataset_type") == "ragqa"
            except (OSError, json.JSONDecodeError, AttributeError):
                pass
            if busy:
                status = "Baixando / preparando"
            elif prepared:
                status = "Concluído · dataset ativo"
            elif raw_ragqa.is_file() and raw_ragqa.stat().st_size:
                status = "Baixado · preparação pendente"
            else:
                status = "Ainda não baixado"
        safe_addstr(win, 9, col, f"Status: {status}"[:w-col-2], cp(C_SUCCESS if status.startswith("Concluído") else C_DIM))

        # Log
        log_y = 11
        log_h = h - log_y - 3
        if log_h > 2:
            draw_box(win, log_y-1, 1, log_h+2, w-2, "LOG")
            self.app.state.log.draw(win, log_y, 2, log_h, w-4)

        safe_addstr(win, h-2, col,
                    "ESC voltar  ↑↓ campos  ←→/Espaço alternar  Enter confirmar", cp(C_DIM))

    def _draw_field(self, win: "curses._CursesWindow", y: int, x: int,
                    label: str, value: str, selected: bool, w: int) -> None:
        lbl = cp(C_PROMPT, bold=True) if selected else cp(C_DIM)
        val = cp(C_SELECTED, bold=True) if selected else cp(C_WARNING)
        safe_addstr(win, y, x, f"{label}: ", lbl)
        safe_addstr(win, y, x+11, value[:w-x-14], val)

    def handle_key(self, key: int) -> Optional[str]:
        if key == 27:
            return "home"
        busy = self.app.state.is_busy()
        if key == ord("k") and busy:
            self.app.state.kill()
            return None
        if busy:
            return None
        if key == curses.KEY_UP:
            self.field = (self.field - 1) % 4
        elif key == curses.KEY_DOWN:
            self.field = (self.field + 1) % 4
        elif key in (curses.KEY_LEFT, curses.KEY_RIGHT, ord(" ")):
            if self.field == 0:
                self.dataset_idx = cycle_index(self.dataset_idx, key, len(DATASET_TYPES))
            elif self.field == 1:
                self.source_idx = 1 - self.source_idx
        elif key in (curses.KEY_ENTER, 10, 13):
            if self.field == 3:
                self._start()
            else:
                self.field = (self.field + 1) % 4
        elif self.field == 2:
            if key in (curses.KEY_BACKSPACE, 127, 8):
                self.size_mb = self.size_mb[:-1]
            elif chr(key).isdigit() or chr(key) == ".":
                self.size_mb += chr(key)
        return None

    def _start(self) -> None:
        cmd = [sys.executable, "train.py", "--prepare-only",
               "--dataset-type", DATASET_TYPES[self.dataset_idx],
               "--source",       ["local", "hf"][self.source_idx]]
        if self.source_idx == 1:
            cmd += ["--size-mb", self.size_mb or "10"]
        self.app.state.log.add(f"$ {' '.join(cmd)}", cp(C_PROMPT, bold=True))
        self.app.state.launch(cmd, PROJECT_ROOT)


# ───────────────────── tela de avaliação de checkpoints ──────────────────────
class EvaluationScreen(Screen):
    """Configura a avaliação; o trabalho pesado roda no filho evaluate.py."""
    def __init__(self, app: "TUIApp") -> None:
        super().__init__(app)
        self.cursor = 0
        self.selected: set[str] = set()
        self.preset_index = 0
        self.questions, self.generations = "3", "1"
        self.max_tokens, self.temperature, self.top_k = "80", "0.75", ""
        self.repetition_penalty, self.seed = "1.0", ""
        self.greedy = False
        self.validation_loss = False
        self.last_report = ""
        self.recommended = ""
        self.show_results = False
        self.result_cursor = 0
        self.message = "Selecione checkpoints com Espaço."

    def _presets(self) -> list[Path]:
        return list_presets()

    def _fields(self, ckpts: list[Path]) -> list[str]:
        return [*(f"ckpt:{path}" for path in ckpts), "preset", "questions", "generations", "greedy", "temperature", "top_k", "penalty", "seed", "validation_loss", "run", "copy"]

    def draw(self, win: "curses._CursesWindow") -> None:
        h, w = win.getmaxyx(); win.clear()
        if self.show_results:
            self._draw_results(win, h, w)
            return
        ckpts, presets = list_checkpoints(), self._presets()
        fields = self._fields(ckpts)
        self.cursor = clamp(self.cursor, 0, max(0, len(fields) - 1))
        draw_box(win, 0, 0, h-1, w, "TESTAR CHECKPOINTS")
        if not ckpts:
            safe_addstr(win, 2, 3, "Nenhum checkpoint disponível.", cp(C_WARNING, bold=True)); return
        safe_addstr(win, 2, 3, "Checkpoints (Espaço seleciona; todos recebem as mesmas perguntas/configuração):", cp(C_TITLE, bold=True))
        y = 3
        for index, path in enumerate(ckpts[:max(1, min(6, h - 16))]):
            marker = "[X]" if str(path) in self.selected else "[ ]"
            attr = cp(C_SELECTED, bold=True) if self.cursor == index else cp(C_DIM)
            safe_addstr(win, y + index, 4, f"{marker} {path.name}"[:w-8], attr)
        base = y + min(len(ckpts), max(1, min(6, h - 16))) + 1
        preset_name = presets[self.preset_index % len(presets)].stem if presets else "(nenhum preset)"
        rows = [
            ("Preset", preset_name), ("Perguntas", self.questions), ("Gerações por pergunta", self.generations),
            ("Modo", "Greedy" if self.greedy else "Amostragem"), ("Temperatura", self.temperature),
            ("Top-K (vazio=todos)", self.top_k or "—"), ("Penalidade repetição", self.repetition_penalty), ("Seed (vazio=aleatória)", self.seed or "—"), ("Loss de validação", "Sim" if self.validation_loss else "Não"),
        ]
        offset = len(ckpts)
        for index, (label, value) in enumerate(rows):
            selected = self.cursor == offset + index
            safe_addstr(win, base + index, 4, f"{label:<25}: {value}"[:w-8], cp(C_SELECTED, bold=True) if selected else cp(C_DIM))
        run_index, copy_index = offset + len(rows), offset + len(rows) + 1
        busy = self.app.state.is_busy()
        safe_addstr(win, base + len(rows) + 1, 4, "  [EXECUTANDO...]  " if busy else "  [INICIAR AVALIAÇÃO]  ", cp(C_WARNING if busy else C_SUCCESS, bold=True) if self.cursor == run_index else cp(C_DIM))
        copy_label = "  [COPIAR RECOMENDADO PARA evaluated_best/]  "
        safe_addstr(win, base + len(rows) + 2, 4, copy_label[:w-8], cp(C_SELECTED, bold=True) if self.cursor == copy_index else cp(C_DIM))
        if self.last_report:
            safe_addstr(win, base + len(rows) + 4, 4, f"Relatório: {self.last_report}"[:w-8], cp(C_SUCCESS))
            safe_addstr(win, base + len(rows) + 5, 4, f"Recomendado: {Path(self.recommended).name if self.recommended else 'indisponível'}"[:w-8], cp(C_SUCCESS))
        safe_addstr(win, h-2, 2, "ESC voltar  ↑↓ navegar  ←→/Espaço alterar  Enter editar/executar  K cancelar", cp(C_DIM))

    def _draw_results(self, win: "curses._CursesWindow", h: int, w: int) -> None:
        draw_box(win, 0, 0, h-1, w, "RESULTADOS · COMPARAÇÃO DE CHECKPOINTS")
        try:
            report = json.loads(Path(self.last_report).read_text(encoding="utf-8"))
            items = [item for item in report.get("checkpoints", []) if not item.get("error")]
        except (OSError, ValueError):
            items = []
        if not items:
            safe_addstr(win, 3, 4, "Não foi possível abrir o relatório.", cp(C_ERROR, bold=True)); return
        self.result_cursor = clamp(self.result_cursor, 0, len(items)-1)
        safe_addstr(win, 2, 3, "Checkpoint                    Geral  Ref.  Instr. Estr. Repet. Div.  Val.loss", cp(C_HEADER, bold=True))
        for index, item in enumerate(items[:max(1, h - 11)]):
            metrics = item["metrics"]
            val = metrics.get("validation_loss")
            row = f"{Path(item['checkpoint']).name[:28]:<28} {metrics['overall_score']:.2f}  {metrics['reference_match']:.2f}  {metrics['instruction_following']:.2f}  {metrics['structural_quality']:.2f}  {metrics['repetition_rate']:.2f}  {metrics['diversity']:.2f}  {val:.3f}" if val is not None else f"{Path(item['checkpoint']).name[:28]:<28} {metrics['overall_score']:.2f}  {metrics['reference_match']:.2f}  {metrics['instruction_following']:.2f}  {metrics['structural_quality']:.2f}  {metrics['repetition_rate']:.2f}  {metrics['diversity']:.2f}  n/d"
            safe_addstr(win, 3+index, 3, row[:w-6], cp(C_SELECTED, bold=True) if index == self.result_cursor else cp(C_DIM))
        selected = items[self.result_cursor]
        details = selected.get("results", [])
        if details:
            answer = details[0]["generations"][0]["text"].replace("\n", " ")
            safe_addstr(win, h-6, 3, f"Exemplo {details[0]['id']}: {answer}"[:w-6], cp(C_DIM))
        safe_addstr(win, h-4, 3, "A pontuação combina sinais configuráveis; não define inteligência geral.", cp(C_WARNING))
        safe_addstr(win, h-2, 2, "ESC voltar à configuração  ↑↓ escolher  Enter escolher manualmente  C copiar escolhido", cp(C_DIM))

    def _edit_value(self, key: str, current: str, char: int) -> None:
        if char in (curses.KEY_BACKSPACE, 127, 8): setattr(self, key, current[:-1])
        elif 32 <= char < 127 and chr(char) in "0123456789.-": setattr(self, key, current + chr(char))

    def handle_key(self, key: int) -> Optional[str]:
        if self.show_results:
            if key == 27: self.show_results = False; return None
            try:
                report = json.loads(Path(self.last_report).read_text(encoding="utf-8")); items = [item for item in report.get("checkpoints", []) if not item.get("error")]
            except (OSError, ValueError): items = []
            if key == curses.KEY_UP and items: self.result_cursor = (self.result_cursor - 1) % len(items)
            elif key == curses.KEY_DOWN and items: self.result_cursor = (self.result_cursor + 1) % len(items)
            elif key in (curses.KEY_ENTER, 10, 13) and items:
                self.recommended = items[self.result_cursor]["checkpoint"]; self.app.state.set_status("Checkpoint preferido selecionado manualmente", "ok")
            elif key in (ord("c"), ord("C")) and items:
                try: copy_selected_checkpoint(items[self.result_cursor]["checkpoint"]); self.app.state.set_status("Cópia do checkpoint selecionado criada", "ok")
                except (OSError, RuntimeError) as exc: self.app.state.set_status(str(exc), "err")
            return None
        if key == 27: return "home"
        if key in (ord("r"), ord("R")) and self.last_report: self.show_results = True; return None
        if key in (ord("k"), ord("K")) and self.app.state.is_busy(): self.app.state.kill(); return None
        if self.app.state.is_busy(): return None
        ckpts, presets = list_checkpoints(), self._presets(); fields = self._fields(ckpts)
        if not fields: return None
        if key == curses.KEY_UP: self.cursor = (self.cursor - 1) % len(fields); return None
        if key == curses.KEY_DOWN: self.cursor = (self.cursor + 1) % len(fields); return None
        field = fields[self.cursor]
        if field.startswith("ckpt:"):
            if key in (ord(" "), curses.KEY_ENTER, 10, 13):
                path = field[5:]; self.selected.symmetric_difference_update({path})
            return None
        if field == "preset" and presets and key in (curses.KEY_LEFT, curses.KEY_RIGHT, ord(" "), curses.KEY_ENTER, 10, 13):
            self.preset_index = cycle_index(self.preset_index, key if key in (curses.KEY_LEFT, curses.KEY_RIGHT) else curses.KEY_RIGHT, len(presets)); return None
        if field == "greedy" and key in (curses.KEY_LEFT, curses.KEY_RIGHT, ord(" "), curses.KEY_ENTER, 10, 13): self.greedy = not self.greedy; return None
        if field == "validation_loss" and key in (curses.KEY_LEFT, curses.KEY_RIGHT, ord(" "), curses.KEY_ENTER, 10, 13): self.validation_loss = not self.validation_loss; return None
        if field == "run" and key in (curses.KEY_ENTER, 10, 13): self._start(ckpts, presets); return None
        if field == "copy" and key in (curses.KEY_ENTER, 10, 13):
            try:
                if not self.recommended: raise ValueError("Execute uma avaliação com resultado primeiro.")
                target = copy_selected_checkpoint(self.recommended); self.message = f"Cópia preservada em {target}"
                self.app.state.set_status("Checkpoint recomendado copiado", "ok")
            except (OSError, ValueError, RuntimeError) as exc: self.app.state.set_status(str(exc), "err")
            return None
        mapping = {"questions": "questions", "generations": "generations", "temperature": "temperature", "top_k": "top_k", "penalty": "repetition_penalty", "seed": "seed"}
        if field in mapping: self._edit_value(mapping[field], getattr(self, mapping[field]), key)
        return None

    def _start(self, ckpts: list[Path], presets: list[Path]) -> None:
        chosen = [path for path in ckpts if str(path) in self.selected]
        if not chosen: self.app.state.set_status("Selecione pelo menos um checkpoint", "err"); return
        if not presets: self.app.state.set_status("Nenhum preset JSON encontrado", "err"); return
        try:
            int(self.questions); int(self.generations); int(self.max_tokens)
            float(self.temperature); float(self.repetition_penalty)
            if self.top_k: int(self.top_k)
            if self.seed: int(self.seed)
        except ValueError: self.app.state.set_status("Parâmetros de avaliação inválidos", "err"); return
        preset = presets[self.preset_index % len(presets)]
        cmd = [sys.executable, "evaluate.py", "--preset", str(preset), "--questions", self.questions, "--generations", self.generations, "--max-tokens", self.max_tokens, "--temperature", self.temperature, "--repetition-penalty", self.repetition_penalty]
        for path in chosen: cmd += ["--checkpoint", str(path)]
        if self.greedy: cmd.append("--greedy")
        if self.top_k: cmd += ["--top-k", self.top_k]
        if self.seed: cmd += ["--seed", self.seed]
        if self.validation_loss: cmd.append("--validation-loss")
        self.app.state.progress.update(0, len(chosen) * int(self.questions) * int(self.generations), "Avaliação", "Preparando...")
        self.app.state.log.add(f"$ {' '.join(cmd)}", cp(C_PROMPT, bold=True))
        def on_line(line: str) -> None:
            if line.startswith("@@ABROBA_EVAL@@"):
                try:
                    item = json.loads(line[len("@@ABROBA_EVAL@@"):]); self.app.state.progress.update(item["completed"], item["total"], "Avaliação", f"{item['checkpoint']} · {item['question']}")
                except (ValueError, KeyError): pass
            elif line.startswith("@@ABROBA_EVAL_DONE@@"):
                try:
                    item = json.loads(line[len("@@ABROBA_EVAL_DONE@@"):]); self.last_report, self.recommended = item["report"], item.get("recommended") or ""; self.show_results = True
                except (ValueError, KeyError): pass
        self.app.state.launch(cmd, PROJECT_ROOT, on_line=on_line)


# ──────────────────── tela de gerenciamento de checkpoints ───────────────────
class CheckpointScreen(Screen):
    def __init__(self, app: "TUIApp") -> None:
        super().__init__(app)
        self.cursor = 0

    def draw(self, win: "curses._CursesWindow") -> None:
        h, w = win.getmaxyx()
        win.clear()
        ckpts = list_checkpoints()
        draw_box(win, 0, 0, h-1, w, f"CHECKPOINTS ({len(ckpts)} encontrados)")

        if not ckpts:
            safe_addstr(win, 3, 4, "Nenhum checkpoint encontrado.", cp(C_WARNING, bold=True))
            safe_addstr(win, 4, 4, "Treine o modelo primeiro.", cp(C_DIM))
        else:
            headers = f"  {'Nome':<30} {'Tamanho':>10} {'Modificado':>22}"
            safe_addstr(win, 2, 2, headers[:w-4], cp(C_HEADER, bold=True))
            list_h = h - 7
            visible = ckpts[max(0, self.cursor-list_h+1): self.cursor+1]
            start_idx = max(0, self.cursor - list_h + 1)
            for i, ckpt in enumerate(ckpts[:list_h]):
                idx = i
                name = ckpt.name[:30]
                size = human_size(ckpt.stat().st_size)
                mtime = datetime.fromtimestamp(ckpt.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                marker = "►" if idx == self.cursor else " "
                row = f"{marker} {name:<30} {size:>10} {mtime:>22}"
                attr = cp(C_SELECTED, bold=True) if idx == self.cursor else cp(C_DIM)
                safe_addstr(win, 3+i, 2, row[:w-4], attr)

        hints = "ESC voltar  ↑↓ navegar  D deletar selecionado  R renomear"
        safe_addstr(win, h-2, 2, hints[:w-4], cp(C_DIM))

    def handle_key(self, key: int) -> Optional[str]:
        if key == 27:
            return "home"
        ckpts = list_checkpoints()
        if not ckpts:
            return None
        if key == curses.KEY_UP:
            self.cursor = clamp(self.cursor - 1, 0, len(ckpts)-1)
        elif key == curses.KEY_DOWN:
            self.cursor = clamp(self.cursor + 1, 0, len(ckpts)-1)
        elif key in (ord("d"), ord("D")):
            if 0 <= self.cursor < len(ckpts):
                try:
                    ckpts[self.cursor].unlink()
                    self.app.state.set_status(f"Removido: {ckpts[self.cursor].name}", "warn")
                    self.cursor = clamp(self.cursor, 0, len(ckpts)-2)
                except OSError as e:
                    self.app.state.set_status(f"Erro ao remover: {e}", "err")
        return None


# ─────────────────────── tela de configurações ───────────────────────────────
class ConfigScreen(Screen):
    _FIELDS = [
        ("n_layer",               "Camadas Transformer",    "int"),
        ("n_embd",                "Dimensão embedding",     "int"),
        ("n_head",                "Cabeças de atenção",     "int"),
        ("block_size",            "Tamanho do bloco",       "int"),
        ("batch_size",            "Batch size",             "int"),
        ("max_iters",             "Iterações máximas",      "int"),
        ("eval_interval",         "Intervalo de avaliação", "int"),
        ("save_interval",         "Intervalo de salvamento","int"),
        ("learning_rate",         "Learning rate",          "float"),
        ("min_learning_rate",     "LR mínimo",              "float"),
        ("dropout",               "Dropout",                "float"),
        ("patience",              "Paciência early stop",   "int"),
        ("temperature",           "Temperatura de geração", "float"),
        ("keep_checkpoints",      "Checkpoints mantidos (0=todos)", "int"),
        ("seed",                  "Seed aleatória",         "int"),
    ]

    def __init__(self, app: "TUIApp") -> None:
        super().__init__(app)
        self.cursor = 0
        self.editing = False
        self.edit_buf = ""
        self.message = ""
        self.msg_attr = 0

    def draw(self, win: "curses._CursesWindow") -> None:
        h, w = win.getmaxyx()
        win.clear()
        draw_box(win, 0, 0, h-1, w, "CONFIGURAÇÕES · base_config.json")
        cfg = self.app.state.config
        list_h = min(len(self._FIELDS), h-6)

        safe_addstr(win, 2, 4, "Parâmetro", cp(C_HEADER, bold=True))
        safe_addstr(win, 2, 42, "Valor atual", cp(C_HEADER, bold=True))
        safe_addstr(win, 2, 58, "Descrição", cp(C_HEADER, bold=True))

        for i in range(list_h):
            if i >= len(self._FIELDS):
                break
            key, desc, _ = self._FIELDS[i]
            val = str(cfg.get(key, ""))
            is_cur = i == self.cursor
            row_attr = cp(C_SELECTED, bold=True) if is_cur else cp(C_DIM)
            key_attr = cp(C_PROMPT, bold=True) if is_cur else cp(C_WARNING)
            val_attr = cp(C_INPUT, bold=True) if (is_cur and self.editing) else (cp(C_SUCCESS) if is_cur else cp(C_DIM))

            safe_addstr(win, 3+i, 4,  key[:35],  key_attr)
            if is_cur and self.editing:
                safe_addstr(win, 3+i, 42, (self.edit_buf + "▌")[:14], val_attr)
            else:
                safe_addstr(win, 3+i, 42, val[:14], val_attr)
            safe_addstr(win, 3+i, 58, desc[:w-60], cp(C_DIM))

        # Botão salvar
        save_y = list_h + 4
        is_save = self.cursor == len(self._FIELDS)
        btn_attr = cp(C_SELECTED, bold=True) if is_save else cp(C_SUCCESS, bold=True)
        safe_addstr(win, save_y, 4, "  [SALVAR CONFIGURAÇÕES]  ", btn_attr)

        if self.message:
            safe_addstr(win, save_y+1, 4, self.message[:w-8], self.msg_attr)

        hints = "ESC voltar  ↑↓ navegar  Enter editar/confirmar  S salvar"
        safe_addstr(win, h-2, 2, hints[:w-4], cp(C_DIM))

    def handle_key(self, key: int) -> Optional[str]:
        if key == 27 and not self.editing:
            return "home"
        total = len(self._FIELDS) + 1   # +1 para o botão salvar

        if self.editing:
            if key in (curses.KEY_ENTER, 10, 13):
                self._commit_edit()
            elif key == 27:
                self.editing = False
                self.edit_buf = ""
            elif key in (curses.KEY_BACKSPACE, 127, 8):
                self.edit_buf = self.edit_buf[:-1]
            elif 32 <= key < 127:
                self.edit_buf += chr(key)
        else:
            if key == curses.KEY_UP:
                self.cursor = (self.cursor - 1) % total
            elif key == curses.KEY_DOWN:
                self.cursor = (self.cursor + 1) % total
            elif key in (curses.KEY_ENTER, 10, 13):
                if self.cursor == len(self._FIELDS):
                    self._save()
                else:
                    key_name, _, _ = self._FIELDS[self.cursor]
                    self.edit_buf = str(self.app.state.config.get(key_name, ""))
                    self.editing = True
            elif key in (ord("s"), ord("S")):
                self._save()
        return None

    def _commit_edit(self) -> None:
        self.editing = False
        key_name, _, type_ = self._FIELDS[self.cursor]
        previous = self.app.state.config.get(key_name)
        try:
            if type_ == "int":
                self.app.state.config[key_name] = int(self.edit_buf)
            else:
                self.app.state.config[key_name] = float(self.edit_buf)
            if key_name == "temperature" and self.app.state.config[key_name] <= 0:
                raise ValueError("temperatura deve ser > 0")
            if key_name == "keep_checkpoints" and self.app.state.config[key_name] < 0:
                raise ValueError("não pode ser negativo")
            self.message = f"✓ {key_name} atualizado"
            self.msg_attr = cp(C_SUCCESS, bold=True)
        except ValueError:
            self.app.state.config[key_name] = previous
            self.message = f"✗ Valor inválido para {key_name}"
            self.msg_attr = cp(C_ERROR, bold=True)
        self.edit_buf = ""

    def _save(self) -> None:
        if save_config_safe(self.app.state.config):
            gen_screen = self.app.screens.get("generate")
            if gen_screen is not None:  # a tela de geração passa a usar a nova temperatura padrão
                gen_screen.temperature = str(self.app.state.config.get("temperature", DEFAULT_TEMPERATURE))
            self.message = "✓ Configurações salvas!"
            self.msg_attr = cp(C_SUCCESS, bold=True)
            self.app.state.set_status("Config salva", "ok")
        else:
            self.message = "✗ Erro ao salvar"
            self.msg_attr = cp(C_ERROR, bold=True)


# ──────────────────────────── tela de testes ─────────────────────────────────
class TestsScreen(Screen):
    def __init__(self, app: "TUIApp") -> None:
        super().__init__(app)
        self.results: list[tuple[str, bool, str]] = []
        self._running = False

    def draw(self, win: "curses._CursesWindow") -> None:
        h, w = win.getmaxyx()
        win.clear()
        draw_box(win, 0, 0, h-1, w, "TESTES DO SISTEMA")
        col = 4

        safe_addstr(win, 2, col,
                    "Executa os testes unitários do AbrobaGPT:",
                    cp(C_TITLE, bold=True))

        busy = self.app.state.is_busy()
        btn_attr = cp(C_WARNING, bold=True) if busy else cp(C_SELECTED if not self._running else C_DIM, bold=True)
        safe_addstr(win, 4, col,
                    "  [EXECUTANDO TESTES...]  " if busy else "  [EXECUTAR TESTES]  ",
                    btn_attr)

        # Resultados
        for i, (name, ok, detail) in enumerate(self.results[:8]):
            icon = "✓" if ok else "✗"
            attr = cp(C_SUCCESS, bold=True) if ok else cp(C_ERROR, bold=True)
            safe_addstr(win, 6+i, col, f"{icon} {name}: {detail}"[:w-8], attr)

        # Log
        log_y = 6 + len(self.results) + 2
        log_h = h - log_y - 3
        if log_h > 2:
            draw_box(win, log_y-1, 1, log_h+2, w-2, "SAÍDA DOS TESTES")
            self.app.state.log.draw(win, log_y, 2, log_h, w-4)

        safe_addstr(win, h-2, col,
                    "ESC voltar  Enter executar testes",
                    cp(C_DIM))

    def handle_key(self, key: int) -> Optional[str]:
        if key == 27:
            return "home"
        busy = self.app.state.is_busy()
        if key in (curses.KEY_ENTER, 10, 13, ord("r"), ord("R")) and not busy:
            self._run_tests()
        return None

    def _run_tests(self) -> None:
        self.results = []
        self._running = True
        cmd = [sys.executable, "test_llm.py"]
        self.app.state.log.add("Iniciando testes…", cp(C_WARNING, bold=True))

        def on_line(line: str) -> None:
            if "[OK]" in line:
                name = line.split("[OK]")[-1].strip().split(" PASSED")[0]
                self.results.append((name, True, "PASSED"))
            elif "TEST FAILED" in line:
                self.results.append(("Sistema", False, line[-40:]))

        self.app.state.launch(cmd, PROJECT_ROOT, on_line=on_line)


# ───────────────────────── tela de monitor ───────────────────────────────────
class MonitorScreen(Screen):
    def __init__(self, app: "TUIApp") -> None:
        super().__init__(app)

    def draw(self, win: "curses._CursesWindow") -> None:
        h, w = win.getmaxyx()
        win.clear()
        draw_box(win, 0, 0, h-1, w, "MONITOR DE PROCESSO")

        busy = self.app.state.is_busy()
        stat_msg, stat_attr = self.app.state.get_status()

        # Indicador de processo
        y = 2
        if busy:
            frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
            spin = frames[int(time.time()*10) % len(frames)]
            safe_addstr(win, y, 4, f"{spin} PROCESSO ATIVO", cp(C_WARNING, bold=True))
        else:
            safe_addstr(win, y, 4, "● NENHUM PROCESSO ATIVO", cp(C_DIM))

        safe_addstr(win, y+1, 4, f"Status: {stat_msg}", stat_attr)

        # Barra de progresso
        if busy:
            draw_box(win, y+3, 2, 6, w-4, "PROGRESSO")
            self.app.state.progress.draw(win, y+4, 4, w-8)

        # Log completo
        log_y = y + 11 if busy else y + 4
        log_h = h - log_y - 3
        if log_h > 2:
            draw_box(win, log_y-1, 1, log_h+2, w-2, "LOG COMPLETO")
            self.app.state.log.draw(win, log_y, 2, log_h, w-4)

        hints = "ESC voltar  K interromper processo" if busy else "ESC voltar"
        safe_addstr(win, h-2, 2, hints[:w-4], cp(C_DIM))

    def handle_key(self, key: int) -> Optional[str]:
        if key == 27:
            return "home"
        if key in (ord("k"), ord("K")) and self.app.state.is_busy():
            self.app.state.kill()
        return None


# ──────────────────────── aplicação principal ────────────────────────────────
class TUIApp:
    def __init__(self) -> None:
        self.state = AppState()
        self.screens: dict[str, Screen] = {}
        self.current_screen = "home"
        self._running = True

    def _build_screens(self) -> None:
        self.screens = {
            "home":        HomeScreen(self),
            "train":       TrainScreen(self),
            "generate":    GenerateScreen(self),
            "chat":        ChatScreen(self),
            "prepare":     PrepareScreen(self),
            "evaluation":  EvaluationScreen(self),
            "checkpoints": CheckpointScreen(self),
            "config":      ConfigScreen(self),
            "tests":       TestsScreen(self),
            "monitor":     MonitorScreen(self),
            "graphs":      GraphScreen(self),
        }

    def run(self, stdscr: "curses._CursesWindow") -> None:
        curses.curs_set(0)
        curses.noecho()
        curses.cbreak()
        stdscr.keypad(True)
        stdscr.nodelay(True)   # não bloqueia no getch
        init_colors()
        self._build_screens()

        while self._running:
            h, w = stdscr.getmaxyx()
            if h < 24 or w < 80:
                stdscr.clear()
                msg = f"Terminal muito pequeno ({w}x{h}). Necessário: 80x24"
                try:
                    stdscr.addstr(0, 0, msg, curses.color_pair(C_ERROR) | curses.A_BOLD)
                except curses.error:
                    pass
                stdscr.refresh()
                time.sleep(0.1)
                key = stdscr.getch()
                if key == ord("q") or key == ord("Q"):
                    break
                continue

            screen = self.screens.get(self.current_screen)
            if screen is None:
                break

            # Only clear if we are not on the home screen, or the home screen
            # handles its own clear (which it does for the initial logo draw).
            if self.current_screen != "home":
                stdscr.clear()

            screen.draw(stdscr)
            stdscr.refresh()

            key = stdscr.getch()
            if key != -1:
                next_screen = screen.handle_key(key)
                if next_screen == "quit":
                    break
                if next_screen and next_screen in self.screens:
                    self.current_screen = next_screen

            time.sleep(0.033)   # ~30 fps


def configure_utf8_console() -> None:
    """Configura UTF-8 no console Windows sem trocar os caracteres da TUI."""
    try:
        locale.setlocale(locale.LC_ALL, "")
    except locale.Error:
        pass

    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleOutputCP(65001)
            kernel32.SetConsoleCP(65001)
        except (AttributeError, OSError):
            try:
                os.system("chcp 65001 > nul 2>&1")
            except OSError:
                pass

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass



def main() -> None:
    parser = argparse.ArgumentParser(description="AbrobaGPT TUI")
    parser.parse_args()
    configure_utf8_console()
    try:
        app = TUIApp()
        curses.wrapper(app.run)
    except KeyboardInterrupt:
        pass
    finally:
        print("\033[0m\nAté mais! 🤖")


if __name__ == "__main__":
    main()
