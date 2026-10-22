"""Testes unitários para a sequência de imagem inline do protocolo iTerm2."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import TextIO


def build_iterm_image_sequence(image_bytes: bytes, width: str = "20") -> str:
    """Constrói a sequência de escape sem escrever diretamente no terminal."""
    if not image_bytes:
        raise ValueError("A imagem está vazia.")
    if not width or any(ch not in "0123456789%" for ch in width):
        raise ValueError("A largura deve ser um número ou percentual simples.")
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"\x1b]1337;File=inline=1;width={width}:{encoded}\x07"


def render_image(image_path: Path, width: str = "20", output: TextIO | None = None) -> bool:
    """Emite a sequência inline; retorna False se o arquivo não existir."""
    if not image_path.exists() or not image_path.is_file():
        return False
    sequence = build_iterm_image_sequence(image_path.read_bytes(), width)
    stream = output if output is not None else __import__("sys").stdout
    stream.write(sequence)
    stream.flush()
    return True


def test_image_render(tmp_path: Path) -> None:
    """O teste usa um arquivo temporário; não depende de fixture externa inexistente."""
    payload = b"fake-image-payload"
    image_path = tmp_path / "logo.jpg"
    image_path.write_bytes(payload)
    output = io.StringIO()

    assert render_image(image_path, output=output)
    sequence = output.getvalue()
    assert sequence.startswith("\x1b]1337;File=inline=1;width=20:")
    assert base64.b64encode(payload).decode("ascii") in sequence
    assert sequence.endswith("\x07")


def test_image_render_missing_file(tmp_path: Path) -> None:
    assert not render_image(tmp_path / "missing.jpg", output=io.StringIO())


def test_image_sequence_rejects_empty_or_invalid_input() -> None:
    import pytest

    with pytest.raises(ValueError):
        build_iterm_image_sequence(b"")
    with pytest.raises(ValueError):
        build_iterm_image_sequence(b"image", width="20;bad=1")


if __name__ == "__main__":
    import sys

    project_root = Path(__file__).resolve().parent
    logo_path = project_root / "visual" / "AbrobaGPT_logo.jpg"
    if not render_image(logo_path):
        print(f"Imagem não encontrada: {logo_path}")
        print("O teste unitário usa uma imagem temporária e não exige o logo opcional.")
