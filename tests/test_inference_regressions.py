"""Regression tests for AbrobaGPT chat inference and streaming primitives."""
from __future__ import annotations

import math

import pytest

from src.prompting import (
    CHAT_STOP_SEQUENCES,
    build_prompt,
    clean_response,
    make_stop_fn,
)
from server import ApiError, _chat_params


def test_build_prompt_keeps_current_question_and_drops_history_to_fit():
    turns = [("old question", "old answer"), ("recent question", "recent answer")]
    prompt = build_prompt(turns, "current question", lambda s: len(s.split()), 8)
    assert "Pergunta: current question" in prompt
    assert prompt.endswith("Resposta:")
    assert len(prompt.split()) <= 8


def test_clean_response_preserves_legitimate_multiple_paragraphs():
    text = "Primeiro parágrafo.\n\nSegundo parágrafo com detalhes."
    assert clean_response(text) == text


def test_clean_response_stops_before_next_training_question():
    text = "Resposta válida.\n\nPergunta: inventada\nResposta: continuação"
    assert clean_response(text) == "Resposta válida."


def test_blank_line_is_not_itself_a_stop_sequence():
    assert "\n\n" not in CHAT_STOP_SEQUENCES


def test_stop_function_detects_marker_across_token_boundaries_incrementally():
    pieces = {1: "Olá\n", 2: "Per", 3: "gunta:", 4: " ignorar"}
    calls = []

    def decode(ids):
        calls.append(tuple(ids))
        return "".join(pieces[i] for i in ids)

    stop = make_stop_fn(decode, ("\nPergunta:",))
    assert not stop([1])
    assert not stop([1, 2])
    assert stop([1, 2, 3])
    # Cada novo token foi decodificado isoladamente, sem redecodificar o prefixo.
    assert calls == [(1,), (2,), (3,)]


@pytest.mark.parametrize("body", [
    {"message": "oi", "temperature": 0},
    {"message": "oi", "temperature": float("nan")},
    {"message": "oi", "temperature": float("inf")},
    {"message": "oi", "max_tokens": 0},
    {"message": "oi", "max_tokens": 401},
    {"message": "oi", "top_k": 0},
    {"message": "oi", "top_k": -2},
    {"message": "oi", "top_k": 1.5},
    {"message": "oi", "history": "not a list"},
])
def test_chat_parameters_reject_invalid_values(body):
    with pytest.raises(ApiError):
        _chat_params(body)


def test_chat_parameters_accept_defaults_and_zero_is_not_silently_defaulted():
    message, ckpt, max_tokens, temperature, top_k, history = _chat_params({"message": " oi ", "temperature": None, "top_k": None})
    assert message == "oi"
    assert ckpt is None
    assert max_tokens == 80
    assert math.isfinite(temperature) and temperature > 0
    assert top_k is None
    assert history == []


def test_chat_parameters_reject_malformed_history_pair():
    with pytest.raises(ApiError):
        _chat_params({"message": "oi", "history": [["só pergunta"]]})


def test_model_generation_calls_stream_callback_for_each_generated_token():
    torch = pytest.importorskip("torch")
    from src.model import GPT

    model = GPT({"n_layer": 1, "n_embd": 16, "n_head": 1, "block_size": 8, "vocab_size": 16})
    model.eval()
    observed = []
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    output = model.generate(prompt, max_new_tokens=3, temperature=1.0, token_callback=lambda ids: observed.append(list(ids)))
    assert output.shape[1] == 5
    assert len(observed) == 3
    assert [len(ids) for ids in observed] == [1, 2, 3]


def test_model_generation_reports_top_five_real_token_probabilities():
    torch = pytest.importorskip("torch")
    from src.model import GPT

    model = GPT({"n_layer": 1, "n_embd": 16, "n_head": 1, "block_size": 8, "vocab_size": 16})
    model.eval()
    reports = []
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    model.generate(
        prompt,
        max_new_tokens=2,
        temperature=1.0,
        diagnostics_callback=reports.append,
    )
    assert len(reports) == 2
    for report in reports:
        candidates = report["top_candidates"]
        assert len(candidates) == 5
        assert all(0 <= token_id < 16 and 0.0 <= probability <= 1.0 for token_id, probability in candidates)
        assert candidates == sorted(candidates, key=lambda item: item[1], reverse=True)
        assert 0 <= report["sampled_token_id"] < 16
        assert 0.0 <= report["sampled_probability"] <= 1.0


def test_top_k_one_always_selects_highest_logit_token():
    torch = pytest.importorskip("torch")
    from src.model import GPT

    model = GPT({"n_layer": 1, "n_embd": 16, "n_head": 1, "block_size": 8, "vocab_size": 16})
    model.eval()
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    with torch.no_grad():
        expected = int(model(prompt)[0][-1].argmax().item())
    # Com top_k=1, multinomial tem apenas um token com probabilidade não nula.
    out = model.generate(prompt, max_new_tokens=1, temperature=0.37, top_k=1)
    assert int(out[0, -1].item()) == expected


def test_greedy_and_diagnostics_expose_all_filter_stages():
    torch = pytest.importorskip("torch")
    from src.model import GPT

    model = GPT({"n_layer": 1, "n_embd": 16, "n_head": 1, "block_size": 8, "vocab_size": 16})
    model.eval()
    reports = []
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    out = model.generate(prompt, max_new_tokens=1, temperature=0.5, top_k=4, greedy=True,
                         repetition_penalty=1.2, diagnostics_callback=reports.append)
    report = reports[0]
    assert int(out[0, -1]) == report["sampled_token_id"]
    assert report["greedy"] is True
    assert len(report["top5_raw_logits"]) == 5
    assert len(report["top5_after_temperature"]) == 5
    assert len(report["top5_after_top_k"]) == 5
    assert report["repetition_penalty"] == 1.2
