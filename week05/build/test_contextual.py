"""Contextual prefixes and the answer cache, with a fake LLM. Run: python test_contextual.py"""

import tempfile
from pathlib import Path

import os

from contextual import add_context, load_env

calls = []


def fake(doc, prompt):
    calls.append(prompt)
    return "PL sobre clima." if "<trecho>" not in prompt else f"trecho {len(calls)}"


def corpus():
    return [{"filename": "a.pdf", "text": "Art. 1º Fica instituída.", "n_chars": 24},
            {"filename": "a.pdf", "text": "Art. 2º Para os fins.", "n_chars": 21}]


with tempfile.TemporaryDirectory() as d:
    cache = Path(d) / "c.json"
    rows = add_context(corpus(), {"a.pdf": "Art. 1º ... Art. 2º ..."}, "m", fake, cache)
    assert len(calls) == 3, calls  # one per document + one per chunk
    assert rows[0]["doc_summary"] == "PL sobre clima." and rows[1]["chunk_summary"].startswith("trecho")
    assert rows[0]["text"].startswith("Documento: PL sobre clima.\nTrecho: ")
    assert rows[0]["text"].endswith("\n\nArt. 1º Fica instituída.")
    assert rows[0]["n_chars"] == len(rows[0]["text"])

    again = add_context(corpus(), {"a.pdf": "Art. 1º ... Art. 2º ..."}, "m", fake, cache)
    assert len(calls) == 3, "second build must come from the cache"
    assert [r["text"] for r in again] == [r["text"] for r in rows]

    add_context(corpus(), {"a.pdf": "Art. 1º ... Art. 2º ..."}, "other-model", fake, cache)
    assert len(calls) == 6, "a different model is a different cache entry"

    env = Path(d) / ".env"
    env.write_text('# comment\nCTX_A="x=1"\nCTX_B=\nCTX_C=shell-loses\n')
    os.environ["CTX_C"] = "shell"
    load_env(env)
    assert os.environ["CTX_A"] == "x=1" and "CTX_B" not in os.environ and os.environ["CTX_C"] == "shell"

print("ok")
