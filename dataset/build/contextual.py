"""Contextual retrieval: prefix every chunk with one line about its document
and one about the chunk itself, both written by Claude, before it is indexed.

    Documento: <what the document is>
    Trecho: <where the chunk sits in it and what it says>

    <chunk text>

Answers are cached in cache/contextual.json, keyed by (model, prompts,
document, chunk). Rebuilds only pay for what changed, and the cache is what
keeps out/ reproducible: commit it.
"""

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache" / "contextual.json"
WORKERS = 8

SYSTEM = (
    "Você escreve contexto para um índice de busca de documentos legislativos "
    "brasileiros. Responda em português, com uma única frase curta, sem preâmbulo."
)
DOC_PROMPT = "Em no máximo 25 palavras: que documento é este (tipo, número, autor) e qual o tema."
CHUNK_PROMPT = (
    "Trecho do documento acima:\n<trecho>\n{chunk}\n</trecho>\n\n"
    "Em no máximo 30 palavras: onde este trecho se situa no documento e do que ele trata."
)


def load_env(path: Path) -> None:
    """KEY=value lines into os.environ; the shell wins over the file."""
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and not key.lstrip().startswith("#") and value.strip():
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def claude(model: str) -> Callable[[str, str], str]:
    """ask(document, prompt) -> answer. The document goes first and is cached,
    so the per-chunk calls of one document reuse it."""
    import anthropic

    load_env(ROOT / ".env")
    client = anthropic.Anthropic()

    def ask(doc: str, prompt: str) -> str:
        r = client.beta.messages.create(
            model=model,
            max_tokens=4000,  # thinking counts against this; the answer is one line
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",  # a refused request is re-run on the fallback model
            output_config={"effort": "low"},
            system=SYSTEM,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": f"<documento>\n{doc}\n</documento>",
                 "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": prompt},
            ]}],
        )
        if r.stop_reason != "end_turn":
            raise RuntimeError(f"{model} stopped with {r.stop_reason}: {prompt[:80]!r}")
        return " ".join(b.text for b in r.content if b.type == "text").strip()

    return ask


def add_context(corpus: list[dict], docs: dict[str, str], model: str,
                ask: Callable[[str, str], str] | None = None, cache_path: Path = CACHE) -> list[dict]:
    """Fill doc_summary / chunk_summary on every row and prefix both to `text`.

    docs: filename -> the parse_pdf() text the chunks were cut from."""
    ask = ask or claude(model)
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

    def cached(doc: str, prompt: str) -> str:
        key = hashlib.sha256(json.dumps([model, SYSTEM, doc, prompt]).encode()).hexdigest()
        if key not in cache:
            cache[key] = ask(doc, prompt)
        return cache[key]

    files = sorted({r["filename"] for r in corpus})
    try:
        with ThreadPoolExecutor(WORKERS) as pool:
            # documents first: their prefix is then warm in the prompt cache for the chunks
            doc_sum = dict(zip(files, pool.map(lambda f: cached(docs[f], DOC_PROMPT), files)))
            chunk_sum = list(pool.map(
                lambda r: cached(docs[r["filename"]], CHUNK_PROMPT.format(chunk=r["text"])), corpus))
    finally:  # keep what was paid for, even if a call failed halfway
        cache_path.parent.mkdir(exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                              encoding="utf-8")

    for r, summary in zip(corpus, chunk_sum):
        r["doc_summary"] = doc_sum[r["filename"]]
        r["chunk_summary"] = summary
        r["text"] = f"Documento: {r['doc_summary']}\nTrecho: {summary}\n\n{r['text']}"
        r["n_chars"] = len(r["text"])
    return corpus
