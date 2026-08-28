"""Document API for the BM25-only stack.

Same endpoints and same response shape as `../../../new-model`, so the eval and
both UIs work against it unchanged. The only thing that differs is how
`/api/search` ranks: BM25 over the raw text instead of cosine over vectors.

There is no vector column and no model. The index is a list of tokenized chunks
held in memory, rebuilt at boot from Postgres -- 183 chunks, so "rebuilt" costs
milliseconds and there is nothing to persist.

  python main.py --self-check   # ranking asserts, no DB and no server
"""

import os
import re
import sys
import time
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

DATABASE_URL = os.environ["DATABASE_URL"]
# Recorded in the eval run files under `embedding_model`, where the other stacks
# write their sentence-transformers id. Spelled out rather than left null so a
# run row says what it was, not what it was missing.
RETRIEVER = "none (BM25 léxico)"

_index = None  # (BM25Okapi, [chunk_id, ...])


def tokenize(text: str) -> list[str]:
    r"""Lowercase word tokens.

    `\w+` keeps digits and accented letters, which is what matters here: the
    high-signal terms in this corpus are things like `fnarc`, `1502`, `2026` and
    `3º`, and they have to tokenize identically on both sides of the query.

    ponytail: no stemmer and no stopword list. BM25's IDF already discounts
    `de`/`do`/`que` to near zero, and a stemmer is what `week03/bm25` would try
    next (nltk RSLP, or the `portuguese` analyzer if this moves to OpenSearch)
    if recall on paraphrased queries turns out to be the ceiling.
    """
    return re.findall(r"\w+", text.lower())


@contextmanager
def db():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        conn.close()


def wait_for_db(retries: int = 15, delay: int = 2) -> None:
    for attempt in range(1, retries + 1):
        try:
            with db():
                pass
            return
        except psycopg2.OperationalError:
            print(f"DB not ready (attempt {attempt}/{retries})")
            time.sleep(delay)


def build_index() -> None:
    """Read every chunk once and tokenize it. Called at boot, not per request."""
    global _index
    from rank_bm25 import BM25Okapi

    with db() as conn, conn.cursor() as cur:
        cur.execute("SELECT chunk_id, text FROM chunks ORDER BY chunk_id")
        rows = cur.fetchall()
    if not rows:
        print("No chunks in Postgres — did the loader run?")
        return
    # Default k1=1.5, b=0.75. Left alone on purpose: this run is the untuned
    # lexical baseline the vector stacks get compared against.
    _index = (BM25Okapi([tokenize(t) for _, t in rows]), [cid for cid, _ in rows])
    print(f"BM25 index ready: {len(rows)} chunks.")


def get_index():
    if _index is None:
        build_index()
    if _index is None:
        raise HTTPException(status_code=503, detail="Index empty: no chunks loaded")
    return _index


app = FastAPI(on_startup=[wait_for_db, build_index])
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "embedding_model": RETRIEVER,
        "retriever": "bm25-okapi",
        "indexed_chunks": len(_index[1]) if _index else 0,
    }


@app.get("/api/documents")
def list_documents():
    with db() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT d.id, d.filename, d.loaded_at,
                   COUNT(c.chunk_id)::int AS chunk_count
            FROM documents d
            LEFT JOIN chunks c ON c.document_id = d.id
            GROUP BY d.id
            ORDER BY d.filename
        """)
        return list(cur.fetchall())


@app.get("/api/documents/{doc_id}/chunks")
def get_chunks(doc_id: int):
    with db() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT id FROM documents WHERE id = %s", (doc_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="Document not found")
        cur.execute("""
            SELECT chunk_id AS id, chunk_index, text, headings, page_numbers
            FROM chunks WHERE document_id = %s ORDER BY chunk_index
        """, (doc_id,))
        # No `embedding` key at all: the UI's embedding panel already renders
        # "No embedding" when the field is missing, which is the truth here.
        return list(cur.fetchall())


def rank(query: str, k: int) -> list[tuple[str, float]]:
    """Top-k (chunk_id, score), best first. Zero-scoring chunks are dropped."""
    bm25, ids = get_index()
    tokens = tokenize(query)
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)
    order = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)
    # A chunk sharing no term with the query scores 0 and is not a result. The
    # vector stacks always return k rows because cosine is defined everywhere;
    # BM25 can honestly return fewer, and precision should get the credit.
    return [(ids[i], float(scores[i])) for i in order[:k] if scores[i] > 0]


@app.get("/api/search")
def search(
    q: str = Query(..., min_length=1, description="Query text"),
    k: int = Query(5, ge=1, le=50, description="Top-K results"),
):
    ranked = rank(q, k)
    if not ranked:
        return {"query": q, "k": k, "results": []}

    ids = [cid for cid, _ in ranked]
    with db() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT c.chunk_id AS id, c.chunk_index, c.text, c.headings, c.page_numbers,
                   d.id AS document_id, d.filename
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.chunk_id = ANY(%s)
            """,
            (ids,),
        )
        by_id = {r["id"]: r for r in cur.fetchall()}

    # `similarity` is the field name the eval and both UIs read. Here it carries
    # a raw BM25 score, which is unbounded and not comparable across queries —
    # only the ordering within one query means anything.
    results = [{**by_id[cid], "similarity": score} for cid, score in ranked if cid in by_id]
    return {"query": q, "k": k, "results": results}


def self_check() -> int:
    from rank_bm25 import BM25Okapi

    assert tokenize("Art. 3º cria o FNARC — R$ 1.502/2026") == [
        "art", "3º", "cria", "o", "fnarc", "r", "1", "502", "2026"
    ], tokenize("Art. 3º cria o FNARC — R$ 1.502/2026")

    docs = [
        "Fica criado o Fundo Nacional de Adaptação e Reparação Climática (FNARC).",
        "Esta lei institui a política de resíduos sólidos urbanos.",
        "O Comitê Gestor do FNARC será composto por representantes dos Ministérios.",
    ]
    global _index
    _index = (BM25Okapi([tokenize(d) for d in docs]), ["a", "b", "c"])

    # the rare term wins, and the doc that never mentions it is not a result
    top = rank("FNARC", k=3)
    assert [cid for cid, _ in top] == ["a", "c"], top
    assert all(s > 0 for _, s in top)

    # a query sharing nothing with the corpus returns nothing, rather than
    # padding the result list with k arbitrary rows
    assert rank("bicicleta", k=3) == []
    assert rank("!!!", k=3) == []

    # k truncates
    assert len(rank("FNARC lei o", k=1)) == 1

    _index = None
    print("self-check ok")
    return 0


if __name__ == "__main__":
    sys.exit(self_check() if "--self-check" in sys.argv else 0)
