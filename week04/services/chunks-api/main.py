"""Chunks API: buckets.

bucket = one corpus.parquet + its answers.parquet (the gabarito) + one
retriever. Create it, upload into it, search it at /api/buckets/{id}/search,
read its gabarito at /api/buckets/{id}/answers. To compare retrievers, make two
buckets from the same pair of files.

/api/search keeps the response shape of week01/week03 so `eval/` runs
unchanged; without `bucket=` it uses the most recently indexed bucket.
"""

import hashlib
import io
import os
import re
import threading
import time
import uuid
from contextlib import contextmanager

import pandas as pd
import psycopg2
import psycopg2.extras
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pgvector.psycopg2 import register_vector
from psycopg2.extras import Json, execute_values
from pydantic import BaseModel

DATABASE_URL = os.environ["DATABASE_URL"]
BATCH = 64

# One line per retriever. e5 wants its role prefixes or the vectors land in the
# wrong region of the space.
RETRIEVERS = {
    "minilm": {"kind": "vector", "model": "sentence-transformers/all-MiniLM-L6-v2",
               "dim": 384, "passage": "", "query": ""},
    "e5-base": {"kind": "vector", "model": "intfloat/multilingual-e5-base",
                "dim": 768, "passage": "passage: ", "query": "query: "},
    "e5-large": {"kind": "vector", "model": "intfloat/multilingual-e5-large",
                 "dim": 1024, "passage": "passage: ", "query": "query: "},
    # bge-m3: symmetric, no prefixes. Qwen3: instruction on the query side only,
    # last-token pooling comes from the model config via sentence-transformers.
    "bge-m3": {"kind": "vector", "model": "BAAI/bge-m3", "dim": 1024, "passage": "", "query": ""},
    "qwen3-0.6b": {"kind": "vector", "model": "Qwen/Qwen3-Embedding-0.6B", "dim": 1024, "passage": "",
                   "query": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "},
    "bm25": {"kind": "bm25", "model": "rank-bm25 (BM25Okapi)"},
}
REQUIRED_COLUMNS = {"chunk_id", "filename", "chunk_index", "pages", "text"}
ANSWER_COLUMNS = {"query_id", "chunk_id"}

_embedders: dict[str, object] = {}
_embed_lock = threading.Lock()
_bm25: dict[str, tuple] = {}  # bucket_id -> (BM25Okapi, [chunk_id])


# ── infra ──────────────────────────────────────────────────────────────────

@contextmanager
def db():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        register_vector(conn)
        yield conn
    finally:
        conn.close()


def cursor(conn):
    return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)


def wait_for_db(retries: int = 15, delay: int = 2) -> None:
    for attempt in range(1, retries + 1):
        try:
            psycopg2.connect(DATABASE_URL).close()
            return
        except psycopg2.OperationalError:
            print(f"DB not ready (attempt {attempt}/{retries})")
            time.sleep(delay)


def init_schema() -> None:
    raw = psycopg2.connect(DATABASE_URL)
    with raw, raw.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    raw.close()
    with db() as conn, conn.cursor() as cur:
        # A pgdata volume from an earlier layout (corpora/indexes, chunks.corpus_id)
        # would survive CREATE IF NOT EXISTS. Nothing there is worth keeping: drop it.
        cur.execute("""
            DO $$ BEGIN
              IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'chunks')
                 AND NOT EXISTS (SELECT 1 FROM information_schema.columns
                                 WHERE table_name = 'chunks' AND column_name = 'bucket_id') THEN
                DROP TABLE IF EXISTS vectors, chunks, indexes, corpora CASCADE;
              END IF;
            END $$;
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS buckets (
                id             TEXT PRIMARY KEY,
                name           TEXT NOT NULL,
                created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                retriever      TEXT,
                status         TEXT NOT NULL DEFAULT 'empty',   -- empty | indexing | ready | error
                chunk_count    INTEGER NOT NULL DEFAULT 0,
                embedded_count INTEGER NOT NULL DEFAULT 0,
                parquet_sha    TEXT,
                error          TEXT,
                indexed_at     TIMESTAMPTZ
            );
            -- VECTOR without a dimension: buckets with 384d and 768d models share
            -- the table. Exact scan filtered by bucket_id; ponytail: HNSW per dim
            -- if a bucket grows 100x.
            CREATE TABLE IF NOT EXISTS chunks (
                bucket_id   TEXT NOT NULL REFERENCES buckets(id) ON DELETE CASCADE,
                chunk_id    TEXT NOT NULL,
                filename    TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                pages       JSONB,
                heading     TEXT,
                text        TEXT NOT NULL,
                embedding   VECTOR,
                PRIMARY KEY (bucket_id, chunk_id)
            );
            -- The gabarito travels with the corpus it was resolved against: a
            -- chunk_id only means something inside its own bucket.
            CREATE TABLE IF NOT EXISTS qrels (
                bucket_id TEXT NOT NULL REFERENCES buckets(id) ON DELETE CASCADE,
                query_id  TEXT NOT NULL,
                chunk_id  TEXT NOT NULL,
                part      INTEGER NOT NULL DEFAULT 0,
                coverage  REAL,
                PRIMARY KEY (bucket_id, query_id, chunk_id, part)
            );
        """)
        conn.commit()


def embedder(name: str):
    with _embed_lock:
        if name not in _embedders:
            from sentence_transformers import SentenceTransformer
            print(f"Loading {RETRIEVERS[name]['model']}…")
            _embedders[name] = SentenceTransformer(RETRIEVERS[name]["model"])
        return _embedders[name]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


app = FastAPI(on_startup=[wait_for_db, init_schema])
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── buckets ────────────────────────────────────────────────────────────────

class NewBucket(BaseModel):
    name: str


BUCKET_SQL = """
    SELECT b.*,
           (SELECT COUNT(*)::int FROM qrels q WHERE q.bucket_id = b.id) AS qrels_count,
           (SELECT COUNT(DISTINCT query_id)::int FROM qrels q WHERE q.bucket_id = b.id) AS queries_count
    FROM buckets b
"""


def present(b: dict) -> dict:
    return {**b, "search_url": f"/api/buckets/{b['id']}/search",
            "answers_url": f"/api/buckets/{b['id']}/answers"}


def get_bucket(cur, bid: str) -> dict:
    cur.execute(BUCKET_SQL + " WHERE b.id = %s", (bid,))
    b = cur.fetchone()
    if not b:
        raise HTTPException(404, "bucket not found")
    return b


def read_parquet(upload_bytes: bytes, what: str, required: set[str]):
    try:
        df = pd.read_parquet(io.BytesIO(upload_bytes))
    except Exception as e:
        raise HTTPException(400, f"{what}: not a readable parquet: {e}")
    missing = required - set(df.columns)
    if missing:
        raise HTTPException(400, f"{what}: missing columns {sorted(missing)}")
    return df


@app.get("/api/buckets")
def list_buckets():
    with db() as conn, cursor(conn) as cur:
        cur.execute(BUCKET_SQL + " ORDER BY b.created_at DESC")
        return [present(b) for b in cur.fetchall()]


@app.post("/api/buckets", status_code=201)
def create_bucket(body: NewBucket):
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "name is required")
    with db() as conn, cursor(conn) as cur:
        cur.execute("INSERT INTO buckets (id, name) VALUES (%s, %s) RETURNING id",
                    (uuid.uuid4().hex[:8], name))
        bid = cur.fetchone()["id"]
        conn.commit()
        return present(get_bucket(cur, bid))


@app.get("/api/buckets/{bid}")
def read_bucket(bid: str):
    with db() as conn, cursor(conn) as cur:
        return present(get_bucket(cur, bid))


@app.delete("/api/buckets/{bid}", status_code=204)
def delete_bucket(bid: str):
    _bm25.pop(bid, None)
    with db() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM buckets WHERE id = %s", (bid,))
        if cur.rowcount == 0:
            raise HTTPException(404, "bucket not found")
        conn.commit()


# ── upload + indexing ──────────────────────────────────────────────────────

def embed_bucket(bid: str, retriever: str) -> None:
    spec = RETRIEVERS[retriever]
    try:
        with db() as conn, conn.cursor() as cur:
            cur.execute("SELECT chunk_id, text FROM chunks WHERE bucket_id = %s ORDER BY chunk_id", (bid,))
            rows = cur.fetchall()
        model = embedder(retriever)
        for i in range(0, len(rows), BATCH):
            batch = rows[i:i + BATCH]
            vecs = model.encode([spec["passage"] + t for _, t in batch], normalize_embeddings=True)
            with db() as conn, conn.cursor() as cur:
                # execute_values allows exactly one %s, so bucket_id rides inside each row.
                execute_values(
                    cur,
                    "UPDATE chunks SET embedding = v.emb::vector FROM (VALUES %s) AS v(bid, cid, emb) "
                    "WHERE chunks.bucket_id = v.bid AND chunks.chunk_id = v.cid",
                    [(bid, cid, v.tolist()) for (cid, _), v in zip(batch, vecs)],
                    template="(%s, %s, %s::real[])",
                )
                cur.execute("UPDATE buckets SET embedded_count = %s WHERE id = %s", (i + len(batch), bid))
                conn.commit()
        with db() as conn, conn.cursor() as cur:
            cur.execute("UPDATE buckets SET status = 'ready', indexed_at = NOW() WHERE id = %s", (bid,))
            conn.commit()
        print(f"bucket {bid} ready: {len(rows)} vectors ({retriever})")
    except Exception as e:  # keep the row so the UI can show what went wrong
        print(f"bucket {bid} failed: {e}")
        with db() as conn, conn.cursor() as cur:
            cur.execute("UPDATE buckets SET status = 'error', error = %s WHERE id = %s", (str(e)[:500], bid))
            conn.commit()


@app.post("/api/buckets/{bid}/upload")
async def upload(bid: str, bg: BackgroundTasks, file: UploadFile = File(...),
                 answers: UploadFile = File(...), retriever: str = Form(...)):
    if retriever not in RETRIEVERS:
        raise HTTPException(400, f"unknown retriever; choose one of {list(RETRIEVERS)}")
    raw = await file.read()
    df = read_parquet(raw, "corpus", REQUIRED_COLUMNS)
    if "heading" not in df.columns:
        df["heading"] = ""
    rows = [
        (bid, str(r.chunk_id), str(r.filename), int(r.chunk_index),
         Json([int(p) for p in r.pages]), str(r.heading or ""), str(r.text))
        for r in df.itertuples()
    ]

    ans = read_parquet(await answers.read(), "answers", ANSWER_COLUMNS)
    if "part" not in ans.columns:
        ans["part"] = 0
    if "coverage" not in ans.columns:
        ans["coverage"] = None
    known = {r[1] for r in rows}
    unknown = sorted(set(map(str, ans.chunk_id)) - known)
    if unknown:
        raise HTTPException(400, f"answers reference {len(unknown)} chunk_id(s) not in corpus, "
                                 f"e.g. {unknown[:3]} — wrong pair of files?")
    qrels = [(bid, str(a.query_id), str(a.chunk_id), int(a.part),
              None if pd.isna(a.coverage) else float(a.coverage)) for a in ans.itertuples()]

    is_bm25 = RETRIEVERS[retriever]["kind"] == "bm25"
    _bm25.pop(bid, None)

    with db() as conn, cursor(conn) as cur:
        b = get_bucket(cur, bid)
        # A bucket is written once. Its id is the stable handle of one (corpus,
        # retriever); swapping either would silently invalidate every result
        # recorded against it. Want a different corpus? Create another bucket.
        if b["status"] not in ("empty", "error"):
            raise HTTPException(409, f"bucket is {b['status']}: buckets are immutable once indexed, create a new one")
        cur.execute("DELETE FROM chunks WHERE bucket_id = %s", (bid,))  # leftovers from a failed attempt
        cur.execute("DELETE FROM qrels WHERE bucket_id = %s", (bid,))
        execute_values(
            cur,
            "INSERT INTO chunks (bucket_id, chunk_id, filename, chunk_index, pages, heading, text) VALUES %s",
            rows,
        )
        execute_values(
            cur,
            "INSERT INTO qrels (bucket_id, query_id, chunk_id, part, coverage) VALUES %s "
            "ON CONFLICT DO NOTHING",
            qrels,
        )
        cur.execute("""
            UPDATE buckets SET retriever = %s, status = %s, chunk_count = %s, embedded_count = %s,
                   parquet_sha = %s, error = NULL, indexed_at = CASE WHEN %s THEN NOW() END
            WHERE id = %s
        """, (retriever, "ready" if is_bm25 else "indexing", len(rows),
              len(rows) if is_bm25 else 0, hashlib.sha256(raw).hexdigest(), is_bm25, bid))
        conn.commit()
        b = get_bucket(cur, bid)
    if not is_bm25:
        bg.add_task(embed_bucket, bid, retriever)
    return present(b)


# ── browse ─────────────────────────────────────────────────────────────────

@app.get("/api/buckets/{bid}/answers")
def list_answers(bid: str):
    """The gabarito: one row per (query, chunk). Same shape as dataset's answers.parquet."""
    with db() as conn, cursor(conn) as cur:
        get_bucket(cur, bid)
        cur.execute("SELECT query_id, chunk_id, part, coverage FROM qrels WHERE bucket_id = %s "
                    "ORDER BY query_id, part, chunk_id", (bid,))
        return list(cur.fetchall())


@app.get("/api/buckets/{bid}/documents")
def list_documents(bid: str):
    with db() as conn, cursor(conn) as cur:
        get_bucket(cur, bid)
        cur.execute("""
            SELECT filename, COUNT(*)::int AS chunk_count
            FROM chunks WHERE bucket_id = %s GROUP BY filename ORDER BY filename
        """, (bid,))
        return list(cur.fetchall())


@app.get("/api/buckets/{bid}/documents/{filename}/chunks")
def document_chunks(bid: str, filename: str):
    with db() as conn, cursor(conn) as cur:
        cur.execute("""
            SELECT chunk_id AS id, chunk_index, text, heading, pages, embedding
            FROM chunks WHERE bucket_id = %s AND filename = %s ORDER BY chunk_index
        """, (bid, filename))
        rows = cur.fetchall()
        for r in rows:
            if r["embedding"] is not None:
                r["embedding"] = r["embedding"].to_list()  # pgvector Vector, not numpy
        return rows


# ── search ─────────────────────────────────────────────────────────────────

def bm25_index(cur, bid: str):
    if bid not in _bm25:
        from rank_bm25 import BM25Okapi
        cur.execute("SELECT chunk_id, text FROM chunks WHERE bucket_id = %s ORDER BY chunk_id", (bid,))
        rows = cur.fetchall()
        _bm25[bid] = (BM25Okapi([tokenize(r["text"]) for r in rows]), [r["chunk_id"] for r in rows])
    return _bm25[bid]


def rank(cur, b: dict, q: str, k: int) -> list[tuple[str, float]]:
    spec = RETRIEVERS[b["retriever"]]
    if spec["kind"] == "bm25":
        bm25, ids = bm25_index(cur, b["id"])
        tokens = tokenize(q)
        if not tokens:
            return []
        scores = bm25.get_scores(tokens)
        order = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)
        # zero-score chunks share no term with the query: not a result
        return [(ids[i], float(scores[i])) for i in order[:k] if scores[i] > 0]
    vec = embedder(b["retriever"]).encode(spec["query"] + q, normalize_embeddings=True).tolist()
    cur.execute("""
        SELECT chunk_id, (1 - (embedding <=> %s::vector))::float AS similarity
        FROM chunks WHERE bucket_id = %s AND embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector LIMIT %s
    """, (vec, b["id"], vec, k))
    return [(r["chunk_id"], r["similarity"]) for r in cur.fetchall()]


def do_search(cur, b: dict, q: str, k: int) -> dict:
    if b["status"] != "ready":
        raise HTTPException(409, f"bucket is {b['status']}")
    ranked = rank(cur, b, q, k)
    out = {"query": q, "k": k, "bucket": b["id"], "retriever": b["retriever"], "results": []}
    if ranked:
        cur.execute("""
            SELECT chunk_id AS id, chunk_index, text, heading, pages, filename
            FROM chunks WHERE bucket_id = %s AND chunk_id = ANY(%s)
        """, (b["id"], [c for c, _ in ranked]))
        by_id = {r["id"]: r for r in cur.fetchall()}
        out["results"] = [{**by_id[c], "similarity": s} for c, s in ranked if c in by_id]
    return out


@app.get("/api/buckets/{bid}/search")
def bucket_search(bid: str, q: str = Query(..., min_length=1), k: int = Query(5, ge=1, le=50)):
    with db() as conn, cursor(conn) as cur:
        return do_search(cur, get_bucket(cur, bid), q, k)


def default_bucket(cur) -> dict | None:
    cur.execute(BUCKET_SQL + " WHERE b.status = 'ready' ORDER BY b.indexed_at DESC NULLS LAST LIMIT 1")
    return cur.fetchone()


@app.get("/api/search")
def search(q: str = Query(..., min_length=1), k: int = Query(5, ge=1, le=50),
           bucket: str | None = None):
    """Alias for eval/: same shape as week01/week03. Default = last indexed bucket."""
    with db() as conn, cursor(conn) as cur:
        b = get_bucket(cur, bucket) if bucket else default_bucket(cur)
        if not b:
            raise HTTPException(404, "no ready bucket")
        return do_search(cur, b, q, k)


@app.get("/api/health")
def health():
    with db() as conn, cursor(conn) as cur:
        b = default_bucket(cur)
    # `embedding_model` is what eval/ records per run.
    model = f"{RETRIEVERS[b['retriever']]['model']} @ {b['name']} ({b['id']})" if b else None
    return {
        "status": "ok",
        "embedding_model": model,
        "default_bucket": present(b) if b else None,
        "retrievers": {k: v["model"] for k, v in RETRIEVERS.items()},
    }
