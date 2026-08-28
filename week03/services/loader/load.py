#!/usr/bin/env python3
"""Load the frozen dataset into Postgres, embed it, index it. Then exit.

week01 does not manage files any more. The corpus is split once in `dataset/`
and committed there as parquet; this reads those chunks, embeds them and writes
them to Postgres. No PDFs, no object store, no docling.

week03 swaps the embedding model for a multilingual one — the corpus is in
Portuguese and the old model was not.

The load is idempotent: it records which (corpus, embedding model) pair is
currently in the database and does nothing when they already match. Change
either one and it wipes and reloads, because a half-swapped index is worse than
a slow boot.

  python load.py               # load if needed
  python load.py --force       # wipe and reload regardless
  python load.py --self-check  # row coercion asserts, no DB needed
"""

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import psycopg2
from pgvector.psycopg2 import register_vector

DATABASE_URL = os.environ["DATABASE_URL"]
DATASET_DIR = Path(os.environ.get("DATASET_DIR", "/dataset"))
# multilingual-e5, trained on 100 languages including pt-BR. all-MiniLM-L6-v2
# is English-only: it embedded "Art. 3o Fica criado o FNARC" as noise.
# e5 wants its inputs prefixed by role — "passage: " for what is indexed,
# "query: " for what is searched. Without the prefixes the vectors land in
# the wrong region of the space and retrieval quietly degrades.
MODEL_NAME = "intfloat/multilingual-e5-base"
EMBED_DIM = 768
BATCH = 64

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(MODEL_NAME)
    return _embedder


@contextmanager
def db():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        register_vector(conn)
        yield conn
    finally:
        conn.close()


def wait_for_db(retries: int = 15, delay: int = 2) -> None:
    for attempt in range(1, retries + 1):
        try:
            psycopg2.connect(DATABASE_URL).close()
            return
        except psycopg2.OperationalError as e:
            print(f"DB not ready (attempt {attempt}/{retries}): {e}")
            time.sleep(delay)
    sys.exit("Postgres never came up.")


def init_schema() -> None:
    raw = psycopg2.connect(DATABASE_URL)
    with raw.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        raw.commit()
    raw.close()

    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id        SERIAL PRIMARY KEY,
                filename  TEXT UNIQUE NOT NULL,
                loaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        # chunk_id is the natural key from the dataset (PL_1502_2026::0003), not
        # a SERIAL. A SERIAL changes on every reload, which is exactly what made
        # the old ground truth rot.
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id     TEXT PRIMARY KEY,
                document_id  INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index  INTEGER NOT NULL,
                text         TEXT NOT NULL,
                headings     JSONB,
                page_numbers JSONB,
                embedding    VECTOR({EMBED_DIM})
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS dataset_load (
                singleton   BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
                corpus_sha  TEXT NOT NULL,
                embed_model TEXT NOT NULL,
                loaded_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)")
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
            ON chunks USING hnsw (embedding vector_cosine_ops)
        """)
        conn.commit()


def read_corpus() -> list[dict]:
    """Chunks from corpus.parquet, with list columns coerced back to JSON."""
    import pandas as pd

    path = DATASET_DIR / "corpus.parquet"
    if not path.exists():
        sys.exit(
            f"{path} not found. Build the dataset first:\n"
            f"  cd dataset && docker compose run --rm build"
        )
    return [coerce(r) for r in pd.read_parquet(path).to_dict("records")]


def coerce(row: dict) -> dict:
    """parquet hands list columns back as numpy arrays, which json.dumps rejects."""
    return {
        "chunk_id": str(row["chunk_id"]),
        "filename": str(row["filename"]),
        "chunk_index": int(row["chunk_index"]),
        "text": str(row["text"]),
        "headings": [str(h) for h in row["headings"]],
        "pages": [int(p) for p in row["pages"]],
    }


def load(force: bool = False) -> int:
    wait_for_db()
    init_schema()

    manifest = json.loads((DATASET_DIR / "manifest.json").read_text(encoding="utf-8"))
    sha = manifest["corpus_sha256"]

    with db() as conn, conn.cursor() as cur:
        cur.execute("SELECT corpus_sha, embed_model FROM dataset_load")
        current = cur.fetchone()
    if not force and current == (sha, MODEL_NAME):
        print(f"Already loaded: corpus {sha[:12]} · {MODEL_NAME}. Nothing to do.")
        return 0
    if current:
        print(f"Loaded corpus {current[0][:12]} · {current[1]} — replacing.")

    rows = read_corpus()
    print(f"Loading {len(rows)} chunks from corpus.parquet")

    with db() as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE documents RESTART IDENTITY CASCADE")
        doc_ids: dict[str, int] = {}
        for r in rows:
            if r["filename"] not in doc_ids:
                cur.execute(
                    "INSERT INTO documents (filename) VALUES (%s) RETURNING id",
                    (r["filename"],),
                )
                doc_ids[r["filename"]] = cur.fetchone()[0]
            cur.execute(
                """INSERT INTO chunks
                       (chunk_id, document_id, chunk_index, text, headings, page_numbers)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (
                    r["chunk_id"], doc_ids[r["filename"]], r["chunk_index"], r["text"],
                    json.dumps(r["headings"]), json.dumps(r["pages"]),
                ),
            )
        conn.commit()
    print(f"Inserted {len(doc_ids)} documents · {len(rows)} chunks")

    print(f"Embedding with {MODEL_NAME}…")
    embedder = get_embedder()
    for i in range(0, len(rows), BATCH):
        batch = rows[i:i + BATCH]
        vectors = embedder.encode(
            [f"passage: {r['text']}" for r in batch], normalize_embeddings=True
        )
        with db() as conn, conn.cursor() as cur:
            for r, vec in zip(batch, vectors):
                cur.execute(
                    "UPDATE chunks SET embedding = %s WHERE chunk_id = %s",
                    (vec.tolist(), r["chunk_id"]),
                )
            conn.commit()
        print(f"  {min(i + BATCH, len(rows))}/{len(rows)}")

    with db() as conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO dataset_load (corpus_sha, embed_model) VALUES (%s, %s)
               ON CONFLICT (singleton) DO UPDATE
               SET corpus_sha = EXCLUDED.corpus_sha,
                   embed_model = EXCLUDED.embed_model,
                   loaded_at = NOW()""",
            (sha, MODEL_NAME),
        )
        conn.commit()

    print(f"Done. corpus {sha[:12]} · {MODEL_NAME}")
    return 0


def self_check() -> int:
    import numpy as np

    row = {
        "chunk_id": "PL_1502_2026::0003",
        "filename": "PL_1502_2026.pdf",
        "chunk_index": np.int64(3),
        "text": "Art. 3o Fica criado o FNARC.",
        "headings": np.array(["Projeto de Lei"], dtype=object),
        "pages": np.array([3, 4]),
    }
    c = coerce(row)
    # json.dumps rejects numpy scalars and arrays — the coercion is the whole point
    assert json.dumps(c["pages"]) == "[3, 4]"
    assert json.dumps(c["headings"]) == '["Projeto de Lei"]'
    assert isinstance(c["chunk_index"], int) and c["chunk_index"] == 3
    # empty list columns survive
    empty = coerce({**row, "headings": np.array([], dtype=object), "pages": np.array([])})
    assert json.dumps(empty["headings"]) == "[]"
    assert json.dumps(empty["pages"]) == "[]"
    print("self-check ok")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="wipe and reload regardless")
    ap.add_argument("--self-check", action="store_true", help="row coercion only, no DB")
    args = ap.parse_args()
    sys.exit(self_check() if args.self_check else load(args.force))
