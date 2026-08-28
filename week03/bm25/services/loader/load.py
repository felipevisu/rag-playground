#!/usr/bin/env python3
"""Load the frozen dataset into Postgres. No embedding step. Then exit.

This is the lexical arm of the week03 comparison. Same corpus, same chunk_id,
same ground truth as `../new-model` -- the only difference is that nothing here
produces a vector. Retrieval happens at query time in `document-api`, with BM25
over the raw text, so there is no index to build ahead of time and no model to
download.

The load stays idempotent, keyed on (corpus, retriever). Change either and it
wipes and reloads.

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

DATABASE_URL = os.environ["DATABASE_URL"]
DATASET_DIR = Path(os.environ.get("DATASET_DIR", "/dataset"))
# Recorded in dataset_load the way the model name is recorded in the other
# stacks: it is what makes a reload necessary when the retrieval side changes.
RETRIEVER = "bm25-okapi"


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
            psycopg2.connect(DATABASE_URL).close()
            return
        except psycopg2.OperationalError as e:
            print(f"DB not ready (attempt {attempt}/{retries}): {e}")
            time.sleep(delay)
    sys.exit("Postgres never came up.")


def init_schema() -> None:
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id        SERIAL PRIMARY KEY,
                filename  TEXT UNIQUE NOT NULL,
                loaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        # chunk_id is the natural key from the dataset (PL_1502_2026::0003), not
        # a SERIAL. Keeping it identical across stacks is what lets week02's
        # ground truth score this one without a single edit.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id     TEXT PRIMARY KEY,
                document_id  INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index  INTEGER NOT NULL,
                text         TEXT NOT NULL,
                headings     JSONB,
                page_numbers JSONB
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS dataset_load (
                singleton  BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
                corpus_sha TEXT NOT NULL,
                retriever  TEXT NOT NULL,
                loaded_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)")
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
        cur.execute("SELECT corpus_sha, retriever FROM dataset_load")
        current = cur.fetchone()
    if not force and current == (sha, RETRIEVER):
        print(f"Already loaded: corpus {sha[:12]} · {RETRIEVER}. Nothing to do.")
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
        cur.execute(
            """INSERT INTO dataset_load (corpus_sha, retriever) VALUES (%s, %s)
               ON CONFLICT (singleton) DO UPDATE
               SET corpus_sha = EXCLUDED.corpus_sha,
                   retriever = EXCLUDED.retriever,
                   loaded_at = NOW()""",
            (sha, RETRIEVER),
        )
        conn.commit()

    print(f"Inserted {len(doc_ids)} documents · {len(rows)} chunks")
    print(f"Done. corpus {sha[:12]} · {RETRIEVER}")
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
