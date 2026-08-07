import asyncio
import json
import os
import tempfile
import time
from contextlib import asynccontextmanager, contextmanager
from urllib.parse import unquote

import psycopg2
from fastapi import BackgroundTasks, FastAPI, Request
from minio import Minio
from pgvector.psycopg2 import register_vector

DATABASE_URL = os.environ["DATABASE_URL"]
MINIO_ENDPOINT = os.environ["MINIO_ENDPOINT"]
MINIO_ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
MINIO_SECRET_KEY = os.environ["MINIO_SECRET_KEY"]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)

_converter = None
_chunker = None
_embedder = None
_process_sem = None


def get_sem():
    global _process_sem
    if _process_sem is None:
        import threading
        _process_sem = threading.Semaphore(1)
    return _process_sem


def get_converter():
    global _converter
    if _converter is None:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        opts = PdfPipelineOptions()
        opts.do_ocr = False
        opts.do_table_structure = False
        _converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
    return _converter


def get_chunker():
    global _chunker
    if _chunker is None:
        from docling.chunking import HybridChunker
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _chunker = HybridChunker(tokenizer=tokenizer, max_tokens=512, merge_peers=True)
    return _chunker


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


def init_schema_with_retry(retries: int = 15, delay: int = 2) -> None:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            raw = psycopg2.connect(DATABASE_URL)
            with raw.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                raw.commit()
            raw.close()

            with db() as conn, conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id            SERIAL PRIMARY KEY,
                        filename      TEXT UNIQUE NOT NULL,
                        status        TEXT NOT NULL DEFAULT 'ready',
                        error_message TEXT,
                        processed_at  TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS chunks (
                        id           SERIAL PRIMARY KEY,
                        document_id  INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                        chunk_index  INTEGER NOT NULL,
                        text         TEXT NOT NULL,
                        headings     JSONB,
                        page_numbers JSONB,
                        embedding    VECTOR({EMBED_DIM}),
                        created_at   TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                cur.execute(f"""
                    ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding VECTOR({EMBED_DIM})
                """)
                cur.execute("""
                    ALTER TABLE documents ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'ready'
                """)
                cur.execute("""
                    ALTER TABLE documents ADD COLUMN IF NOT EXISTS error_message TEXT
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)")
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
                    ON chunks USING hnsw (embedding vector_cosine_ops)
                """)
                conn.commit()
            print("Schema ready.")
            return
        except psycopg2.OperationalError as e:
            last_err = e
            print(f"DB not ready (attempt {attempt}/{retries}): {e}")
            time.sleep(delay)
    raise RuntimeError(f"Could not init schema: {last_err}")


def backfill_embeddings() -> None:
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.embedding IS NULL AND d.status = 'ready'
        """)
        pending = cur.fetchone()[0]
    if pending == 0:
        print("No backfill needed.")
        return

    print(f"Backfilling embeddings for {pending} chunks…")
    embedder = get_embedder()
    BATCH = 64

    while True:
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT c.id, c.text FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.embedding IS NULL AND d.status = 'ready'
                LIMIT %s
            """, (BATCH,))
            rows = cur.fetchall()
            if not rows:
                break
            ids = [r[0] for r in rows]
            texts = [r[1] for r in rows]
            vectors = embedder.encode(texts, normalize_embeddings=True)
            for cid, vec in zip(ids, vectors):
                cur.execute(
                    "UPDATE chunks SET embedding = %s WHERE id = %s",
                    (vec.tolist(), cid),
                )
            conn.commit()
            print(f"  backfilled batch of {len(rows)}")
    print("Backfill complete.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_schema_with_retry()
    loop = asyncio.get_event_loop()
    print("Pre-loading docling models…")
    await loop.run_in_executor(None, get_chunker)
    await loop.run_in_executor(None, get_converter)
    print("Pre-loading embedding model…")
    await loop.run_in_executor(None, get_embedder)
    await loop.run_in_executor(None, backfill_embeddings)
    print("Worker ready.")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


def _extract_pages(chunk) -> list:
    pages: set = set()
    try:
        for item in chunk.meta.doc_items:
            for prov in getattr(item, "prov", []):
                p = getattr(prov, "page_no", None)
                if p is not None:
                    pages.add(p)
    except Exception:
        pass
    return sorted(pages)


def process_pdf(bucket: str, key: str) -> None:
    filename = key
    print(f"[queued] {filename} — waiting for slot")
    with get_sem():
        print(f"[start] {filename}")
        doc_id = None

        try:
            with db() as conn, conn.cursor() as cur:
                cur.execute("SELECT id, status FROM documents WHERE filename = %s", (filename,))
                row = cur.fetchone()
                if row:
                    doc_id, status = row
                    if status == "ready":
                        print(f"[skip] {filename} already ready")
                        return
                    print(f"[retry] {filename} was {status}, reprocessing")
                    cur.execute(
                        "UPDATE documents SET status = 'chunking', error_message = NULL WHERE id = %s",
                        (doc_id,),
                    )
                else:
                    cur.execute(
                        "INSERT INTO documents (filename, status) VALUES (%s, 'chunking') RETURNING id",
                        (filename,),
                    )
                    doc_id = cur.fetchone()[0]
                    print(f"[new] {filename} doc_id={doc_id}")
                conn.commit()

            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
            os.close(tmp_fd)

            try:
                print(f"[download] {bucket}/{key}")
                minio_client.fget_object(bucket, key, tmp_path)
                print(f"[convert] {filename} (this may take 1-3 min on CPU)")
                result = get_converter().convert(tmp_path)
                print(f"[chunking] {filename}")
                chunks_data = list(get_chunker().chunk(result.document))
                print(f"[chunked] {filename} -> {len(chunks_data)} chunks")
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

            with db() as conn, conn.cursor() as cur:
                cur.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
                for i, chunk in enumerate(chunks_data):
                    headings = list(chunk.meta.headings) if chunk.meta.headings else []
                    pages = _extract_pages(chunk)
                    cur.execute(
                        """INSERT INTO chunks (document_id, chunk_index, text, headings, page_numbers)
                           VALUES (%s, %s, %s, %s, %s)""",
                        (doc_id, i, chunk.text, json.dumps(headings), json.dumps(pages)),
                    )
                cur.execute("UPDATE documents SET status = 'embedding' WHERE id = %s", (doc_id,))
                conn.commit()
            print(f"[embed] {filename} — generating {len(chunks_data)} embeddings")

            texts = [c.text for c in chunks_data]
            vectors = get_embedder().encode(texts, normalize_embeddings=True)

            with db() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM chunks WHERE document_id = %s ORDER BY chunk_index", (doc_id,)
                )
                chunk_ids = [r[0] for r in cur.fetchall()]
                for cid, vec in zip(chunk_ids, vectors):
                    cur.execute("UPDATE chunks SET embedding = %s WHERE id = %s", (vec.tolist(), cid))
                cur.execute(
                    "UPDATE documents SET status = 'ready', processed_at = NOW() WHERE id = %s", (doc_id,)
                )
                conn.commit()

            print(f"[done] {filename} — {len(chunks_data)} chunks, embeddings stored")

        except Exception as e:
            print(f"[error] {filename}: {type(e).__name__}: {e}")
            if doc_id:
                try:
                    with db() as conn, conn.cursor() as cur:
                        cur.execute(
                            "UPDATE documents SET status = 'error', error_message = %s WHERE id = %s",
                            (str(e)[:500], doc_id),
                        )
                        conn.commit()
                except Exception as db_err:
                    print(f"[error] failed to update error status: {db_err}")
            raise


@app.post("/webhook")
async def webhook(request: Request, background: BackgroundTasks):
    payload = await request.json()
    records = payload.get("Records", [])
    print(f"[webhook] received {len(records)} record(s)")

    queued = []
    for record in records:
        event_name = record.get("eventName", "")
        if not event_name.startswith("s3:ObjectCreated:"):
            print(f"[webhook] ignored event: {event_name}")
            continue
        s3 = record.get("s3", {})
        bucket = s3.get("bucket", {}).get("name")
        key = s3.get("object", {}).get("key")
        if not bucket or not key:
            print(f"[webhook] missing bucket or key, skipping")
            continue
        key = unquote(key)
        if not key.lower().endswith(".pdf"):
            print(f"[webhook] non-PDF skipped: {key}")
            continue
        print(f"[webhook] scheduling: {bucket}/{key}")
        background.add_task(process_pdf, bucket, key)
        queued.append(f"{bucket}/{key}")

    return {"status": "queued", "items": queued}
