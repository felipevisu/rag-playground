import io
import os
import time
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from minio import Minio
from minio.error import S3Error
from pgvector.psycopg2 import register_vector

DATABASE_URL = os.environ["DATABASE_URL"]
MINIO_ENDPOINT = os.environ["MINIO_ENDPOINT"]
MINIO_ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
MINIO_SECRET_KEY = os.environ["MINIO_SECRET_KEY"]
MINIO_BUCKET = os.environ["MINIO_BUCKET"]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)

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
            with db():
                pass
            return
        except psycopg2.OperationalError:
            print(f"DB not ready (attempt {attempt}/{retries})")
            time.sleep(delay)


def warm_embedder() -> None:
    print("Loading embedding model…")
    get_embedder()
    print("Document API ready.")


app = FastAPI(on_startup=[wait_for_db, warm_embedder])
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/documents")
def list_documents():
    with db() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT d.id, d.filename, d.processed_at, d.status, d.error_message,
                   COUNT(c.id)::int AS chunk_count,
                   COUNT(c.embedding)::int AS embedded_count
            FROM documents d
            LEFT JOIN chunks c ON c.document_id = d.id
            GROUP BY d.id
            ORDER BY d.processed_at DESC
        """)
        return list(cur.fetchall())


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")

    content = await file.read()
    filename = file.filename

    with db() as conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO documents (filename, status)
               VALUES (%s, 'pending')
               ON CONFLICT (filename) DO UPDATE SET status = 'pending', processed_at = NOW()
               RETURNING id""",
            (filename,),
        )
        doc_id = cur.fetchone()[0]
        conn.commit()

    minio_client.put_object(
        MINIO_BUCKET,
        filename,
        io.BytesIO(content),
        len(content),
        content_type="application/pdf",
    )

    return {"id": doc_id, "filename": filename, "status": "pending"}


@app.get("/api/documents/{doc_id}/chunks")
def get_chunks(doc_id: int):
    with db() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT id FROM documents WHERE id = %s", (doc_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="Document not found")
        cur.execute("""
            SELECT id, chunk_index, text, headings, page_numbers, embedding
            FROM chunks WHERE document_id = %s ORDER BY chunk_index
        """, (doc_id,))
        rows = cur.fetchall()
        for r in rows:
            emb = r.get("embedding")
            if emb is not None:
                r["embedding"] = [float(x) for x in emb]
        return rows


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: int):
    with db() as conn, conn.cursor() as cur:
        cur.execute("SELECT filename FROM documents WHERE id = %s", (doc_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Document not found")
        filename = row[0]
        cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
        conn.commit()

    try:
        minio_client.remove_object(MINIO_BUCKET, filename)
    except S3Error as e:
        print(f"MinIO delete warning for {filename}: {e}")

    return {"deleted": doc_id, "filename": filename}


@app.get("/api/search")
def search(
    q: str = Query(..., min_length=1, description="Query text"),
    k: int = Query(5, ge=1, le=50, description="Top-K results"),
):
    vec = get_embedder().encode(q, normalize_embeddings=True).tolist()
    with db() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT c.id, c.chunk_index, c.text, c.headings, c.page_numbers,
                   d.id AS document_id, d.filename,
                   (1 - (c.embedding <=> %s::vector))::float AS similarity
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.embedding IS NOT NULL
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
            """,
            (vec, vec, k),
        )
        return {"query": q, "k": k, "results": list(cur.fetchall())}


@app.get("/api/minio/console-url")
def minio_console_url():
    return {"url": "http://localhost:9001", "bucket": MINIO_BUCKET}
