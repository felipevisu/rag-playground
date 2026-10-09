# rag-playground

A retrieval-augmented generation (RAG) pipeline, improved one week at a time.
Each week changes **one** thing and measures whether retrieval got better.

```
question → [layer 1: documents] → [chunk search] → [rerank] → top-k
            week09                  week01·03·06     week07·10
            + question rewrites (week08) at every stage
```

The corpus is a set of PDF documents with a fixed question set and an answer key,
so every change is compared on the same ground.

## The weeks

| week | what it adds | in one line |
|---|---|---|
| [01](week01) | **Baseline RAG** | Chunks → embeddings in pgvector → nearest 5 chunks → an LLM answers and cites its sources. |
| [02](week02) | **Metrics** | A question set with known-correct chunks, scored with hit rate, recall, precision, MRR and nDCG. "Looks good" becomes a number. |
| [03](week03) | **Two retrievers side by side** | A multilingual embedding model vs. BM25 keyword search, on the same questions. They fail on different questions. |
| [04](week04) | **Buckets** | One stack, many indexes. A bucket = corpus + answer key + retriever, with its own search endpoint. Comparing retrievers means creating a bucket, not a new stack. |
| [05](week05) | **Dataset builder** | PDFs in, chunked corpus out. Pluggable chunkers (fixed, recursive, sentence, structure-aware), optional LLM-written chunk context, and an answer key stored as literal text spans so it re-resolves for any chunker. |
| [06](week06) | **Hybrid search** | BM25 and vector search run together and are fused by rank (Reciprocal Rank Fusion). BM25 gets language-aware tokenization: stemming, accent folding, acronym expansion. |
| [07](week07) | **Reranking** | The first stage fetches ~30 candidates; a cross-encoder reads question + chunk together and reorders them. A per-query option, not a per-bucket one. |
| [08](week08) | **Multi-query** | Each question is rewritten into several variants by an LLM (cached once); each variant searches and the rankings are fused. More phrasings, more vocabulary covered. |
| [09](week09) | **Two-layer retrieval** | A short summary per document picks the most relevant documents first; chunk search then runs only inside them. |
| [10](week10) | **A reranker that helps** | The reranker sees which document a chunk belongs to, votes alongside the first stage (RRF) instead of overriding it, and an LLM-based reranker (Qwen3-Reranker) is added. Optional Mac GPU rerank server. |
| [11](week11) | **Bigger corpus** | ~6× more documents across three topic segments and a much larger question set, to test whether the pipeline holds at scale. |

What each week fixes from the previous one:

- **01 → 02**: no way to tell if a change helped; now there is a score.
- **02 → 03**: embeddings miss exact terms (codes, acronyms); BM25 misses synonyms. They miss *different* questions.
- **03 → 04**: every comparison needed a new deployment; now it is a new bucket.
- **04 → 05**: the chunker was a constant; now it is a variable.
- **05 → 06**: instead of picking BM25 or vectors, use both.
- **06 → 07**: the first stage finds; a second, more expensive stage orders.
- **07 → 08**: users don't phrase questions like the documents do; multiple phrasings reach more of them.
- **08 → 09**: a chunk doesn't know which document it came from; a document summary does.
- **09 → 10**: the reranker alone reshuffled an already-good top 5 and made it worse; now it has document context and only votes.
- **10 → 11**: everything so far was tuned on a small corpus; scale it up.

## Supporting folders

| folder | what it is |
|---|---|
| [eval](eval) | Runs the question set against any bucket/option combination and stores every run as versioned JSON, with a diff against the previous run. |
| [pocs/bm25](pocs/bm25) | BM25 from scratch, step by step, with every score explained. |
| [pocs/jev-rerank](pocs/jev-rerank) | Terminal benchmark: hybrid retrieval with and without an LLM judge reranking the top 10. |

## Latest runs

The three newest runs, full rows as in the eval report ([eval](eval)). Time and cost are what the
run spent waiting on Voyage and Jev, at list prices.

| run | chunker | retriever | summaries | rerank | variants | k | questions | chunks | hit rate | recall | precision | MRR | nDCG | time | Voyage | Jev | cost |
|---|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 2026-10-09T16-04-45 | sentence-512t-multilingual-e5-large | hybrid-voyage-4-large | hybrid-voyage-4-large · top-5 docs | jev-rerank · top-30 · keep 1 | 5 | 5 | 34 (5%) | 2559 | 1.000 | 0.990 | 0.282 | 0.934 | 0.934 | 2m42 | 1m40 · $0.0007 | 52s · $0.04 | $0.04 |
| 2026-10-09T16-01-33 | sentence-512t-multilingual-e5-large | hybrid-voyage-4-large | hybrid-voyage-4-large · top-5 docs | voyage-rerank · top-30 · keep 1 | 5 | 5 | 34 (5%) | 2559 | 1.000 | 0.985 | 0.277 | 0.912 | 0.907 | 2m51 | 2m41 · $0.03 | — | $0.03 |
| 2026-10-09T15-58-27 | sentence-512t-multilingual-e5-large | hybrid-voyage-4-large | hybrid-voyage-4-large · top-5 docs | — | 5 | 5 | 34 (5%) | 2559 | 1.000 | 0.976 | 0.312 | 0.909 | 0.894 | 1m30 | 80s · $0.0007 | — | $0.0007 |

## Running

```sh
docker compose up --build
```

| | URL |
|---|---|
| buckets (week10) | http://localhost:3000 |
| API | http://localhost:8000/docs |
| eval | http://localhost:8080 |
| dataset / chunkers | http://localhost:8765 |

## Workflow in 4 steps

1. **Chunk** at :8765: pick a config, download the `bucket.zip`.
2. **Index** at :3000: import the zip and choose a retriever (`bm25`, `e5-large`, `hybrid-bge-m3`, …).
   Optional: import `week05/out/description.parquet` as a document-summaries bucket.
3. **Search**:
   ```
   GET /api/buckets/<id>/search?q=<question>&k=5
       &rerank=mminilm-rerank     # week07
       &variants=4                # week08
       &docs=<summaries bucket>   # week09
   ```
4. **Measure** at :8080: chunks bucket + summaries + rerank + variants → ▶ Run.
   Each run is saved as JSON in `eval/runs/`, under version control.

## BM25 vs. embeddings, in one example

```
"Funding sources for the FNARC include, without prejudice to others ..."

BM25       → ['funding', 'sources', 'fnarc', ...]     # counts words; a rare term like 'fnarc' weighs a lot
embedding  → [-0.013, 0.005, 0.005, -0.010, ...]      # 1024 numbers: meaning, not words
```

Each week's folder has its own README with details, numbers and how to run it.
