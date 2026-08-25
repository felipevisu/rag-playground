# dataset

Frozen evaluation dataset for the RAG playground. Built once, committed, reused
by every week.

```
source/            inputs — hand-edited or downloaded. This is the work you lose.
  pdfs/              19 Projetos de Lei (2026, meio ambiente)
  metadata.csv       written by build/download.py
  ground-truth.json  questions, reference answers, anchors
build/             tooling. No data.
  build.py           chunk the PDFs, resolve the ground truth, write out/
  download.py        pull fresh PDFs from the Câmara open-data API
  Dockerfile
  requirements.txt   pinned — the chunk count depends on these versions
out/               generated. Never edit by hand.
  corpus.parquet     the chunks
  queries.parquet    the questions
  answers.parquet    the qrels — which chunks answer which question
  manifest.json      what produced this build
```

The rule the layout enforces: **`out/` is fully reproducible from `source/` +
`build/`.** Delete `out/` and one command brings it back. Delete `source/` and
you have lost human work.

## Why the chunks live here and not in the DB

`chunks.id` in week01's Postgres is a `SERIAL` — it changes on every re-ingest.
`chunk_index` moves too whenever the chunker changes, and the chunker is
tokenizer-coupled to the embedding model (see
`week01/services/worker/main.py:59`), so swapping `all-MiniLM-L6-v2` for another
model reshuffles every boundary.

Ground truth that points at those ids rots. So the chunks themselves are the
artifact: split once, frozen here, and everything downstream indexes *these*
rows. Swap the embedding model, re-embed the same `chunk_id`s, and the eval
numbers stay comparable.

This is not hypothetical. The same 19 PDFs with the same settings produced 180
chunks on one docling build and 183 on the next. That is what `manifest.json`
and the pinned `requirements.txt` are for.

## Building

```sh
docker compose run --rm build                  # full, ~35 min on CPU
docker compose run --rm build --reuse-corpus   # re-resolve only, seconds
python3 build/build.py --self-check            # resolver asserts, no docker
```

`--reuse-corpus` skips docling and re-resolves the ground truth against the
existing `out/corpus.parquet`. Use it while fixing anchors; re-chunking 19 PDFs
to test a one-line edit is 40 minutes of CPU for nothing.

An anchor that resolves to nothing aborts the build rather than writing a
quietly weaker ground truth. Fix the anchor and re-run.

## Schemas

**corpus.parquet** — 183 rows

| column | type | notes |
|---|---|---|
| `chunk_id` | str | `PL_1502_2026::0003` — stable, readable, not a `SERIAL` |
| `filename` | str | source PDF |
| `chunk_index` | int | position within the document |
| `pages` | list[int] | source pages, from docling provenance |
| `headings` | list[str] | section headings docling attached |
| `text` | str | the chunk |
| `n_chars` | int | |

**queries.parquet** — 38 rows

| column | type | notes |
|---|---|---|
| `query_id` | str | `q001` |
| `question` | str | |
| `type` | str | `single_chunk` or `multi_chunk` |
| `expected_answer` | str | reference answer, for generation eval |
| `filename` | str | document the answer lives in |
| `title` | str | document title |
| `n_anchors` | int | passages the answer needs |

**answers.parquet** — 78 rows, one per (query, relevant chunk)

| column | type | notes |
|---|---|---|
| `query_id` | str | → `queries.query_id` |
| `chunk_id` | str | → `corpus.chunk_id` |
| `relevance` | int | always 1; graded relevance is not used yet |
| `filename` | str | denormalized from corpus, so the file reads standalone |
| `chunk_index` | int | idem |
| `pages` | list[int] | idem |

## Ground truth

`source/ground-truth.json` is hand-edited. It stores **anchors** — verbatim text
lifted from the PDF — not chunk ids, because anchors survive a re-chunk and ids
do not.

```json
{
  "query_id": "q039",
  "type": "single_chunk",
  "question": "...",
  "expected_answer": "...",
  "anchors": ["a verbatim sentence or two copied out of the PDF"]
}
```

One anchor per distinct passage the answer needs — not one per question. Shrink
the chunk size later and a two-passage answer correctly resolves to two chunks.

The resolver tolerates the ways docling output drifts: whitespace, accents,
case, re-rendered list markers, and an anchor split across a chunk boundary
(which resolves to both chunks). It will not tolerate a paraphrase — anchors
must be verbatim.

## Refreshing the corpus

```sh
python3 build/download.py --year 2026 --topic "meio ambiente" --limit 20
```

Writes into `source/pdfs/` and `source/metadata.csv`. New PDFs need new
questions in `source/ground-truth.json` and a full rebuild.
