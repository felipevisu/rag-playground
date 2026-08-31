# dataset

Evaluation dataset for the RAG playground: 19 Projetos de Lei (2026, meio
ambiente), 38 questions with verbatim answer passages, and as many chunked
corpora as you want to compare.

```
source/            inputs — hand-edited or downloaded. This is the work you lose.
  pdfs/              the 19 PDFs
  metadata.csv       written by download.py
  download.py        pull fresh PDFs from the Câmara open-data API
  questions-and-answers.json
                     questions + verbatim answer passages. Source of truth.
configs/           one YAML per chunker configuration (the UI writes these too)
build/             tooling. No data.
  chunkers.py        parse + split + pack -> chunks
  resolve.py         answer passages -> chunk ids
  build.py           CLI: one config -> out/<name>/
  serve.py, ui.html  local UI: pick a config, build, browse chunks, download
  test_*.py          self-checks: python test_chunkers.py && python test_resolve.py
out/               generated. Never edit by hand.
  queries.parquet    the questions (config-independent)
  <name>/            one folder per config
    corpus.parquet     the chunks
    answers.parquet    qrels — which chunks hold each answer passage
    manifest.json      config, versions, counts, sha of the PDFs
    config.yaml        copy of the config
```

Rule: **`out/` is fully reproducible from `source/` + `configs/` + `build/`.**

## Setup

```sh
cd build
python3.12 -m venv venv            # 3.12 recommended; 3.14 lacks lzma on some pyenv builds
venv/bin/pip install -r requirements.txt
```

Or with docker: `docker compose run --rm build configs/legal-1200.yaml`,
`docker compose up ui`.

## Building

```sh
venv/bin/python build.py ../configs/legal-1200.yaml    # -> out/legal-1200/
venv/bin/python serve.py                              # UI at http://localhost:8765
```

A build takes seconds (pypdf, no ML). `unit: tokens` downloads the tokenizer
on first use (~1 MB, cached in `~/.cache/huggingface`).

A passage from `questions-and-answers.json` that cannot be found in its PDF
aborts the build and names the passage. Fix it (it must be verbatim) and re-run.

## Chunker configuration

```yaml
name: legal-1200          # -> out/legal-1200/
splitter: legal           # fixed | recursive | sentence | legal
unit: chars               # chars | words | tokens
max: 1200                 # hard ceiling per chunk, in `unit`
min: 300                  # merge neighbours until at least this big (0 = off)
overlap: 0                # tail of the previous chunk repeated at the start of the next
tokenizer: ""             # HF model id, required when unit == tokens
strip_footer: true        # drop the Câmara page footer before chunking
```

| splitter | cuts at | `min` | `overlap` | `heading` |
|---|---|---|---|---|
| `fixed` | every `max` units, wherever that lands | — | ✓ | — |
| `recursive` | `\n\n` → `\n` → `. ` → `; ` → space, coarsest that fits | ✓ | ✓ | — |
| `sentence` | sentence ends, grouped up to `max` | ✓ | ✓ | — |
| `legal` | `Art.`, `§`, `Parágrafo único`, `CAPÍTULO`, `JUSTIFICAÇÃO` | ✓ | ✗ | `Art. 3º` etc. |

`unit: tokens` measures with the embedding model's own tokenizer, so `max`
means what the model will actually see. Known limits are listed in
`chunkers.TOKENIZERS` and shown in the UI; e.g. `all-MiniLM-L6-v2` truncates
at 256 regardless of its config.

The chunk count is a function of the config *and* of the pinned versions in
`requirements.txt`. Diff two `manifest.json` to see what changed.

## Schemas

**corpus.parquet**

| column | type | notes |
|---|---|---|
| `chunk_id` | str | `PL_1502_2026::0003` — unique within one config folder |
| `filename` | str | source PDF |
| `chunk_index` | int | position within the document |
| `pages` | list[int] | pages the chunk spans |
| `heading` | str | `legal` splitter only; `""` otherwise |
| `text` | str | |
| `n_chars` | int | |
| `size` | int | length in the config's `unit` |
| `start`, `end` | int | char span in the parsed document text (used by `resolve.py`) |

**queries.parquet** — 38 rows

| column | type |
|---|---|
| `query_id` | str — `q001` |
| `question` | str |
| `filename` | str — document the answer lives in |

**answers.parquet** — one row per (query, answer passage, chunk)

| column | type | notes |
|---|---|---|
| `query_id` | str | → `queries.query_id` |
| `chunk_id` | str | → `corpus.chunk_id` in the same folder |
| `part` | int | index into the question's `answer[]` |
| `coverage` | float | share of that passage held by this chunk (0–1) |

## Ground truth

`source/questions-and-answers.json`:

```json
{
  "query_id": "q002",
  "question": "Quais são as fontes de recursos do FNARC e como é composto o seu Comitê Gestor?",
  "answer": [
    "§ 1º São fontes de recursos do FNARC, sem prejuízo de outras previstas em lei: a) ...",
    "§ 3º O FNARC será gerido por Comitê Gestor, órgão colegiado deliberativo, ..."
  ]
}
```

`answer` is a list of **verbatim** passages from the PDF — one per distinct
place the answer lives. Whitespace, case and accents don't matter (the
resolver normalizes both sides); paraphrase does.

How a passage becomes qrels (`build/resolve.py`): the PDF is parsed with the
same settings as the chunker; the passage is located in that text (exact
match after normalization, fuzzy alignment as fallback for pypdf's occasional
accent garbage); the chunks whose `[start, end)` intersect it are picked
greedily until the passage is covered. A chunk only counts if it adds ≥ 20
uncovered chars, so `overlap` never yields redundant qrels.

## Refreshing the corpus

```sh
python3 source/download.py --year 2026 --topic "meio ambiente" --limit 20
```

New PDFs need new entries in `questions-and-answers.json` and a rebuild of
every config.
