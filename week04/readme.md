# RAG v5 — buckets

Até o week03 o corpus entrava no stack por um `loader` lendo um parquet fixo de
`dataset/out/`, e cada stack tinha **um** retriever cravado no código. Comparar
dois chunkers ou dois modelos era subir dois stacks.

Aqui tudo é um **bucket**: você cria, dá um nome, o sistema gera um `id`, você
faz upload de um `corpus.parquet` escolhendo o retriever, e o bucket ganha o
seu próprio endpoint de busca.

```
bucket = nome + id + corpus.parquet + answers.parquet (gabarito) + retriever
GET /api/buckets/{id}/search?q=...
GET /api/buckets/{id}/answers
```

O gabarito sobe junto com o corpus porque só faz sentido contra ele: um
`chunk_id` é do chunker que o produziu. O upload recusa um `answers.parquet`
que aponte para chunk que não está no corpus enviado.

Comparar retrievers = dois buckets com o mesmo parquet. Comparar chunkers =
dois buckets com parquets diferentes.

| retriever | como | dim |
|---|---|---|
| `minilm`   | `sentence-transformers/all-MiniLM-L6-v2` (week01) | 384 |
| `e5-base`  | `intfloat/multilingual-e5-base` (week03/new-model), prefixos `passage:`/`query:` | 768 |
| `e5-large` | `intfloat/multilingual-e5-large` | 1024 |
| `bge-m3`   | `BAAI/bge-m3` — multilíngue, sem prefixos, 8k ctx | 1024 |
| `qwen3-0.6b` | `Qwen/Qwen3-Embedding-0.6B` — instrução só na query, 32k ctx | 1024 |
| `bm25`     | `rank-bm25` em RAM (week03/bm25), sem vetor | — |

Adicionar outro é uma linha em `RETRIEVERS` (`services/chunks-api/main.py`).

## Serviços

| | porta | |
|---|---|---|
| `chunks-ui`  | :3000 | página única (`index.html`, mesmo estilo do `dataset/build`) — cria bucket, upload, chunks |
| `chunks-api` | :8000 | FastAPI — `/docs` |
| `postgres`   | :5432 | pgvector |

Sem `loader`, sem `chat`, sem `.env`. Mesmas portas do week01: um stack de cada vez.

```bash
docker compose up --build
```

O primeiro upload com cada modelo baixa os pesos (~90 MB MiniLM, ~1.1 GB
e5-base, ~2.2 GB e5-large e bge-m3, ~1.2 GB qwen3-0.6b) pro cache `~/.cache/huggingface`.

## Fluxo

1. Gere um corpus em `dataset/` → `dataset/out/<config>/corpus.parquet`.
2. **create** um bucket na sidebar (só o nome; o `id` é gerado).
3. No bucket: escolha `corpus.parquet` + `answers.parquet` da mesma pasta
   `dataset/out/<config>/`, o retriever, **upload**. Colunas exigidas: corpus
   `chunk_id, filename, chunk_index, pages, text` (`heading` opcional); answers
   `query_id, chunk_id` (`part`, `coverage` opcionais).
4. BM25 fica `ready` na hora; vetorial embeda em background e o chip mostra `n/N`.
5. Copie o endpoint `GET /api/buckets/{id}/search?q=` do header.

Bucket é **write-once**: depois de `ready`, upload devolve 409. Outro corpus ou outro modelo = outro bucket. Só `empty` e `error` aceitam upload.

## API

| | |
|---|---|
| `GET    /api/buckets` | todos os buckets, com `search_url` |
| `POST   /api/buckets` `{"name"}` | cria vazio, devolve `id` |
| `GET    /api/buckets/{id}` | detalhe: status, retriever, contagens, sha |
| `DELETE /api/buckets/{id}` | cascade |
| `POST   /api/buckets/{id}/upload` multipart `file` + `answers` + `retriever` | só em `empty`/`error`; 409 depois de indexado |
| `GET    /api/buckets/{id}/answers` | o gabarito: `query_id, chunk_id, part, coverage` |
| `GET    /api/buckets/{id}/documents` · `…/documents/{filename}/chunks` | navegação; chunk traz `embedding` |
| `GET    /api/buckets/{id}/search?q&k` | **o endpoint do bucket** |
| `GET    /api/search?q&k&bucket=` | alias; sem `bucket` = último indexado. Pro `eval/` |
| `GET    /api/health` | retrievers disponíveis, bucket default |

`status`: `empty` → `indexing` → `ready` | `error` (mensagem em `error`).

`/api/search` devolve o mesmo shape dos stacks anteriores (`results[].id`,
`filename`, `similarity`), então o [`../eval`](../eval) roda com `STACK=week04`
sem mudar. `similarity` é cosseno nos vetoriais e score BM25 bruto no léxico —
só a ordem dentro de uma busca significa algo.

## Schema

```
buckets (id TEXT PK, name, created_at, retriever, status, chunk_count, embedded_count, parquet_sha, error, indexed_at)
chunks  (bucket_id → buckets CASCADE, chunk_id, filename, chunk_index, pages, heading, text, embedding VECTOR)
        PK (bucket_id, chunk_id)
qrels   (bucket_id → buckets CASCADE, query_id, chunk_id, part, coverage)
        PK (bucket_id, query_id, chunk_id, part)
```

`embedding VECTOR` sem dimensão: buckets de 384d e 768d dividem a tabela. A
busca é um scan exato filtrado por `bucket_id` — centenas de chunks, instantâneo.
`ponytail:` HNSW por dimensão se um bucket crescer 100×.

BM25 não guarda nada: o índice é montado em RAM na primeira busca do bucket e
descartado no delete/re-upload.

## Medir

O [`../eval`](../eval) escolhe um bucket e lê dele tanto a busca quanto o
gabarito. As perguntas (`dataset/out/queries.parquet`) não dependem de chunker
e ficam no eval.
