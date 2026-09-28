# week04 — buckets

**Antes:** cada retriever era um stack. Comparar = subir outro docker compose.
**Agora:** um stack, vários **buckets**. Bucket = corpus + gabarito + retriever, com endpoint próprio.

```
bucket "legal-1200 · bm25"      → GET /api/buckets/a1b2c3d4/search?q=...
bucket "legal-1200 · e5-large"  → GET /api/buckets/e5f6a7b8/search?q=...
```

Mesmo corpus, dois retrievers → dois buckets, comparáveis no eval.

## Retrievers

| | |
|---|---|
| `bm25` | palavra exata, em RAM |
| `minilm` · `e5-base` · `e5-large` · `bge-m3` · `qwen3-0.6b` | vetor, pgvector |

Adicionar outro = uma linha em `RETRIEVERS` (`services/chunks-api/main.py`).

## Fluxo

1. Crie um bucket (só o nome).
2. Suba `corpus.parquet` + `answers.parquet` da mesma pasta do dataset e escolha o retriever.
3. BM25 fica `ready` na hora; vetor calcula em background (`n/N`).
4. Use o endpoint do bucket, ou meça com o [`../eval`](../eval).

Bucket não muda depois de pronto: outro corpus ou modelo = outro bucket.
Assim um resultado do eval sempre aponta pra uma coisa só.

## Rodando

```bash
docker compose up --build        # ui :3000 · api :8000/docs · postgres :5432
```

**Próximo:** o retriever virou variável; o chunker ainda não. → [week05](../week05)
