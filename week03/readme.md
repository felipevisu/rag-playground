# week03 — dois braços, um gabarito

Duas formas de recuperar do mesmo corpus, medidas pelo mesmo eval. O `chunk_id`
é idêntico nos dois, então o gabarito do [`../week02`](../week02) pontua os dois
sem uma linha de mudança.

| | [`new-model/`](new-model) | [`bm25/`](bm25) |
|---|---|---|
| Recuperação | vetor denso, cosseno | BM25 Okapi, léxico |
| Modelo | `intfloat/multilingual-e5-base` | nenhum |
| Postgres | pgvector + HNSW | postgres puro |
| Compose | `week03-new-model` | `week03-bm25` |

## Resultado (38 perguntas, k=5)

| Métrica | MiniLM (week01) | e5-base | BM25 |
|---|---|---|---|
| hit rate | 0.684 | **0.921** | 0.868 |
| recall | 0.476 | **0.774** | 0.689 |
| precision | 0.158 | **0.284** | 0.242 |
| MRR | 0.393 | 0.690 | **0.700** |
| nDCG | 0.358 | **0.664** | 0.633 |

BM25, sem modelo nenhum, fica a 0.05 do embedding multilíngue e ganha em MRR. E
erra em perguntas diferentes: só 1 das 38 falha nos dois. Detalhe em cada readme.

## Rodando

Os três stacks sobem juntos — portas deslocadas de propósito.

```bash
cd new-model && docker compose up --build   # 3010 / 3011 / 8001 / 5433
cd bm25      && docker compose up --build   # 3020 / 3021 / 8002 / 5434

cd ../eval
STACK=week03-new-model docker compose run --rm eval --label "e5-base 768d"
STACK=week03-bm25      docker compose run --rm eval --label "bm25 puro"
docker compose run --rm eval --list
```
