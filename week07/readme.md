# RAG v6 — reranking com cross-encoder

Mesmo stack do [week06](../week06) (BM25, vetorial, híbrido). A novidade é um
**segundo estágio**: o retriever traz 30 candidatos (`RERANK_DEPTH`), um cross-encoder relê
cada um junto com a pergunta e reordena. Sai o top-k.

## Bi-encoder × cross-encoder

Os embeddings das semanas anteriores são **bi-encoders**: pergunta e chunk
viram vetores separados, comparados por cosseno.

```
bi-encoder:     enc(pergunta) · enc(chunk)          → cosseno
cross-encoder:  modelo("[pergunta] [SEP] [chunk]")  → um score de relevância
```

- **Bi-encoder**: o vetor do chunk é calculado uma vez, no upload. Buscar é
  barato. Mas o modelo nunca vê pergunta e chunk juntos: tudo que ele sabe
  do chunk tem que caber num vetor.
- **Cross-encoder**: a atenção cruza cada palavra da pergunta com cada
  palavra do chunk. Bem mais preciso, mas nada é pré-calculado: uma passada
  do modelo por par (pergunta, chunk). Pro corpus inteiro não dá; pra 50
  candidatos dá.

Por isso os dois estágios:

```
bm25 / vetor / híbrido  → top-30   (barato, busca recall)
cross-encoder           → top-5    (caro, busca precisão)
```

O reranker só reordena: se a resposta não está nos 30, ele não acha. O teto
é o recall@30 do primeiro estágio.

## Rerankers

| nome | modelo | |
|---|---|---|
| `mminilm-rerank` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | multilíngue (mMARCO, tem português), ~118M, **o rápido** |
| `bge-reranker` | `BAAI/bge-reranker-v2-m3` | multilíngue, ~570M, base bge-m3 |
| `mxbai-rerank` | `mixedbread-ai/mxbai-rerank-base-v2` | multilíngue, ~500M, base Qwen2 |

O v1 do mxbai é só inglês; o corpus é português, por isso o v2.
LLM como reranker ficou de fora (custo de tokens).

## Como está no código

`services/chunks-api/main.py`: o rerank é uma opção **da busca**, não do
bucket. Nada novo é indexado, então qualquer bucket (bm25, vetorial, híbrido)
roda com ou sem reranker:

```
GET /api/buckets/<id>/search?q=...&k=5                        # só o retriever
GET /api/buckets/<id>/search?q=...&k=5&rerank=bge-reranker    # top-30 → cross-encoder → top-5
```

```python
if rerank_with:
    ranked = rerank(cur, b["id"], rerank_with, q, rank(cur, b, q, max(k, RERANK_DEPTH)))[:k]
else:
    ranked = rank(cur, b, q, k)

# rerank: CrossEncoder.predict([(q, heading + text), ...]) e ordena pelo score
```

`/api/health` lista os rerankers em `rerankers`.

`similarity` na resposta é o score do cross-encoder: 0–1 no bge e no mxbai, logit (pode ser negativo) no mMiniLM.
Só a ordem dentro de uma busca significa algo.

Botão novo no `docker-compose.yml`: `RERANK_DEPTH` (padrão 30). Mais fundo =
mais recall pro reranker e mais CPU por busca.

## Rodando

```bash
docker compose up --build       # ui :3000 · api :8000/docs · postgres :5432
```

Mesmo volume do week05/06: os buckets existentes continuam lá.

1. Use um bucket que já existe (ex. `hybrid-<modelo>`), ou crie um.
2. No painel do [`../eval`](../eval) (:8080), escolha o bucket e o rerank
   (`sem rerank`, `bge-reranker`, `mxbai-rerank`) e rode uma vez com cada um.
   Pela CLI: `--rerank bge-reranker`. O run grava `config.rerank`.
3. A primeira busca com cada reranker baixa o modelo (~1–2 GB, uma vez só,
   cache em `~/.cache/huggingface`).

Custo, em CPU dentro do Docker (~280 GFLOPS; o Mac nativo faz ~1450), 30
candidatos de ~512 tokens por pergunta:

| reranker | por pergunta | eval (38) |
|---|---|---|
| `mminilm-rerank` | ~3,3 s | ~2 min |
| `bge-reranker` | ~45 s | ~30 min |
| `mxbai-rerank` | ~60 s | ~40 min |

Itere com `mminilm-rerank`; bge/mxbai só pra confirmar no fim.

`python main.py` roda o self-check (fusão, tokenizer, registro dos rerankers) sem banco.
