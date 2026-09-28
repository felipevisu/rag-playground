# week07 — rerank

**Antes:** o retriever acha e ordena. **Agora:** ele acha 30, um **cross-encoder** relê e ordena.

```
bm25 / vetor / híbrido  → top-30   barato, cobre muito
cross-encoder           → top-5    caro, lê pergunta + chunk juntos
```

```
GET /api/buckets/<id>/search?q=...&k=5                           # só o retriever
GET /api/buckets/<id>/search?q=...&k=5&rerank=mminilm-rerank     # + rerank
```

Embedding compara dois vetores feitos separados. O cross-encoder lê
`[pergunta] [chunk]` de uma vez: mais preciso, mas não dá pra pré-calcular.
Por isso só nos 30 candidatos. Se a resposta não está nos 30, ele não acha.

| reranker | por pergunta (CPU) |
|---|---|
| `mminilm-rerank` | ~3 s — use pra iterar |
| `bge-reranker` | ~45 s |
| `mxbai-rerank` | ~60 s |

É opção **da busca**, não do bucket: qualquer bucket roda com ou sem.

## Rodando

```bash
docker compose up --build        # ui :3000 · api :8000/docs
```

No eval (:8080), rode o mesmo bucket com e sem rerank.

**Próximo:** o usuário não pergunta com as palavras da lei. → [week08](../week08)
