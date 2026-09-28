# RAG v7 — query transformation (multi-query)

Mesmo stack do [week07](../week07) (BM25, vetorial, híbrido, rerank). A novidade
mexe na **pergunta**, não no índice: a pergunta tem N versões reescritas,
cada versão é buscada, e os rankings são fundidos por RRF.

```
"qual o prazo de validade da CNH para idosos?"
  ├─ "prazo de validade da Carteira Nacional de Habilitação para condutores idosos"
  ├─ "periodicidade do exame de aptidão física e mental ... art. 147 do CTB, Lei 9.503/1997"
  ├─ "validade da CNH de motorista com mais de 70 anos e exigência de renovação do exame médico"
  └─ "regras de validade e renovação da CNH por faixa etária"

cada uma → retriever do bucket (top-RRF_DEPTH) → RRF → top-k   (→ rerank, se pedido)
```

## Por quê

O usuário pergunta com as palavras dele; a lei está escrita com as dela
("CNH para idosos" × "condutores com idade igual ou superior a 65 anos").
O BM25 não acha o que não compartilha termo; o embedding acha perto, mas um
vetor só de uma pergunta curta é frágil. Várias formulações cobrem mais
vocabulário, e um chunk que aparece bem em várias delas sobe no RRF.

Ambiente de teste: as reescritas são **pré-geradas** uma vez
(`gen_rewrites.py`, `claude-haiku-4-5`, 5 por pergunta) em
`services/chunks-api/rewrites.json`, versionado. A API só lê o arquivo, nunca
chama o Claude: buscar custa N buscas e zero tokens, e o eval é reprodutível.
Pergunta fora do arquivo → 404.

## Como está no código

`services/chunks-api/main.py`: `?variants=N` é opção **da busca**, como o
`?rerank=`. Nada é reindexado; qualquer bucket serve.

```
GET /api/buckets/<id>/search?q=...&k=5                                  # só o retriever
GET /api/buckets/<id>/search?q=...&k=5&variants=4                       # 1 + 4 buscas → RRF → top-5
GET /api/buckets/<id>/search?q=...&k=5&variants=4&rerank=bge-reranker   # → top-30 → cross-encoder → top-5
```

```python
queries = [q, *rewrite(q, variants)] if variants else [q]
ranked = rrf(*(rank(cur, b, x, RRF_DEPTH) for x in queries))[:depth]
if rerank_with:
    ranked = rerank(..., queries[0], ranked)[:k]   # o reranker lê a pergunta original
```

- `gen_rewrites.py`: lê `week05/out/queries.parquet`, structured output
  (`{"queries": [...]}`), prompt em português pedindo sinônimos, termo
  jurídico, versão mais específica e mais genérica, mantendo números de
  lei/artigo e siglas. Só gera o que falta no `rewrites.json`.
- `rewrite(q, n)`: as n primeiras do arquivo.
- A pergunta original entra sempre na fusão.
- A resposta traz `queries` (original + reescritas); o eval grava as
  reescritas por pergunta (`rewrites`) e mostra no detalhe.
- `/api/health` expõe `transform: {model, max_variants, questions}` (máx. 5).

## Rodando

```bash
docker compose up --build       # ui :3000 · api :8000/docs · postgres :5432

# só se as perguntas mudarem (precisa de ANTHROPIC_API_KEY no .env e anthropic + pyarrow):
python services/chunks-api/gen_rewrites.py
```

Mesmo volume do week05/06/07: os buckets existentes continuam lá. O
`docker-compose.yml` da raiz agora inclui o week08 no lugar do week07.

No painel do [`../eval`](../eval) (:8080) aparece o seletor **variantes**
(`sem variantes`, 1–5). Rode o mesmo bucket com e sem, e com e sem rerank.
Pela CLI: `--variants 4`. O run grava `config.variants` e `config.transform_model`.

`python main.py` roda o self-check (fusão, rerank, multi-query) sem banco nem API.
