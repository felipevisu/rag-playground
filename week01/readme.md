# week01 — RAG básico

**Antes:** nada. **Agora:** pergunta → chunks parecidos → Claude responde citando a fonte.

```
"O que o PL 1502/2026 define como racismo ambiental?"
   → vetor da pergunta (MiniLM, 384 números)
   → 5 chunks mais próximos no pgvector
   → Claude responde só com esses 5, e mostra de onde tirou
```

## Como funciona

1. `loader` lê `corpus.parquet` (183 chunks já cortados) e grava no Postgres.
2. Cada chunk vira um vetor (`all-MiniLM-L6-v2`), com índice HNSW.
3. A pergunta vira vetor, busca os 5 mais próximos, Claude responde.

`chunk_id` (`PL_1502_2026::0003`) é a chave do dataset, não um `SERIAL`: sobrevive
a um reload, e é isso que deixa o gabarito do week02 apontar pro chunk certo.

## Rodando

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
docker compose up --build
```

| | |
|---|---|
| chat | http://localhost:3001 |
| documentos e chunks | http://localhost:3000 |
| API | http://localhost:8000 |

Perguntas pra testar: [`perguntas-exemplo.md`](perguntas-exemplo.md).

**Próximo:** funciona, mas funciona *bem*? → [week02](../week02)
