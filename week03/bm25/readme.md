# week03/bm25 — sem embedding nenhum

**Antes:** busca por vetor. **Agora:** BM25, conta palavras em comum.

```
pergunta: "fontes de recursos do FNARC"
chunk:    "São fontes de recursos do FNARC, sem prejuízo..."
          ↑ 'fnarc' é raro no corpus → pesa muito
          ↑ 'de', 'do' estão em todo chunk → pesam quase nada
```

- Sem modelo, sem vetor, sem pgvector. Boot em segundos.
- Tokenizador cru (`\w+`), sem stemmer: é o baseline. O BM25 em português vem no week06.
- Pode devolver **menos de k**: chunk sem nenhuma palavra da pergunta não entra.

## Resultado

| | MiniLM | e5-base | **BM25** |
|---|---|---|---|
| hit rate | 0.684 | **0.921** | 0.868 |
| MRR | 0.393 | 0.690 | **0.700** |

BM25 ganha em MRR: 29 das 38 perguntas citam o número do PL, e número exato é
onde o vetor borra. Os erros quase não se sobrepõem com os do e5.

Passo a passo da fórmula: [`pocs/bm25`](../../pocs/bm25).

## Rodando

```bash
cp .env.example .env   # ANTHROPIC_API_KEY, só pro chat
docker compose up --build        # chat :3021 · documentos :3020 · api :8002 · postgres :5434
```
