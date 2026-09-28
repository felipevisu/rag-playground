# week03/new-model — embedding que fala português

**Antes (week01):** `all-MiniLM-L6-v2`, treinado só em inglês.
**Agora:** `intfloat/multilingual-e5-base`. Só isso mudou.

| | MiniLM | e5-base |
|---|---|---|
| idiomas | inglês | 100, com português |
| dimensão | 384 | 768 |
| prefixos | — | `passage:` no chunk, `query:` na pergunta |

Sem os prefixos o e5 piora **em silêncio**, sem erro nenhum. Por isso ficam
grudados no `encode()` dos dois lados.

## Resultado

| | MiniLM | e5-base |
|---|---|---|
| hit rate | 0.684 | **0.921** |
| recall | 0.476 | **0.774** |
| MRR | 0.393 | **0.690** |

## Rodando

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
docker compose up --build        # chat :3011 · documentos :3010 · api :8001 · postgres :5433
```
