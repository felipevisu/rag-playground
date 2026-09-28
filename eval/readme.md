# eval

Mede cada versão do RAG com as mesmas 38 perguntas e guarda o histórico.
Nasceu no [week02](../week02); cada semana depois só ganhou um seletor novo.

| semana | o painel ganhou |
|---|---|
| week04 | escolher o **chunks bucket** (corpus + gabarito + retriever) |
| week07 | **rerank** |
| week08 | **variantes** da pergunta |
| week09 | **descrições** (camada 1) + quais PDFs foram escolhidos por pergunta |

## Rodando

```sh
docker compose up -d ui          # painel em http://localhost:8080
```

Escolha os seletores, ▶ Rodar. A API (week09) precisa estar de pé.

Pelo terminal:

```sh
docker compose run --rm eval --options                                   # buckets disponíveis
docker compose run --rm eval --bucket 59cb0095
docker compose run --rm eval --bucket 59cb0095 --rerank mminilm-rerank --variants 4 --docs 3fa1c2d4
docker compose run --rm eval --list                                      # histórico
```

Cada run imprime a diferença contra o anterior:

```
  hit_rate   0.684 ↑ 0.737  (+0.053)
  precision  0.158 ↓ 0.097  (-0.061)
```

## Onde fica

`runs/<run_id>.json`: config completa, as 5 métricas (explicadas no
[week02](../week02)), separadas em perguntas de 1 chunk e de vários, e o
detalhe por pergunta. Os JSON **são** o banco: versionados, a métrica muda no
mesmo commit que a causou.

`python run.py --self-check` confere as métricas com os exemplos do week02.
