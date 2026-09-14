# rag-playground

Três stacks que se encadeiam: **dataset** gera o corpus e o gabarito, **week05** indexa e busca, **eval** mede.

## Endereços

| Stack | Serviço | URL | Como subir |
|---|---|---|---|
| dataset | UI de build/browse dos chunks | http://localhost:8765 | `cd dataset && docker compose up ui` |
| week05 | UI de buckets e busca | http://localhost:3000 | `cd week05 && docker compose up --build` |
| week05 | API (Swagger) | http://localhost:8000/docs | idem |
| week05 | Postgres (pgvector) | `postgresql://rag:rag@localhost:5432/rag` | idem |
| eval | Painel de avaliação | http://localhost:8080 | `cd eval && docker compose up -d ui` |

week05 usa as mesmas portas do week01 — só um dos dois no ar por vez.

## Fluxo

1. **[dataset](dataset/)** — 19 PDFs de Projetos de Lei + 38 perguntas com trechos de resposta verbatim.
   `docker compose run --rm build configs/legal-1200.yaml` gera `out/<config>/` com `corpus.parquet`, `answers.parquet` (qrels) e `manifest.json`. `out/queries.parquet` é independente de chunker.
2. **[week05](week05/)** — cria um bucket, sobe `corpus.parquet` + `answers.parquet` de `dataset/out/<config>/` e escolhe o retriever: `bm25`, vetorial (`minilm`, `e5-base`, `e5-large`, `bge-m3`, `qwen3-0.6b`) ou `hybrid-<modelo>` (BM25 + vetor via RRF). Espera `ready`.
3. **[eval](eval/)** — roda as 38 perguntas contra um bucket e grava `runs/<run_id>.json` (hit rate, recall, precision, MRR, nDCG). Precisa do week05 no ar. Pelo painel em :8080 ou:
   ```sh
   cd eval
   docker compose run --rm eval --options                      # buckets prontos
   docker compose run --rm eval --bucket <id> --label "bm25"   # roda
   docker compose run --rm eval --list                         # histórico
   ```

## Dados (`dataset/out/<config>/`)

```mermaid
erDiagram
    QUERIES ||--o{ ANSWERS : "query_id"
    CORPUS  ||--o{ ANSWERS : "chunk_id"

    QUERIES {
        str query_id PK "q001"
        str question
        str filename "PDF onde a resposta vive"
    }
    CORPUS {
        str chunk_id PK "PL_1502_2026::0003"
        str filename "PDF de origem"
        int chunk_index
        int[] pages
        str heading "so splitter legal"
        str text
        int n_chars
        int size "no unit da config"
        int start "span no texto parseado"
        int end
    }
    ANSWERS {
        str query_id FK
        str chunk_id FK
        int part "indice em answer[] da pergunta"
        float coverage "fatia do trecho neste chunk (0-1)"
    }
```

`queries.parquet` fica em `out/` e não depende de chunker. `corpus.parquet` e
`answers.parquet` ficam em `out/<config>/`: os `chunk_id` só fazem sentido dentro
da mesma pasta. `ANSWERS` é a tabela de qrels — uma linha por (pergunta, trecho de
resposta, chunk que o contém).

## BM25 × embedding: o que cada um faz com o texto

O mesmo trecho vira duas coisas diferentes:

```
"São fontes de recursos do FNARC, sem prejuízo de outras previstas em lei"
```

**BM25 → lista de tokens** (`re.findall(r"\w+", text.lower())`):

```python
['são', 'fontes', 'de', 'recursos', 'do', 'fnarc', 'sem', 'prejuízo', 'de', 'outras', 'previstas', 'em', 'lei']
```

A busca conta quais desses tokens aparecem na pergunta. `fnarc` é raro no corpus,
então pesa muito; `de` aparece em todo chunk, então pesa quase nada. Sinônimo não
casa: `cachorro` ≠ `animal`.

**Embedding → vetor de números** (`e5-large`, 1024 dimensões):

```python
[-0.013, 0.005, 0.005, -0.010, -0.002, -0.044, ...]   # 1024 floats
```

A busca mede o cosseno entre o vetor da pergunta e o de cada chunk. Palavras não
importam, só a posição no espaço: "cachorro de rua sem dono" e "animal comunitário
sem tutor" caem perto. Sigla que o modelo nunca viu (`FNARC`) cai em lugar nenhum.

O `hybrid-<modelo>` roda os dois e funde os rankings por posição (RRF). Detalhes em
[week05](week05/).

Detalhes de cada stack no `readme.md` da pasta.
