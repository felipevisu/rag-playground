# RAG v5 — busca híbrida (BM25 + embeddings)

Mesmo stack de buckets do week04. A novidade é um tipo de retriever a mais:
o **híbrido**, que junta a busca lexical (BM25) com a vetorial (embeddings)
usando **Reciprocal Rank Fusion (RRF)**.

## Por que juntar os dois

Cada busca erra de um jeito diferente:

- **BM25** acha o chunk que tem *a mesma palavra* da pergunta. Ótimo pra
  número de artigo, sigla, nome próprio. Cego pra sinônimo e paráfrase.
- **Embedding** acha o chunk que *fala da mesma coisa*, mesmo com outras
  palavras. Ótimo pra pergunta em linguagem natural. Escorrega em termo
  exato raro, que o modelo nunca viu direito.

Um chunk que os dois acham é quase certo que é bom. Um chunk que só um acha
ainda pode ser. O híbrido aproveita os dois sem escolher.

## O problema: os scores não se somam

BM25 devolve um número sem teto (3.2, 17.8, 41.0…). Cosseno devolve algo
entre 0 e 1. Somar isso direto é somar metro com quilo. Dá pra normalizar
(min-max, z-score), mas aí cada query precisa de ajuste e um outlier
distorce tudo.

## A solução: fundir por posição, não por score

RRF ignora o score e usa só a **posição** de cada chunk em cada ranking:

```
score(chunk) = Σ  1 / (k + posição_no_ranking_r)     para cada ranker r onde o chunk aparece
                r
```

com `k = 60`. Exemplo com dois rankers:

| chunk | posição BM25 | posição vetor | RRF |
|---|---|---|---|
| A | 1 | 3 | 1/61 + 1/63 = **0.0323** |
| B | 2 | — | 1/62 = 0.0161 |
| C | — | 1 | 1/61 = 0.0164 |
| D | 7 | 9 | 1/67 + 1/69 = 0.0294 |

A ganha por estar bem nos dois. D, mediano nos dois, passa na frente de B e
C, que só um ranker viu — é o efeito de consenso. O `k=60` amortece: quem
está em 1º não esmaga quem está em 5º, então o segundo ranker ainda tem voz.

Sem normalização, sem peso pra calibrar, funciona com qualquer combinação de
rankers. É o método de Cormack, Clarke & Büttcher (2009), e o padrão em
Elasticsearch, Weaviate, OpenSearch etc.

## Como está no código

`services/chunks-api/main.py`:

```python
def rrf(*rankings, k=60):
    fused = {}
    for ranking in rankings:
        for pos, (chunk_id, _) in enumerate(ranking, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0) + 1 / (k + pos)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)

# híbrido = top-50 do BM25 + top-50 do vetor, fundidos, corta em k
rrf(bm25_rank(..., 50), vector_rank(..., 50))[:k]
```

Cada modelo vetorial ganha automaticamente um `hybrid-<modelo>`
(`hybrid-e5-base`, `hybrid-bge-m3`, …). O bucket híbrido embeda igual ao
vetorial; o índice BM25 é montado em RAM na primeira busca.

`python main.py` roda um self-check da fusão sem banco.

## Rodando

```bash
docker compose up --build       # ui :3000 · api :8000/docs · postgres :5432
```

1. Crie um bucket, suba `corpus.parquet` + `answers.parquet` de `dataset/out/<config>/`
   escolhendo o retriever `hybrid-<modelo>`.
2. Espere `ready`.
3. Meça com [`../eval`](../eval): o mesmo parquet em três buckets
   (`bm25`, `<modelo>`, `hybrid-<modelo>`) mostra o que a fusão ganha.

`similarity` na resposta é o score RRF (máximo ≈ 0.03 com dois rankers).
Só a ordem dentro de uma busca significa algo.

## Próximo passo: BM25 em português

O BM25 não sabe de idioma; quem decide é o tokenizador, e o nosso é o mínimo:
`re.findall(r"\w+", text.lower())`. Acentos ficam (`\w` é Unicode) e as
stopwords o IDF já esmaga. O que falta:

- **Stemming.** `contrato`, `contratos`, `contratual`, `contratação` são
  quatro termos sem relação. Pergunta com "pagamento" e chunk com "pagos"
  = zero match. É o furo principal em português.
- **Acento sem normalização.** Query `acao` não casa com `ação` no corpus.

Ordem de custo/benefício:

1. Stripar acentos com `unicodedata` (stdlib) em corpus e query.
2. Stemmer português: `nltk` (RSLP ou Snowball `portuguese`) ou `PyStemmer`
   (Snowball em C). Uma dependência e um `stem()` dentro de `tokenize`.

Medir antes e depois com o `eval` no bucket `bm25`. O índice é montado na
primeira busca, então trocar o `tokenize` vale sem re-upload: só reiniciar a
API. O híbrido usa o mesmo BM25, então herda o ganho.

## Retrievers

| retriever | tipo |
|---|---|
| `bm25` | léxico, `rank-bm25` em RAM |
| `minilm` `e5-base` `e5-large` `bge-m3` `qwen3-0.6b` | vetorial, pgvector cosseno |
| `hybrid-<modelo>` | BM25 + vetorial, RRF |

Detalhes de bucket, API e schema: iguais ao [`../week04`](../week04).
