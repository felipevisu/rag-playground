# week05 — dataset e chunkers

**Antes:** um corpus fixo, cortado de um jeito só. **Agora:** 57 PDFs, vários
chunkers, e um gabarito que se recalcula pra cada um.

```
source/pdfs/  +  configs/sentence-450t.yaml   →  out/sentence-450t-.../bucket.zip
source/pdfs/  +  configs/legal-1200.yaml      →  out/legal-1200c-.../bucket.zip
```

Cada zip vira um bucket (week04+), e o eval compara os dois.

## Chunkers

| splitter | corta em |
|---|---|
| `fixed` | a cada `max` unidades, onde cair |
| `recursive` | parágrafo → linha → frase → espaço |
| `sentence` | fim de frase, agrupando até `max` |
| `legal` | `Art.`, `§`, `CAPÍTULO`, `JUSTIFICAÇÃO` |

```yaml
splitter: sentence
unit: tokens            # chars | words | tokens
max: 450
tokenizer: intfloat/multilingual-e5-large   # mede como o modelo vê
context: false          # true: Claude escreve 1 linha sobre o documento e o trecho
```

## O gabarito acompanha o chunker

As respostas são **trechos literais** do PDF, não chunks:

```json
{ "question": "Quais são as fontes de recursos do FNARC ...?",
  "answer": ["§ 1º São fontes de recursos do FNARC, sem prejuízo de outras ..."] }
```

O build acha cada trecho no texto e marca os chunks que o cobrem. Troca o
chunker, o gabarito se refaz sozinho. Trecho não encontrado → o build para.

## Saída

```
out/queries.parquet          as 38 perguntas (não dependem de chunker)
out/description.parquet      1 resumo por PDF (~1000 caracteres), pro week09
out/<config>/corpus.parquet  os chunks
out/<config>/answers.parquet quais chunks respondem cada pergunta
```

## Rodando

```sh
docker compose run --rm build configs/legal-1200.yaml   # gera out/legal-1200c-min300/
docker compose up ui                                    # http://localhost:8765 — build, navega, baixa o zip
```

**Próximo:** BM25 e vetor erram em perguntas diferentes; por que escolher? → [week06](../week06)
