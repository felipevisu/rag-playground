# RAG v4 — BM25 puro, sem embeddings

O braço léxico do week03. Mesmo corpus, mesmo `chunk_id`, mesmo gabarito do
[`../../week02`](../../week02) — e **nenhum vetor em lugar nenhum**.

Serve para responder uma pergunta que o [`../new-model`](../new-model) não
responde sozinho: quanto do ganho do embedding multilíngue é semântica de
verdade, e quanto é só o MiniLM sendo ruim em português?

## O que é BM25

Ranking léxico. Conta termo em comum entre pergunta e chunk, com duas correções
que o TF-IDF cru não tem:

- **saturação** — a 10ª ocorrência de "FNARC" vale muito menos que a 2ª
- **normalização por tamanho** — chunk longo não ganha só por ser longo

Termo raro pesa mais (IDF). Num corpus de PL, os termos raros são exatamente os
que identificam o documento: `FNARC`, `1502`, `2026`, `reassentamento`.

## O que mudou em relação ao `../new-model`

| | new-model | bm25 |
|---|---|---|
| Recuperação | cosseno sobre vetor | BM25 Okapi sobre texto |
| Modelo | `intfloat/multilingual-e5-base` | nenhum |
| Postgres | `pgvector/pgvector:pg16` | `postgres:16` |
| Coluna `embedding` | `VECTOR(768)` | não existe |
| Índice | HNSW no Postgres | lista tokenizada em RAM |
| Dependências | torch, sentence-transformers | `rank-bm25` (≈200 linhas sobre numpy) |
| Boot | ~1.1 GB de download | segundos |

O loader ficou trivial: lê o parquet, insere, sai. Sem passo de embedding. A
idempotência continua, agora chaveada em `(corpus_sha, retriever)` em vez de
`(corpus_sha, embed_model)`.

O índice BM25 é construído no boot do `document-api`, lendo os 183 chunks do
Postgres. Não é persistido — reconstruir custa milissegundos, e um índice em
disco seria estado a mais para sincronizar sem ganho nenhum nessa escala.

## BM25 pode devolver menos de k resultados

Chunk que não compartilha nenhum termo com a pergunta pontua 0 e **não** entra
na lista. O stack vetorial sempre devolve k linhas, porque cosseno é definido
para qualquer par — inclusive para os 5 chunks menos irrelevantes de um corpus
que não tem a resposta. BM25 pode dizer "não achei nada", e a precision recebe o
crédito por isso.

## Sem stemmer, sem stopwords

Tokenizador é `\w+` em minúsculas. Sem radicalização, sem lista de stopwords: o
IDF do próprio BM25 já joga `de`/`do`/`que` para perto de zero, e o objetivo
aqui é o baseline léxico cru, não o melhor BM25 possível.

O que viria depois, se recall em pergunta parafraseada virar o teto: stemmer
RSLP (nltk), ou o analyzer `portuguese` do OpenSearch, que traz stemmer +
stopwords + folding de acento prontos. Elasticsearch e OpenSearch usam BM25 como
ranking padrão desde o Lucene 6 — este stack é a mesma fórmula, sem o índice
invertido em disco e sem o analyzer.

## Rodando

```bash
cp .env.example .env   # ANTHROPIC_API_KEY, usado só pelo chat
docker compose up --build
```

| | week01 | new-model | bm25 |
|---|---|---|---|
| Chat | 3001 | 3011 | 3021 |
| Document manager | 3000 | 3010 | 3020 |
| `document-api` | 8000 | 8001 | 8002 |
| Postgres | 5432 | 5433 | 5434 |

Checagens sem subir nada:

```bash
docker compose run --rm --no-deps loader --self-check
docker compose run --rm --no-deps --entrypoint python document-api main.py --self-check
```

## Resultado (38 perguntas, k=5)

```bash
cd ../../eval
STACK=week03-bm25 docker compose run --rm eval --label "bm25 puro"
```

| Métrica | MiniLM 384d | e5-base 768d | **BM25** |
|---|---|---|---|
| hit rate | 0.684 | **0.921** | 0.868 |
| recall | 0.476 | **0.774** | 0.689 |
| precision | 0.158 | **0.284** | 0.242 |
| MRR | 0.393 | 0.690 | **0.700** |
| nDCG | 0.358 | **0.664** | 0.633 |

### Lendo isso

**BM25 sem modelo nenhum destrói o MiniLM.** +0.184 de hit rate, +0.213 de
recall, quase o dobro de MRR. Confirma o diagnóstico do `../new-model`: o MiniLM
não estava fazendo semântica em português, estava fazendo coincidência léxica —
e mal. BM25 faz coincidência léxica de propósito e faz melhor.

**BM25 chega perto do e5 e ganha em MRR.** O e5 acha mais chunks certos
(recall 0.774 vs 0.689), mas o BM25 põe o primeiro chunk certo mais perto do topo
(MRR 0.700 vs 0.690). Faz sentido: 29 das 38 perguntas citam o número do PL, e
número exato é onde o vetor borra e o BM25 não erra.

**Os erros quase não se sobrepõem.** É o argumento para o híbrido, e dá para
medir: das 38 perguntas, o e5 zera em 3 (`q003`, `q009`, `q012`) e o BM25 zera em
5 (`q001`, `q003`, `q024`, `q032`, `q034`). Só **uma** é falha dos dois.

Unindo as duas listas de 5, o recall teto sobe para **0.862**, contra 0.774 do e5
sozinho. Ou seja: os chunks que faltam ao e5 o BM25 já está trazendo — o que
falta é um jeito de ordenar a união. É exatamente o que RRF faz.

Próximo passo natural: fundir as duas listas com RRF e ver quanto desse 0.862 dá
para colher. As execuções estão em [`../../eval/runs`](../../eval/runs).
