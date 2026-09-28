# RAG v8 — duas camadas (documento → chunk)

Mesmo stack do [week08](../week08) (BM25, vetorial, híbrido, rerank, multi-query).
A novidade: antes de buscar chunks, a busca pode escolher **documentos**.

```
"qual o prazo de validade da CNH para idosos?"
  camada 1: bucket de descrições (1 resumo por PDF)  → top-doc_top arquivos
  camada 2: bucket de chunks, só desses arquivos     → top-k   (→ rerank, se pedido)
```

## Por quê

Um chunk de 450 tokens raramente diz de que lei ele é; a descrição do documento
diz (tipo e número, tema, a quem se aplica, leis citadas). Filtrar por documento
primeiro tira da disputa chunks parecidos de projetos sobre outra coisa.
O risco: se a camada 1 erra o documento, a camada 2 não tem como acertar.

## Dois tipos de bucket

| kind | upload | o que é |
|---|---|---|
| `chunks` | `bucket.zip` do chunker (:8765) | como antes: corpus + gabarito + retriever |
| `descriptions` | `description.parquet` (`filename`, `description`) | um resumo por documento + **o retriever que você escolher** |

São independentes: um bucket de descrições indexado com `hybrid-e5-large`
serve de camada 1 para qualquer bucket de chunks (inclusive os já existentes),
desde que cubra todos os arquivos dele (senão 400: os chunks do arquivo sem
descrição ficariam inalcançáveis).

As descrições estão em `week05/out/description.parquet`: 57 resumos de ~1000
caracteres, escritos lendo cada PDF — tipo e número reais (PL, PLP, PDL, MP,
PRC), autor, tema, o que cria/altera, prazos, valores, leis citadas.

## Como está no código

`services/chunks-api/main.py`. A camada 1 é opção **da busca**, como `?rerank=`:

```
GET /api/buckets/<chunks>/search?q=...&k=5                         # uma camada
GET /api/buckets/<chunks>/search?q=...&k=5&docs=<descrições>       # duas camadas, top-DOC_TOP docs
GET /api/buckets/<chunks>/search?q=...&k=5&docs=<descrições>&doc_top=3
GET /api/buckets/<descrições>/search?q=...&k=5                     # só a camada 1: quais documentos?
```

- `POST /api/buckets/descriptions` (multipart `file`, `retriever`) cria o bucket
  `descrições · <retriever>`; a UI (:3000) manda `.parquet` pra cá e `.zip` pro import.
- `rank()`: `pick_documents` ranqueia os documentos com o retriever **do bucket
  de descrições**, e `rank_one(..., files=)` busca os chunks só desses arquivos,
  com o retriever do bucket de chunks.
- Na camada 1 o BM25 **e** o vetor leem a descrição (`hybrid-*` funde os dois por RRF).
  Testado BM25 no documento inteiro (chunks juntados por arquivo): ficou pior.
- Na camada 2 o índice BM25 continua o do corpus inteiro (IDF igual com ou sem
  filtro); o filtro só corta.
- Com `?variants=N` cada reescrita escolhe os próprios documentos antes do RRF.
- A resposta traz `documents` (a escolha da camada 1 para a pergunta original).

## Rodando

```bash
docker compose up --build       # ui :3000 · api :8000/docs · postgres :5432
```

Mesmo volume do week05–08: os buckets existentes viram `kind = chunks`.
Suba o `description.parquet` na UI com o retriever que quiser testar.

No painel do [`../eval`](../eval) (:8080) aparece o seletor **descrições** ao lado
do **chunks bucket**. Rode o mesmo bucket de chunks sem e com cada bucket de
descrições. O detalhe mostra, por pergunta, os documentos da camada 1 e marca
quando o documento certo ficou de fora. Pela CLI: `--docs <id>`. O run grava
`config.docs`, `docs_retriever` e `doc_top`. `DOC_TOP` no compose muda o padrão.

`python main.py` roda o self-check (inclui as duas camadas) sem banco.
