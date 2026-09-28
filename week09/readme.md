# week09 — duas camadas (documento → chunk)

**Antes:** a busca olha todos os chunks de todos os PDFs. **Agora:** primeiro escolhe
os PDFs certos pelo resumo, depois busca só nos chunks deles.

```
"Que tipos de denúncia o 'Disque Animal' recebe ...?"
  camada 1: resumos dos 57 PDFs     → PL 561/2026 e mais 4
  camada 2: chunks só desses 5 PDFs → top-5
```

Um chunk de 450 tokens raramente diz de que projeto ele é; o resumo diz.
Risco: se a camada 1 erra o PDF, a camada 2 não tem como acertar.

## Dois tipos de bucket

| tipo | sobe | |
|---|---|---|
| chunks | `bucket.zip` do week05 | como antes |
| descrições | `week05/out/description.parquet` | 1 resumo por PDF, com o retriever que você escolher |

Na busca, liga um no outro com `?docs=`:

```
GET /api/buckets/<chunks>/search?q=...&k=5                        # uma camada
GET /api/buckets/<chunks>/search?q=...&k=5&docs=<descrições>      # duas camadas, top-5 PDFs
GET /api/buckets/<chunks>/search?q=...&docs=<descrições>&doc_top=3
GET /api/buckets/<descrições>/search?q=...                        # só a camada 1: quais PDFs?
```

- BM25 e vetor da camada 1 leem o resumo (testado BM25 no PDF inteiro: ficou pior).
- Qualquer bucket de chunks usa qualquer bucket de descrições, desde que cubra todos os PDFs dele.
- Com `variants`, cada reescrita escolhe os próprios PDFs.

## Rodando

```bash
docker compose up --build        # ui :3000 · api :8000/docs
```

1. Em :3000, importe `description.parquet` (ex. `hybrid-e5-large`). `.zip` vira chunks, `.parquet` vira descrições.
2. No eval (:8080): **chunks bucket** + **descrições** → ▶ Rodar. Rode também sem descrições.
3. O detalhe mostra, por pergunta, os PDFs escolhidos e avisa "documento certo fora".

`DOC_TOP` no compose muda o padrão (menor = mais preciso, mais risco).
