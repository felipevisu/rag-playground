# rag-playground

Um RAG sobre Projetos de Lei da Câmara, melhorado uma semana de cada vez.
Cada semana muda **uma** coisa e mede se ficou melhor.

```
pergunta → [camada 1: documentos] → [busca de chunks] → [rerank] → top-5
            week09                   week01·03·06        week07·10
            + reescritas da pergunta (week08) em todas as etapas
```

## A evolução

| semana | o que mudou | exemplo |
|---|---|---|
| [01](week01) | **RAG básico**: chunks → vetores (MiniLM) no pgvector → Claude responde citando a fonte | "O que o PL 1502/2026 define como racismo ambiental?" → resposta + chunks usados |
| [02](week02) | **Métricas**: 38 perguntas com gabarito, 5 notas (hit rate, recall, precision, MRR, nDCG) | chunk certo na 3ª posição → hit 1, MRR 0.33 |
| [03](week03) | **Dois braços**: embedding multilíngue (e5) × BM25 (palavra exata) | `FNARC` o BM25 acha, "cachorro de rua" o embedding acha |
| [04](week04) | **Buckets**: sobe um corpus, escolhe o retriever, ganha um endpoint | mesmo corpus em `bm25` e `e5-large` = dois buckets, comparáveis |
| [05](week05) | **Dataset**: 57 PDFs, 4 chunkers, gabarito resolvido por chunk, contexto por Claude | `sentence-450t` × `legal-1200c` → dois `bucket.zip` |
| [06](week06) | **Híbrido**: BM25 + vetor fundidos por posição (RRF), BM25 em português | chunk 1º no BM25 e 3º no vetor vence quem só um viu |
| [07](week07) | **Rerank**: top-30 relido por um cross-encoder junto com a pergunta | `?rerank=mminilm-rerank` |
| [08](week08) | **Multi-query**: a pergunta vira 5 versões (Claude), cada uma busca, RRF | "a quais órgãos são encaminhadas?" → também "destinatários administrativos" |
| [09](week09) | **Duas camadas**: um resumo por documento escolhe os PDFs, depois os chunks só deles | `?docs=<descrições>` → top-5 PDFs → chunks |
| [10](week10) | **Rerank melhor**: reranker vê o nome do PL, vota com o 1º estágio (RRF), + Qwen3-Reranker | `?rerank=qwen3-rerank`, `RERANK_KEEP=1.0` |

O que cada semana resolve da anterior:

- **01 → 02**: "parece bom" vira número.
- **02 → 03**: o embedding erra termo exato; o BM25 erra sinônimo. Erram em perguntas diferentes.
- **03 → 04**: comparar deixa de ser subir outro stack; vira criar outro bucket.
- **04 → 05**: o chunker vira variável, não constante.
- **05 → 06**: em vez de escolher entre BM25 e vetor, usa os dois.
- **06 → 07**: o primeiro estágio acha, o segundo ordena.
- **07 → 08**: o usuário não fala como a lei; várias formulações cobrem mais vocabulário.
- **08 → 09**: chunk não sabe de que lei ele é; o resumo do documento sabe.
- **09 → 10**: o reranker sozinho reembaralhava um top-5 que já era bom; agora sabe o PL e só vota.

## Rodando

```sh
docker compose up --build
```

| | URL |
|---|---|
| buckets (week10) | http://localhost:3000 |
| API | http://localhost:8000/docs |
| eval | http://localhost:8080 |
| dataset / chunkers | http://localhost:8765 |

## Fluxo em 4 passos

1. **Chunkar** em :8765: escolha uma config, baixe o `bucket.zip`.
2. **Indexar** em :3000: importe o zip escolhendo o retriever (`bm25`, `e5-large`, `hybrid-bge-m3`…).
   Opcional: importe `week05/out/description.parquet` como bucket de descrições.
3. **Buscar**:
   ```
   GET /api/buckets/<id>/search?q=quem pode adotar animais comunitários?&k=5
       &rerank=mminilm-rerank     # week07
       &variants=4                # week08
       &docs=<descrições>         # week09
   ```
4. **Medir** em :8080: chunks bucket + descrições + rerank + variantes → ▶ Rodar.
   Cada run vira um JSON em `eval/runs/`, versionado.

## BM25 × embedding, num exemplo

```
"São fontes de recursos do FNARC, sem prejuízo de outras previstas em lei"

BM25       → ['são', 'fontes', 'de', 'recursos', 'do', 'fnarc', ...]  # conta palavras; 'fnarc' é raro, pesa muito
embedding  → [-0.013, 0.005, 0.005, -0.010, ...]                      # 1024 números: "sentido", não palavras
```

Detalhes de cada semana no `readme.md` da pasta.
