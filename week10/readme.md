# week10 — rerank que ajuda (ou pelo menos não atrapalha)

**Antes (week07–09):** o cross-encoder relia o top-30 e a ordem dele substituía a do
primeiro estágio. Com mMiniLM, perdia mais do que ganhava:

| run (k=7, 57 PDFs, descrições top-4, 5 variantes) | nDCG | recall |
|---|---|---|
| sem rerank | 0.871 | 1.000 |
| + mminilm-rerank | 0.817 | 0.963 |

Por pergunta: 7 melhoraram, 13 pioraram, quase sempre o chunk certo do 1º pro 2º,
ou o segundo chunk de uma pergunta multi-chunk empurrado pra fora.

## Três mudanças

**1. O reranker sabe de que PL é o chunk.** As perguntas dizem "o PL 182/2026";
o chunk quase nunca. O BM25 e a camada 1 usam isso, o cross-encoder não via.

```
antes:  (pergunta, "heading\ntexto")
agora:  (pergunta, "PL 182/2026\nheading\ntexto")
```

**2. O reranker vota, não manda.** A ordem dele entra num RRF com a do primeiro estágio:

```
score(chunk) = RERANK_KEEP / (60 + rank_1º_estágio) + 1 / (60 + rank_reranker)
```

`RERANK_KEEP=0` é o comportamento antigo (só o reranker). `1.0` = votos iguais.

**3. `qwen3-rerank`** (Qwen3-Reranker-0.6B, mesma família do melhor embedder daqui).
É um LLM que responde sim/não: pergunta e chunk vão dentro do prompt de chat dele, com
`max_length` 1024 pra o chunk inteiro caber; truncar cortaria o fim do prompt que pede a resposta.
Lento na CPU, como o bge.

## Rodando

```bash
docker compose up --build        # ui :3000 · api :8000/docs · mesmos buckets do week09
```

No eval (:8080), **k=5** (o padrão): com k=7 o recall já é 1.0 e o rerank só reordena;
com k=5 ele também decide quem fica de fora. Com rerank escolhido, o bloco Rerank mostra
**top** (`rerank_depth`) e **keep** (`rerank_keep`); os dois vão pro nome da run.

1. Sem rerank → baseline.
2. `mminilm-rerank`, keep **1** → mudanças 1 + 2.
3. `mminilm-rerank`, keep **0** → só a mudança 1.
4. `qwen3-rerank` e `bge-reranker`, top **15**, com o keep que venceu.

Na API, os mesmos valores por busca: `?rerank=qwen3-rerank&rerank_depth=15&rerank_keep=0`.
`RERANK_DEPTH`/`RERANK_KEEP` no compose só mudam o padrão.

## Rerank na GPU do Mac (opcional, ~10× mais rápido)

O Docker no Mac não tem GPU e o torch Linux dele não usa o Accelerate. Por pergunta (top 30, 5 variantes):

| | Docker | Mac GPU (MPS) |
|---|---|---|
| mminilm-rerank | 3.2 s | 0.4 s |
| bge-reranker | 37 s | 4.1 s |

Só o reranker sai do Docker; o resto fica igual. Mesmas notas (float32), runs comparáveis.

```bash
cd week10/rerank-server
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # uma vez
.venv/bin/python serve.py                                             # :8001, deixa aberto

# outro terminal, na raiz do repo: recria só a API apontando pro Mac
RERANK_URL=http://host.docker.internal:8001 docker compose up -d chunks-api
```

`curl localhost:8000/api/health` mostra `rerank_url`. Pra voltar: `docker compose up -d chunks-api` sem a variável.
Servidor fora do ar = a busca com rerank dá 502 com o motivo (não volta pra CPU calado).

## Reranker pago: Voyage

`voyage-rerank` (rerank-2.5) e `voyage-rerank-lite` (rerank-2.5-lite), chamados direto da API
(o `RERANK_URL` não se aplica). Uma run ≈ 38 × 30 chunks × ~600 tokens ≈ 700k tokens ≈ US$ 0.04
(e os 200M tokens grátis cobrem).

```bash
cp week10/.env.example week10/.env      # cole a VOYAGE_API_KEY
docker compose up -d chunks-api         # na raiz: recria a API com a chave
```

Sem chave, a busca com voyage dá 400 dizendo isso. Sem cartão cadastrado o limite de requisições
é baixo: a API espera e tenta de novo (429) até 3 vezes, então a run fica lenta mas não quebra.

## Embeddings pagos: Voyage

Retrievers `voyage-4-large`, `voyage-4` e os híbridos `hybrid-voyage-4-large`, `hybrid-voyage-4`
(BM25 + Voyage por RRF). Mesma chave do reranker. Pergunta e chunk recebem vetores diferentes
via `input_type` (query/document), não por prefixo de texto.

Em :3000, importe o mesmo `bucket.zip` escolhendo `hybrid-voyage-4-large`: vira um bucket novo,
os antigos não mudam. O corpus inteiro é < 1M tokens (~US$ 0.12 no large, dentro do grátis).
Pra camada 1 com Voyage, importe também o `description.parquet` com o mesmo retriever.

Editar `main.py` durante uma indexação a mata (`--reload` reinicia a API): espere o bucket ficar `ready`.
