# jev-rerank — hybrid com e sem Jev no top 10

O RAG do week10 reduzido a um benchmark de terminal: sem banco, sem UI, tudo em memória.

- **Dados** (`data/`, copiados do week05): `sentence-512t-multilingual-e5-large`, 279 chunks, 38 perguntas.
- **Primeiro estágio:** `hybrid-qwen3-0.6b` = BM25 (stemmer pt) + Qwen3-Embedding-0.6B, top 50 de cada, fundidos por RRF.
- **Rerank:** o Jev lê os top 10, um chunk por chamada, e responde *"Does `passage` contain
  information that answers `query`?"*. A nota é P(sim). O chunk vai com o nome do PL na frente.

Duas runs com os mesmos candidatos: `hybrid (sem jev)` mantém a ordem do primeiro estágio;
`hybrid + jev` reordena o top 10 pela nota do Jev.

## Rodando

```bash
cd pocs/jev-rerank
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # uma vez

export TYPESAFE_API_KEY=sk-...        # console.typesafe.ai/keys (ou a do week10/.env); sem ela, só a run sem Jev
.venv/bin/python bench.py             # k=5, Jev no top 10
.venv/bin/python bench.py --k 7 --depth 15
.venv/bin/python bench.py --check     # self-check, sem modelo e sem rede
```

Os embeddings são recalculados a cada execução (~2 min na GPU do Mac). Uma run com Jev custa
≈ 38 × 10 × ~600 tokens ≈ 230k tokens ≈ US$ 0.01.

O relatório traz nDCG, recall, MRR e hit@k das duas runs e quais perguntas melhoraram ou
pioraram no nDCG com o Jev.

## Resultados (2026-10-06, k=5, Jev no top 10, jev-1.13.0)

| run | nDCG | recall | MRR | hit |
|---|---|---|---|---|
| hybrid (sem jev) | 0.773 | 0.833 | 0.818 | 0.947 |
| **hybrid + jev** | **0.911** | **0.908** | **0.961** | **0.974** |

- **Com Jev:** 15 perguntas melhoraram e **nenhuma piorou** (nDCG). O maior ganho foi em
  "O que o PL 1502/2026 define como racismo ambiental?" (+1.00: o chunk certo estava fora do top 5 e foi pro 1º).
- **Custo e tempo:** 364k tokens de entrada (≈ US$ 0.015) e 21 s para as 38 perguntas (380 chamadas).
  Os embeddings levaram mais 110 s.
- **Teto:** o recall@5 fica limitado ao que o hybrid já trouxe no top 10. O Jev só reordena;
  um chunk certo que não está no top 10 continua de fora.

Sem banco, a expansão de siglas do BM25 e as camadas do week10 (descrições, variantes) ficam de fora.
Por isso os números não batem 1:1 com o eval.
