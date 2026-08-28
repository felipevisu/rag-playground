# RAG v3 — embedding multilíngue

Igual ao [`../week01`](../week01), com uma troca só: o modelo de embedding.

`all-MiniLM-L6-v2` é treinado **só em inglês**. O corpus é um punhado de Projetos
de Lei em português. Ele produzia vetor para `"Art. 3º Fica criado o FNARC"` do
mesmo jeito que produz para qualquer coisa — sem entender nada. Recuperação
funcionava por acidente léxico, não por significado.

| | week01 | week03 |
|---|---|---|
| Modelo | `sentence-transformers/all-MiniLM-L6-v2` | `intfloat/multilingual-e5-base` |
| Idiomas | inglês | 100, inclusive pt-BR |
| Dimensão | 384 | 768 |
| Prefixos | nenhum | `passage: ` ao indexar, `query: ` ao buscar |
| Tamanho | ~90 MB | ~1.1 GB |

Nada mais mudou. Mesmo dataset, mesmo schema, mesmos serviços, mesmo `chunk_id`
— então o gabarito do week02 continua apontando para as linhas certas e as duas
versões são comparáveis número a número.

## Os prefixos não são opcionais

e5 é assimétrico: foi treinado com o papel do texto escrito no próprio texto.
Documento indexado entra como `passage: …`, pergunta entra como `query: …`. Sem
isso os dois caem em regiões diferentes do espaço e a busca piora **em silêncio**
— nenhum erro, só resultado pior. Por isso o prefixo está grudado no `encode()`
dos dois lados ([`loader/load.py`](services/loader/load.py),
[`document-api/main.py`](services/document-api/main.py)) e não num parâmetro.

## 384 → 768

O `EMBED_DIM` do loader define a coluna `VECTOR(n)`, então a troca de modelo é
também uma troca de schema. Não tem migração: o loader percebe que o par
`(corpus, embed_model)` gravado em `dataset_load` mudou, derruba tudo e recarrega.
É a mesma idempotência do week01, funcionando como projetada.

## Rodando

Portas deslocadas para week01 e week03 subirem juntos e serem comparados:

| UI | week01 | week03 |
|---|---|---|
| Chat | http://localhost:3001 | http://localhost:3011 |
| Document manager | http://localhost:3000 | http://localhost:3010 |
| `document-api` | :8000 | :8001 |
| Postgres | :5432 | :5433 |

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
docker compose up --build
# Primeiro boot baixa ~1.1 GB de modelo.
```

## Medindo se melhorou

O [`../eval`](../eval) aponta para um stack de cada vez, via `STACK`:

```bash
cd ../eval
docker compose run --rm eval --label "minilm 384d"                   # week01
STACK=week03 docker compose run --rm eval --label "e5-base 768d"     # week03
docker compose run --rm eval --list                                  # lado a lado
```

Os dois stacks precisam estar de pé. As métricas — hit rate, recall, precision,
MRR, nDCG — estão explicadas no [`../week02`](../week02).

### Resultado (38 perguntas, k=5)

| Métrica | MiniLM | e5-base | |
|---|---|---|---|
| hit rate | 0.684 | **0.921** | +0.237 |
| recall | 0.476 | **0.774** | +0.298 |
| precision | 0.158 | **0.284** | +0.126 |
| MRR | 0.393 | **0.690** | +0.296 |
| nDCG | 0.358 | **0.664** | +0.306 |

MRR quase dobrou: não é só que o chunk certo aparece mais, é que ele aparece
mais perto do topo. As duas execuções estão em [`../eval/runs`](../eval/runs).
