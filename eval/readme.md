# eval

Mede a qualidade do retrieval e guarda o histórico, para que "melhorei?" tenha
resposta em número e não em impressão.

Perguntas e gabarito vêm de [`../dataset/out`](../dataset). Os resultados vêm do
`document-api` do week01. Este diretório não tem opinião sobre o que é um chunk
certo — isso mora no dataset.

```
run.py            roda a avaliação e grava em runs/
server.py         serve o painel e atende POST /api/run
index.html        painel: botão de rodar, histórico, detalhe por pergunta
runs/
  <run_id>.json   uma execução completa (config, métricas, por pergunta)
  index.json      resumo de todas, regerado a cada run
```

Os arquivos JSON **são** o banco: versionados no git (a métrica muda no mesmo
diff do commit que causou a mudança), editáveis à mão para adicionar notas, e
lidos pelo `index.html` direto do navegador.

## Rodando

O week01 precisa estar de pé (`cd ../week01 && docker compose up -d`).

### Pelo painel

```sh
docker compose up -d ui
```

→ **http://localhost:8080/**

Preencha rótulo, nota e `k`, clique **▶ Rodar**. O `server.py` importa o
`run.py` e executa no próprio processo; a resposta só volta no fim (~1 min), e o
painel recarrega sozinho já com o run novo selecionado. Log ao vivo em
`docker compose logs -f ui`. Uma avaliação por vez — a segunda leva 409, porque
o `run_id` tem resolução de segundo e duas gravariam por cima uma da outra.

O `ui` fala com o `document-api`, então precisa da rede do week01 no ar.

### Pelo terminal

O serviço `eval` fica atrás de um profile de propósito: sem isso, um
`docker compose up` dispararia uma avaliação inteira sem querer. `run` funciona
normalmente.

```sh
docker compose run --rm eval --label "baseline minilm k=5"
docker compose run --rm eval --k 10 --label "top-10" --note "mais contexto ajuda?"
docker compose run --rm eval --list        # histórico no terminal
docker compose run --rm eval --self-check  # asserts das métricas, sem API
```

Cada run imprime o delta contra o anterior:

```
vs 2026-08-28T12-22-43 (baseline minilm k=5):
  hit_rate   0.684 ↑ 0.737  (+0.053)
  precision  0.158 ↓ 0.097  (-0.061)
```

## Notas

`notes` é campo livre em cada `runs/<run_id>.json`. Passe `--note` na hora de
rodar, ou edite o arquivo depois e rode `--reindex` para o `index.json` e o
relatório acompanharem.

## Métricas

Todas macro-médias sobre as 38 perguntas — cada pergunta pesa igual,
independente de quantos chunks ela precisa.

| Métrica | Pergunta que responde |
|---|---|
| Hit Rate | trouxe **pelo menos um** chunk certo? |
| Recall | trouxe **quantos %** dos chunks certos? |
| Precision | dos que trouxe, **quantos %** prestavam? |
| MRR | o chunk certo veio **no topo** ou lá embaixo? |
| nDCG | combina "achou" + "posição" numa nota só |

`--self-check` roda os dois exemplos documentados em
[`../week02/readme.md`](../week02/readme.md) com os números publicados lá, mais
os casos de borda: nada encontrado, tudo no topo, IDCG limitado por k, e
resultado vazio (divisão por zero).

Cada run também quebra as métricas por tipo de pergunta (`single_chunk` /
`multi_chunk`) — as duas costumam se mover em direções diferentes, e a média
sozinha esconde isso.

## O que fica registrado

Cada run grava a configuração que o produziu, para que uma métrica diferente
possa ser atribuída em vez de adivinhada:

```json
"config": {
  "k": 5,
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "corpus_sha256": "1e05e97f4479…",
  "chunker": { "max_tokens": 512, "merge_peers": true, … },
  "n_queries": 38,
  "n_chunks": 183
}
```

`corpus_sha256` e `chunker` vêm do `manifest.json` do dataset; o modelo vem do
`/api/health` do `document-api`. Comparar dois runs cujo `corpus_sha256` difere
compara duas coisas ao mesmo tempo.
