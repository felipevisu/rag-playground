# eval

Mede a qualidade do retrieval e guarda o histórico, para que "melhorei?" tenha
resposta em número e não em impressão.

Perguntas vêm de [`../dataset/out/queries.parquet`](../dataset) — não dependem
de chunker. Tudo o que depende vem de um **bucket** do [`../week04`](../week04):
a busca em `/api/buckets/<id>/search` e o gabarito em `/api/buckets/<id>/answers`.
O bucket subiu `corpus.parquet` + `answers.parquet` da mesma pasta
`dataset/out/<config>/`, então os chunk_ids batem por construção. Este diretório
não tem opinião sobre o que é um chunk certo — isso mora no dataset.

Cada run escolhe **um bucket**. Só aparecem buckets `ready` com gabarito.

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

O week04 precisa estar de pé (`cd ../week04 && docker compose up -d`), com pelo
menos um bucket `ready`.

### Pelo painel

```sh
docker compose up -d ui
```

→ **http://localhost:8080/**

Escolha o bucket, preencha rótulo (opcional — o padrão é
`<retriever> · <nome do bucket>`), nota e `k`, clique **▶ Rodar**. O `server.py` importa o
`run.py` e executa no próprio processo; a resposta só volta no fim (~1 min), e o
painel recarrega sozinho já com o run novo selecionado. Log ao vivo em
`docker compose logs -f ui`. Uma avaliação por vez — a segunda leva 409, porque
o `run_id` tem resolução de segundo e duas gravariam por cima uma da outra.

O `ui` fala com o `chunks-api`, então precisa da rede do week04 no ar.

### Pelo terminal

O serviço `eval` fica atrás de um profile de propósito: sem isso, um
`docker compose up` dispararia uma avaliação inteira sem querer. `run` funciona
normalmente.

```sh
docker compose run --rm eval --options     # buckets prontos com gabarito
docker compose run --rm eval --bucket d556057b --label "bm25 · legal-1200"
docker compose run --rm eval --bucket d556057b --k 10 --label "top-10"
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
  "bucket": "d556057b",
  "bucket_name": "legal-1200 bm25",
  "retriever": "bm25",
  "corpus_sha256": "9b09baec3732…",
  "n_queries": 38,
  "n_chunks": 241,
  "n_qrels": 132
}
```

Tudo vem do bucket. `type` (`single_chunk`/`multi_chunk`) é derivado do
gabarito — quantos chunks a resposta ocupa **neste** chunker — não declarado na
pergunta. Comparar dois runs cujo `corpus_sha256` difere compara chunker e
retriever ao mesmo tempo; o nome do bucket é o que diz qual config de chunker
foi.
