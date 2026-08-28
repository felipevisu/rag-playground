# BM25 from scratch

A dependency-free BM25 in Python, built step by step, so you can open a score
and see where every decimal came from.

```bash
python bm25.py
```

---

## The corpus

Six chunks of a fake backup-service manual, in Portuguese (so stemming has real
work to do). Italics = what it says.

1. *nightly schedule, job IDs* — "O serviço de backup executa automaticamente todos os dias às 3h da manhã. Cada execução cria um job com identificador numérico único."
2. *a failed job, disk full* — "Quando um job falha, o código de erro fica no log. O erro ERR_DISK_FULL indica que o volume de destino ficou sem espaço em disco durante a cópia."
3. *restoring a backup* — "Para restaurar um backup, use o comando restore informando o identificador do job. A restauração sobrescreve os arquivos existentes no destino."
4. *a different error: expired credentials* — "O erro ERR_AUTH_DENIED aparece quando as credenciais do serviço expiraram. Renove o token de acesso no painel e execute o job novamente."
5. *monitoring free disk space* — "O espaço em disco do servidor deve ser monitorado semanalmente. Recomenda-se manter pelo menos 20% do volume livre para os backups."
6. *turning the nightly schedule off* — "Para desativar a rotina noturna, edite o agendamento no painel e remova a janela de execução. Nenhuma cópia será criada até reativar o agendamento."

---

## Step 1 — Tokenize

```python
def tokenizar(texto):
    return re.findall(r"\w+", texto.lower())
```

```
[1] 22  ['o', 'serviço', 'de', 'backup', 'executa', 'automaticamente', 'todos', 'os',
         'dias', 'às', '3h', 'da', 'manhã', 'cada', 'execução', 'cria', 'um', 'job',
         'com', 'identificador', 'numérico', 'único']

[2] 28  ['quando', 'um', 'job', 'falha', 'o', 'código', 'de', 'erro', 'fica', 'no',
         'log', 'o', 'erro', 'err_disk_full', 'indica', 'que', 'o', 'volume', 'de',
         'destino', 'ficou', 'sem', 'espaço', 'em', 'disco', 'durante', 'a', 'cópia']

[6] 24  ['para', 'desativar', 'a', 'rotina', 'noturna', 'edite', 'o', 'agendamento',
         'no', 'painel', 'e', 'remova', 'a', 'janela', 'de', 'execução', 'nenhuma',
         'cópia', 'será', 'criada', 'até', 'reativar', 'o', 'agendamento']
```

- `\w` is unicode-aware, so `serviço` survives; it includes `_`, so
  `err_disk_full` stays one token.
- Word order is now gone. "job falha" and "falha job" are identical to BM25.

---

## Step 2 — Stopwords + stemming

```python
def tokenizar(texto):
    tokens = []
    for token in re.findall(r"\w+", texto.lower()):
        if token in STOPWORDS:          # o, de, um, para, com ...
            continue
        tokens.append(stem(token))      # longest suffix first, keep >= 4 chars
    return tokens
```

```
falhou   -> falh      backups  -> backup    err_disk_full -> err_disk_full
falha    -> falh      disco    -> disc      3h            -> 3h
falharam -> falh      noturna  -> noturn    erro          -> erro
```

Same six chunks, analyzed:

```
[1] 15  ['serviç', 'backup', 'execut', 'automaticament', 'todo', 'dias', '3h',
         'manhã', 'cada', 'execuçã', 'cria', 'job', 'identific', 'numéric', 'únic']

[2] 16  ['job', 'falh', 'códig', 'erro', 'fica', 'log', 'erro', 'err_disk_full',
         'indic', 'volum', 'destin', 'ficou', 'espaç', 'disc', 'durant', 'cópi']

[3] 13  ['restaur', 'backup', 'use', 'comand', 'restor', 'inform', 'identific',
         'job', 'restauraçã', 'sobrescrev', 'arquiv', 'existent', 'destin']

[4] 13  ['erro', 'err_auth_denied', 'aparec', 'credenciai', 'serviç', 'expir',
         'renov', 'token', 'acess', 'painel', 'execut', 'job', 'novament']

[5] 13  ['espaç', 'disc', 'servidor', 'deve', 'monitor', 'semanalment',
         'recomend', 'mant', 'meno', '20', 'volum', 'livr', 'backup']

[6] 14  ['desativ', 'rotin', 'noturn', 'edit', 'agend', 'painel', 'remov',
         'janel', 'execuçã', 'nenhum', 'cópi', 'criad', 'reativ', 'agend']
```

| | raw | analyzed |
|---|---|---|
| lengths | `22 28 21 22 22 24` | `15 16 13 13 13 14` |
| average | 23.2 | **14.0** |
| distinct terms | 90 | **66** |

- Stopwords were 40% of the corpus.
- Stems needn't be words (`credenciai`, `meno`) — only *consistent* between
  query and document.
- The stemmer already fails: `execução → execuçã` but `execute → execut`.
- The same `tokenizar` must run at index time **and** query time.

---

## Step 3 — Inverted index

```python
for doc_id, chunk in enumerate(CHUNKS):
    tokens = tokenizar(chunk)
    DOC_LEN.append(len(tokens))
    for termo, freq in Counter(tokens).items():
        POSTINGS[termo][doc_id] = freq
```

```python
"job"           -> {0: 1, 1: 1, 2: 1, 3: 1}   # 4 docs, once each
"backup"        -> {0: 1, 2: 1, 4: 1}
"erro"          -> {1: 2, 3: 1}               # twice in doc 1
"disc"          -> {1: 1, 4: 1}
"falh"          -> {1: 1}
"agend"         -> {5: 2}
"cancel"        -> None                       # not in the corpus

DOC_LEN = [15, 16, 13, 13, 13, 14]   N = 6   AVGDL = 14.0
```

`tf` is a dict lookup; `df` is `len()` of the postings list. A query only ever
opens the entries for its own terms.

---

## Step 4 — IDF

```python
idf(t) = math.log(1 + (N - df + 0.5) / (df + 0.5))
```

| `df` | IDF | without the `1 +` | terms |
|---:|---:|---:|---:|
| 1 | **1.5404** | 1.2993 | 53 |
| 2 | **1.0296** | 0.5878 | 11 |
| 3 | **0.6931** | 0.0000 | 1 |
| 4 | **0.4418** | −0.5878 | 1 |

```
falh, err_disk_full, desativ, agend ...  df=1   1.5404
disc, erro, espaç, volum, painel ...     df=2   1.0296
backup                                   df=3   0.6931
job                                      df=4   0.4418
```

- `falh` is worth **3.5x** `job`. Nobody encoded that — it fell out of counting.
- Column 3 is why the `1 +` exists: past `df = N/2` the original goes
  **negative**, penalizing a document for containing your search term.
- Natural log. `log10` runs fine and is silently wrong.

---

## Step 5 — The formula

```
                             f(t,D) · (k1 + 1)
score(D,Q) = Σ  IDF(t) · ─────────────────────────────────
            t∈Q          f(t,D) + k1 · (1 - b + b·|D|/avgdl)
```

```python
K1 = 1.5    # saturation
B  = 0.75   # length normalization

def contribuicao(termo, doc_id, freq):
    norma = K1 * (1 - B + B * DOC_LEN[doc_id] / AVGDL)
    saturacao = (freq * (K1 + 1)) / (freq + norma)
    return idf(termo) * saturacao
```

`explain("por que o job falhou com erro de disco cheio")`:

```
chunk 2  (16 tokens)  score = 4.2363
    falh    tf=1  idf=1.5404  x sat=0.9396  = +1.4474
    erro    tf=2  idf=1.0296  x sat=1.3659  = +1.4063
    disc    tf=1  idf=1.0296  x sat=0.9396  = +0.9674
    job     tf=1  idf=0.4418  x sat=0.9396  = +0.4151

chunk 4  (13 tokens)  score = 1.5203
    erro    tf=1  idf=1.0296  x sat=1.0332  = +1.0638
    job     tf=1  idf=0.4418  x sat=1.0332  = +0.4565
```

- **The score is a sum.** 1.4474 + 1.4063 + 0.9674 + 0.4151 = 4.2363. No
  comprehension, just accumulated evidence.
- **Saturation:** `erro` at `tf=2` got `sat=1.3659` vs `0.9396` at `tf=1` —
  1.45x for double the frequency, not 2x. Ceiling is `k1 + 1 = 2.5`.
- **Length:** `erro` earns more in chunk 4 (13 tokens) than in chunk 2 (16).
  Same term, same `tf`.

---

## Step 6 — Search

Loop over **terms**, walk each postings list, accumulate per document
(*term-at-a-time*, the way Lucene does it):

```python
for termo in dict.fromkeys(tokenizar(query)):
    for doc_id, freq in POSTINGS.get(termo, {}).items():
        scores[doc_id] += contribuicao(termo, doc_id, freq)
```

```
query : "por que o job falhou com erro de disco cheio"
tokens: ['job', 'falh', 'erro', 'disc', 'chei']
visits: 5 of 6 documents

  1. chunk 2   4.2363  ####################  Quando um job falha, o código de erro...
  2. chunk 4   1.5203  #######               O erro ERR_AUTH_DENIED aparece quando...
  3. chunk 5   1.0638  #####                 O espaço em disco do servidor deve ser...
  4. chunk 3   0.4565  ##                    Para restaurar um backup, use o comando...
  5. chunk 1   0.4281  ##                    O serviço de backup executa automatica...
  -. chunk 6   0.0000                        Para desativar a rotina noturna, edite...
```

Documents with no query term are never visited — not even to be skipped.

### What the analyzer was worth

Same query, same formula, same `k1`/`b` — only the analyzer changes:

| config | 1st | 2nd | margin |
|---|---|---|---|
| raw tokens | 4.8408 | 2.5561 | 1.89x — *and 2nd is wrong* |
| + stopwords | 2.7889 | 1.5203 | 1.83x |
| + stopwords + stemming | **4.2363** | 1.5203 | **2.79x** |

On raw tokens, chunk 1 placed 2nd on the strength of the word **"com"**
(*with*), which happens to sit in exactly 1 of 6 documents and so earned
`idf=1.5404` — 62% of that document's score. In a small corpus IDF doesn't
protect you from stopwords; it promotes them.

---

## Step 7 — Where it breaks

*"como cancelar o backup automático da madrugada"* — how do I cancel the
automatic overnight backup. The answer is chunk 6: *"to **disable** the
**nightly routine**, edit the **schedule**"*.

```
tokens: ['cancel', 'backup', 'automátic', 'madrug']
visits: 3 of 6 documents

  1. chunk 3   0.7162  ####################  Para restaurar um backup...   ← RESTORING
  2. chunk 5   0.7162  ####################  O espaço em disco...          ← disk space
  3. chunk 1   0.6716  ###################   O serviço de backup...
  -. chunk 2   0.0000
  -. chunk 4   0.0000
  -. chunk 6   0.0000                        Para desativar a rotina...    ← THE ANSWER
```

Nothing survives the crossing: `cancelar→desativar`, `madrugada→noturna`,
`automático→agendamento`.

**Not a tuning problem.** 5 values of `k1` × 5 of `b` = 25 cells, all 0.0000.
They live inside the saturation factor, computed only where `tf > 0` — they
redistribute weight among matched terms, they can't invent one.

**Not the algorithm either.** Same intent, the document's own words:

```
query : "desativar agendamento noturno"
visits: 1 of 6 documents

  1. chunk 6   5.2815  ####################  Para desativar a rotina noturna...
  -. all others 0.0000
```

**5.2815** — the highest score in the project, total separation.

The failure modes are *opposite*: query 1 nails `ERR_DISK_FULL` where a dense
retriever would return three vaguely error-ish chunks; query 2 is the reverse.
That orthogonality is why hybrid retrieval with RRF beats either alone.

---

## Still naive

- **Final `sort` orders every candidate** to show 10. Should be a size-`k` heap.
- **No skipping.** Lucene uses WAND / block-max to skip whole postings blocks
  whose ceiling can't beat the current top-k. That's the line where you stop
  writing your own and use Lucene, Tantivy or `bm25s`.
- **Hand-rolled stemmer.** NLTK's `RSLPStemmer` is strictly better.

## Next

Add dense retrieval, fuse with Reciprocal Rank Fusion (`Σ 1/(60 + rank)`).
That's what finally lifts chunk 6 to the top.