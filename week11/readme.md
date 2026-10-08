# week11 — base maior, 3 segmentos

| segmento | keyword na API | PDFs |
|---|---|---|
| `meio_ambiente` | meio ambiente | 114 (2× o week05, inclui os 57 originais) |
| `saude` | saúde | 114 |
| `educacao` | educação | 114 |

```bash
cd source
python download.py                # baixa até 114 PDFs por segmento (2026 → 2023)
python download.py --only saude   # só um segmento
```

- Saída: `source/pdfs/<segmento>/<TIPO>_<num>_<ano>.pdf` + `source/metadata.csv` (coluna `segmento`).
- Idempotente: o que já está em disco é pulado; rodar de novo retoma.
- Novo segmento = uma linha em `SEGMENTS`; tamanho = `TARGET`; anos = `YEARS`.
- Eval: `source/questions-and-answers.json`, mesmo formato do week05 + campo `segment` (o mesmo PDF pode estar em 2 segmentos). 342 docs, 688 perguntas (`q001`–`q688`), respostas = trechos literais do texto do PDF (extraído com pypdf).
- `out/queries.parquet` (688) e `out/description.parquet` (331 PDFs únicos, ~1.600 caracteres cada, pra camada 1 do week09/10). As perguntas não citam nº de PL; as descrições sim (lado do documento).
- Variações das perguntas (`?variants=`): somadas em `week10/services/chunks-api/rewrites.json` (5 por pergunta).
- Chunks: `cd week11 && docker compose up ui` → http://localhost:8765 (mesmo builder do week05, lendo `pdfs/<segmento>/`; PDF repetido em 2 segmentos vira chunk 1 vez, com `segments`). `also_in` no Q&A marca projetos-irmãos com o mesmo trecho como relevantes também.
