"""Pre-generate the ?variants= rewrites: every eval question, 5 alternatives each,
written once by Claude into rewrites.json. The API only reads that file.

    python gen_rewrites.py [queries.parquet]     # default ../../../week05/out/queries.parquet

Questions already in rewrites.json are kept; only new ones are sent to Claude.
Needs ANTHROPIC_API_KEY (read from week09/.env).
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "rewrites.json"
MODEL = "claude-haiku-4-5"  # small task: the cheap model
N = 5

SYSTEM = (
    "Você reescreve perguntas para um buscador de documentos legislativos brasileiros "
    "(leis, projetos, pareceres). Não responda a pergunta."
)
PROMPT = (
    "Escreva {n} versões alternativas desta pergunta de busca, cada uma com outras palavras: "
    "sinônimos, o termo jurídico/técnico que um texto de lei usaria, uma versão mais específica, "
    "uma mais genérica. Mantenha números de leis, artigos e siglas.\n\n<pergunta>{q}</pergunta>"
)
SCHEMA = {"type": "object", "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
          "required": ["queries"], "additionalProperties": False}


def ask(client, q: str) -> list[str]:
    r = client.messages.create(
        model=MODEL, max_tokens=1000, system=SYSTEM,
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": PROMPT.format(n=N, q=q)}],
    )
    if r.stop_reason != "end_turn":
        raise RuntimeError(f"{MODEL} stopped with {r.stop_reason}: {q!r}")
    text = next(b.text for b in r.content if b.type == "text")
    return [v.strip() for v in json.loads(text)["queries"] if v.strip() and v.strip() != q][:N]


def main(src: Path) -> None:
    import anthropic
    import pyarrow.parquet as pq

    env = HERE.parent.parent / ".env"  # week09/.env; the shell wins over the file
    for line in env.read_text().splitlines() if env.exists() else []:
        key, sep, value = line.partition("=")
        if sep and value.strip():
            os.environ.setdefault(key.strip(), value.strip())

    questions = pq.read_table(src, columns=["query_id", "question"]).to_pylist()
    done = {r["question"]: r for r in json.loads(OUT.read_text())["questions"]} if OUT.exists() else {}
    todo = [q for q in questions if len(done.get(q["question"], {}).get("rewrites", [])) < N]
    print(f"{len(questions)} questions, {len(todo)} to generate with {MODEL}")

    client = anthropic.Anthropic()
    with ThreadPoolExecutor(8) as pool:
        for q, rewrites in zip(todo, pool.map(lambda q: ask(client, q["question"]), todo)):
            done[q["question"]] = {**q, "rewrites": rewrites}
            print(f"  {q['query_id']}: {len(rewrites)}")

    rows = [done[q["question"]] for q in questions]  # dataset order
    OUT.write_text(json.dumps({"model": MODEL, "questions": rows}, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[2] / "week05/out/queries.parquet")
