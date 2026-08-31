#!/usr/bin/env python3
"""Build the eval dataset from source/.

  python build.py configs/legal-1200.yaml

  out/queries.parquet                 from questions-and-answers.json (config-independent)
  out/<config.name>/corpus.parquet    the chunks
  out/<config.name>/answers.parquet   which chunks hold each answer passage (qrels)
  out/<config.name>/manifest.json     what produced them
"""

import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd
import yaml

from chunkers import Config, chunk_dir
from resolve import resolve

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "source" / "pdfs"
QA = ROOT / "source" / "questions-and-answers.json"
OUT = ROOT / "out"


def load_config(path: Path) -> Config:
    cfg = Config(**yaml.safe_load(path.read_text(encoding="utf-8")))
    cfg.validate()
    return cfg


def load_qa() -> list[dict]:
    """Flatten questions-and-answers.json: one row per question."""
    qa = json.loads(QA.read_text(encoding="utf-8"))
    return [
        {"query_id": q["query_id"], "question": q["question"],
         "filename": doc["filename"], "answer": list(q["answer"])}
        for doc in qa["documents"]
        for q in doc["questions"]
    ]


def write_queries() -> Path:
    """Chunker-independent, so it lives at out/ root, shared by every build."""
    OUT.mkdir(exist_ok=True)
    rows = [{k: v for k, v in q.items() if k != "answer"} for q in load_qa()]
    path = OUT / "queries.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"{path.relative_to(ROOT)}: {len(rows)} queries")
    return path


def build(cfg: Config) -> Path:
    out = OUT / cfg.name
    out.mkdir(parents=True, exist_ok=True)

    corpus = chunk_dir(PDF_DIR, cfg)
    if not corpus:
        sys.exit(f"no PDFs in {PDF_DIR}")
    pd.DataFrame(corpus).to_parquet(out / "corpus.parquet", index=False)

    answers, unresolved = resolve(PDF_DIR, cfg, corpus, load_qa())
    if unresolved:
        msg = "\n  ".join(unresolved)
        raise ValueError(
            f"{len(unresolved)} answer passage(s) not found in the PDFs — "
            f"fix source/questions-and-answers.json (passages must be verbatim):\n  {msg}"
        )
    pd.DataFrame(answers).to_parquet(out / "answers.parquet", index=False)

    sha = hashlib.sha256()
    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        sha.update(pdf.name.encode() + pdf.read_bytes())

    def ver(pkg):
        try:
            return version(pkg)
        except PackageNotFoundError:
            return None

    sizes = [r["size"] for r in corpus]
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": asdict(cfg),
        "versions": {p: ver(p) for p in ("pypdf", "transformers", "pyarrow")},
        "counts": {
            "documents": len({r["filename"] for r in corpus}),
            "chunks": len(corpus),
            "size_min": min(sizes),
            "size_median": sorted(sizes)[len(sizes) // 2],
            "size_max": max(sizes),
            "queries": len({a["query_id"] for a in answers}),
            "qrels": len(answers),
        },
        "pdfs_sha256": sha.hexdigest(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (out / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
    print(f"{out.relative_to(ROOT)}: {manifest['counts']}")
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    write_queries()
    build(load_config(Path(sys.argv[1])))
