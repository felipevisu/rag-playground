#!/usr/bin/env python3
"""Run the retrieval eval and append the result to runs/.

Questions and ground truth come from dataset/out/*.parquet. Retrieval comes from
week01's document-api. Nothing here holds an opinion about what a good chunk is
-- that lives in the dataset.

Each run is one JSON file in runs/, plus a regenerated runs/index.json holding
the summaries. The files are the database: git-diffable, hand-editable, and
readable by index.html without a database.

  python run.py --label "baseline minilm k=5"
  python run.py --k 10 --label "top-10" --note "does more context help?"
  python run.py --list          # history, newest first
  python run.py --reindex       # rebuild index.json after editing notes by hand
  python run.py --self-check    # metric asserts, no API and no dataset needed
"""

import argparse
import json
import math
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
DATASET = Path(os.environ.get("DATASET_DIR", HERE.parent / "dataset" / "out"))
API = os.environ.get("API_URL", "http://localhost:8000")

METRICS = ("hit_rate", "recall", "precision", "mrr", "ndcg")


# ── metrics ────────────────────────────────────────────────────────────────

def gain(rank: int) -> float:
    """Binary DCG gain. Deeper rank, less it is worth."""
    return 1 / math.log2(rank + 1)


def score(expected: list[str], retrieved: list[str], k: int) -> dict:
    """Metrics for one query. `retrieved` is chunk_ids in rank order."""
    k_eff = len(retrieved) or k
    ranks = sorted(retrieved.index(c) + 1 for c in expected if c in retrieved)

    dcg = sum(gain(r) for r in ranks)
    # IDCG: the same expected chunks stacked at the top, capped by how many
    # results the search was even allowed to return.
    idcg = sum(gain(i + 1) for i in range(min(len(expected), k_eff)))

    return {
        "k": k_eff,
        "hits": len(ranks),
        "total": len(expected),
        "best_rank": ranks[0] if ranks else None,
        "hit_rate": 1.0 if ranks else 0.0,
        "recall": len(ranks) / len(expected) if expected else 0.0,
        "precision": len(ranks) / k_eff if k_eff else 0.0,
        "mrr": 1 / ranks[0] if ranks else 0.0,
        "ndcg": dcg / idcg if idcg else 0.0,
        "dcg": dcg,
        "idcg": idcg,
        "verdict": (
            "pass" if len(ranks) == len(expected)
            else "partial" if ranks
            else "fail"
        ),
    }


def mean(rows: list[dict], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


def aggregate(rows: list[dict]) -> dict:
    """Macro-average: every query counts the same, regardless of chunk count."""
    return {m: round(mean(rows, m), 4) for m in METRICS}


# ── dataset + api ──────────────────────────────────────────────────────────

def load_dataset() -> tuple[list[dict], dict, dict]:
    import pyarrow.parquet as pq

    missing = [f for f in ("queries.parquet", "answers.parquet") if not (DATASET / f).exists()]
    if missing:
        sys.exit(
            f"{', '.join(missing)} not found in {DATASET}.\n"
            f"Build the dataset first:  cd dataset && docker compose run --rm build"
        )

    queries = pq.read_table(DATASET / "queries.parquet").to_pylist()
    answers = pq.read_table(DATASET / "answers.parquet").to_pylist()

    qrels: dict[str, list[str]] = {}
    for a in answers:
        qrels.setdefault(a["query_id"], []).append(a["chunk_id"])

    manifest_path = DATASET / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    return queries, qrels, manifest


def search(api: str, question: str, k: int) -> list[dict]:
    url = f"{api}/api/search?" + urllib.parse.urlencode({"q": question, "k": k})
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())["results"]


def api_info(api: str) -> dict:
    try:
        with urllib.request.urlopen(f"{api}/api/health", timeout=10) as r:
            return json.loads(r.read())
    except urllib.error.URLError as e:
        sys.exit(f"document-api unreachable at {api}: {e}\nIs week01 up?")


# ── run ────────────────────────────────────────────────────────────────────

def run(label: str, note: str, k: int, api: str) -> dict:
    api = api.rstrip("/")
    health = api_info(api)
    queries, qrels, manifest = load_dataset()

    print(f"{len(queries)} queries · k={k} · {api}")
    per_query, failures = [], 0
    for i, q in enumerate(queries, 1):
        expected = qrels.get(q["query_id"], [])
        try:
            results = search(api, q["question"], k)
        except (urllib.error.URLError, TimeoutError) as e:
            failures += 1
            print(f"  [{i}/{len(queries)}] {q['query_id']} FAILED: {e}")
            results = []
        retrieved = [r["id"] for r in results]
        m = score(expected, retrieved, k)
        per_query.append({
            "query_id": q["query_id"],
            "question": q["question"],
            "type": q["type"],
            "filename": q["filename"],
            "expected": expected,
            "retrieved": [
                {"chunk_id": r["id"], "rank": n, "filename": r["filename"],
                 "chunk_index": r["chunk_index"], "similarity": round(r["similarity"], 4),
                 "expected": r["id"] in expected}
                for n, r in enumerate(results, 1)
            ],
            **m,
        })
        print(f"  [{i}/{len(queries)}] {q['query_id']} {m['verdict']:<7} "
              f"recall {m['recall']:.2f} ndcg {m['ndcg']:.2f}")

    if failures:
        print(f"\n{failures} query/queries failed against the API — "
              f"the metrics below are not comparable to a clean run.", file=sys.stderr)

    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    single = [r for r in per_query if r["type"] == "single_chunk"]
    multi = [r for r in per_query if r["type"] == "multi_chunk"]

    payload = {
        "run_id": run_id,
        "label": label or f"k={k}",
        "notes": note,
        "config": {
            "k": k,
            "api": api,
            "embedding_model": health.get("embedding_model", "unknown"),
            "corpus_sha256": manifest.get("corpus_sha256", "unknown"),
            "chunker": manifest.get("chunker", {}),
            "n_queries": len(queries),
            "n_chunks": manifest.get("counts", {}).get("chunks"),
        },
        "metrics": aggregate(per_query),
        "metrics_by_type": {
            "single_chunk": aggregate(single),
            "multi_chunk": aggregate(multi),
        },
        "failures": failures,
        "per_query": per_query,
    }

    RUNS.mkdir(exist_ok=True)
    path = RUNS / f"{run_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    reindex()

    print(f"\nwrote runs/{path.name}")
    print("  " + "  ".join(f"{m} {payload['metrics'][m]:.3f}" for m in METRICS))
    compare_to_previous(payload)
    return payload


def compare_to_previous(current: dict) -> None:
    """The point of keeping history is seeing the delta without opening a file."""
    runs = load_runs()
    previous = [r for r in runs if r["run_id"] != current["run_id"]]
    if not previous:
        return
    prev = previous[0]
    print(f"\nvs {prev['run_id']} ({prev['label']}):")
    for m in METRICS:
        d = current["metrics"][m] - prev["metrics"][m]
        arrow = "→" if abs(d) < 1e-9 else ("↑" if d > 0 else "↓")
        print(f"  {m:<10} {prev['metrics'][m]:.3f} {arrow} {current['metrics'][m]:.3f}  ({d:+.3f})")


# ── history ────────────────────────────────────────────────────────────────

def load_runs() -> list[dict]:
    """Every run file, newest first. per_query is dropped — summaries only."""
    out = []
    for p in sorted(RUNS.glob("*.json"), reverse=True):
        if p.name == "index.json":
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        d.pop("per_query", None)
        out.append(d)
    return out


def reindex() -> int:
    runs = load_runs()
    (RUNS / "index.json").write_text(
        json.dumps({"runs": runs}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


def show_list() -> int:
    runs = load_runs()
    if not runs:
        print("No runs yet.")
        return 0
    head = f"{'run_id':<21} {'label':<26} " + " ".join(f"{m:>9}" for m in METRICS)
    print(head)
    print("-" * len(head))
    for r in runs:
        print(f"{r['run_id']:<21} {r['label'][:26]:<26} "
              + " ".join(f"{r['metrics'][m]:>9.3f}" for m in METRICS))
        if r.get("notes"):
            print(f"{'':<21} ↳ {r['notes']}")
    return 0


# ── self-check ─────────────────────────────────────────────────────────────

def self_check() -> int:
    """The two worked examples in week02/readme.md, with their published numbers."""
    # single_chunk: one expected chunk, found at rank 3 of 5
    m = score(["c2"], ["a", "b", "c2", "d", "e"], k=5)
    assert m["hit_rate"] == 1.0
    assert m["recall"] == 1.0
    assert round(m["precision"], 3) == 0.2
    assert round(m["mrr"], 3) == 0.333
    assert round(m["ndcg"], 3) == 0.5

    # multi_chunk: three expected, two found at ranks 1 and 2
    m = score(["c3", "c4", "c5"], ["c3", "c4", "x", "y", "z"], k=5)
    assert m["hit_rate"] == 1.0
    assert round(m["recall"], 3) == 0.667
    assert round(m["precision"], 3) == 0.4
    assert m["mrr"] == 1.0
    assert round(m["ndcg"], 3) == 0.765

    # nothing found
    m = score(["c1"], ["x", "y"], k=5)
    assert (m["hit_rate"], m["recall"], m["mrr"], m["ndcg"]) == (0.0, 0.0, 0.0, 0.0)
    assert m["verdict"] == "fail"

    # perfect: every expected chunk at the top
    m = score(["a", "b"], ["a", "b", "x"], k=5)
    assert m["ndcg"] == 1.0 and m["recall"] == 1.0 and m["verdict"] == "pass"

    # IDCG is capped by k: 3 expected but only 2 results allowed means a
    # perfect top-2 must still score 1.0, not 0.7
    m = score(["a", "b", "c"], ["a", "b"], k=2)
    assert m["ndcg"] == 1.0, m["ndcg"]
    assert round(m["recall"], 3) == 0.667

    # empty result set must not divide by zero
    m = score(["a"], [], k=5)
    assert m["ndcg"] == 0.0 and m["precision"] == 0.0

    print("self-check ok")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--label", default="", help="short name for this run")
    ap.add_argument("--note", default="", help="what you changed and why")
    ap.add_argument("--k", type=int, default=5, help="top-K to retrieve (default 5)")
    ap.add_argument("--api", default=API, help=f"document-api base URL (default {API})")
    ap.add_argument("--list", action="store_true", help="show run history")
    ap.add_argument("--reindex", action="store_true", help="rebuild index.json from run files")
    ap.add_argument("--self-check", action="store_true", help="metric asserts only")
    a = ap.parse_args()

    if a.self_check:
        sys.exit(self_check())
    if a.list:
        sys.exit(show_list())
    if a.reindex:
        sys.exit(reindex())
    run(a.label, a.note, a.k, a.api)
