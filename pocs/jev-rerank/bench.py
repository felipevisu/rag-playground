"""Benchmark: hybrid-qwen3-0.6b (BM25 + Qwen3 embeddings, RRF) with and without Jev reranking the top 10.

Everything in memory, no DB: data/ holds week05's sentence-512t-multilingual-e5-large corpus,
its answers and the 38 questions.

    python bench.py                 # k=5, Jev reranks the top 10
    python bench.py --k 7 --depth 15
    python bench.py --check         # self-check, no model, no network
"""
import argparse
import concurrent.futures
import json
import math
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request

import pandas as pd
import Stemmer

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
EMBEDDER = "Qwen/Qwen3-Embedding-0.6B"
QUERY_PREFIX = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
RRF_K = 60       # Cormack et al. 2009
RRF_DEPTH = 50   # candidates per ranker before fusing
JEV_MODEL = "jev-1.13.0"  # pinned: P(yes) shifts between versions
TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
# Asked in English (Jev's best language) about Portuguese text. criteria split "answers"
# from "same subject", the classic first-stage mistake.
JEV_QUESTIONS = {"answers": {
    "type": "noul",
    "instructions": "Does `passage` contain information that answers `query`?",
    "criteria": {"true": "The passage states facts that directly answer the question about this bill",
                 "false": "The passage is about another bill, or only shares keywords or the general topic"},
}}


# ── first stage: BM25 + vectors, fused by RRF ──────────────────────────────

_stemmer = Stemmer.Stemmer("portuguese")
TOKEN_RE = re.compile(r"\d+(?:[/-]\d+)+|\w+")  # "8666/93" stays one token
DIGIT_DOT_RE = re.compile(r"(?<=\d)\.(?=\d)")  # "8.666" == "8666"


def tokenize(text: str) -> list[str]:
    """Stem, strip accents, stem again: "decisão", "decisões", "decisao" meet."""
    words = _stemmer.stemWords(TOKEN_RE.findall(DIGIT_DOT_RE.sub("", text.lower())))
    return _stemmer.stemWords(["".join(c for c in unicodedata.normalize("NFD", w) if not unicodedata.combining(c))
                               for w in words])


def doc_label(filename: str) -> str:
    """'PL_993_2026.pdf' -> 'PL 993/2026': the document as the questions name it."""
    m = re.fullmatch(r"([A-Z]+)_(\d+)_(\d{4})\.pdf", filename)
    return f"{m[1]} {m[2]}/{m[3]}" if m else os.path.splitext(filename)[0].replace("_", " ")


def rrf(*rankings: list[str]) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for r, c in enumerate(ranking, 1):
            scores[c] = scores.get(c, 0.0) + 1 / (RRF_K + r)
    return sorted(scores, key=scores.get, reverse=True)


class Hybrid:
    def __init__(self, corpus: pd.DataFrame):
        import numpy as np
        import rank_bm25
        import torch
        from sentence_transformers import SentenceTransformer

        self.np = np
        self.ids = corpus.chunk_id.tolist()
        texts = [f"{h or ''}\n{t}" for h, t in zip(corpus.heading, corpus.text)]
        self.bm25 = rank_bm25.BM25Okapi([tokenize(t) for t in texts])
        # fp32: Qwen3 ships bf16, which has no fast CPU kernel
        self.model = SentenceTransformer(EMBEDDER, model_kwargs={"torch_dtype": torch.float32})
        t = time.time()
        self.vecs = self.model.encode(texts, normalize_embeddings=True, batch_size=16, show_progress_bar=True)
        print(f"  {len(texts)} chunks embedded in {time.time() - t:.0f}s on {self.model.device}")

    def search(self, q: str, n: int) -> list[str]:
        tokens = set(tokenize(q))
        scores = self.bm25.get_scores(list(tokens))
        bm = [i for i in self.np.argsort(-scores) if not tokens.isdisjoint(self.bm25.doc_freqs[i])][:RRF_DEPTH]
        qv = self.model.encode([QUERY_PREFIX + q], normalize_embeddings=True)[0]
        vec = self.np.argsort(-(self.vecs @ qv))[:RRF_DEPTH]
        return rrf([self.ids[i] for i in bm], [self.ids[i] for i in vec])[:n]


# ── second stage: Jev ──────────────────────────────────────────────────────

def jev_post(key: str, payload: dict) -> dict:
    """POST to TypeSafe. 429 rate limit / 529 overloaded: back off and retry, 3 times."""
    req = urllib.request.Request(TYPESAFE_URL, json.dumps(payload).encode(), {
        "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 529) and attempt < 3:
                wait = e.headers.get("Retry-After", "")
                time.sleep(int(wait) if wait.isdigit() else 2 ** attempt)
                continue
            raise SystemExit(f"typesafe {e.code}: {e.read().decode(errors='replace')[:300]}")


def jev_scores(key: str, q: str, passages: list[str]) -> tuple[list[float], int]:
    """P(passage answers q) per passage, all calls in parallel. Also input tokens used."""
    def one(p):
        return jev_post(key, {"model": JEV_MODEL, "state": {"query": q, "passage": p}, "questions": JEV_QUESTIONS})
    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        rs = list(pool.map(one, passages))
    return [float(r["answers"]["answers"]["noul"]) for r in rs], sum((r.get("usage") or {}).get("input_tokens") or 0 for r in rs)


# ── metrics ────────────────────────────────────────────────────────────────

def score(expected: list[str], retrieved: list[str]) -> dict:
    """Binary-gain nDCG, recall, MRR, hit for one query; retrieved is chunk_ids in rank order."""
    ranks = sorted(retrieved.index(c) + 1 for c in expected if c in retrieved)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(expected), len(retrieved))))
    return {"ndcg": sum(1 / math.log2(r + 1) for r in ranks) / idcg if idcg else 0.0,
            "recall": len(ranks) / len(expected),
            "mrr": 1 / ranks[0] if ranks else 0.0,
            "hit": 1.0 if ranks else 0.0}


def report(runs: dict[str, list[dict]], questions: list[str]) -> None:
    names = list(runs)
    print(f"\n{'run':<24}" + "".join(f"{m:>8}" for m in ("ndcg", "recall", "mrr", "hit")))
    for n in names:
        print(f"{n:<24}" + "".join(f"{sum(r[m] for r in runs[n]) / len(runs[n]):>8.3f}"
                                   for m in ("ndcg", "recall", "mrr", "hit")))
    base = runs[names[0]]
    for n in names[1:]:
        delta = [r["ndcg"] - b["ndcg"] for r, b in zip(runs[n], base)]
        up, down = sum(d > 1e-9 for d in delta), sum(d < -1e-9 for d in delta)
        print(f"\n{n} vs {names[0]}: {up} melhoraram, {down} pioraram, {len(delta) - up - down} iguais (nDCG)")
        for i in sorted(range(len(delta)), key=lambda i: delta[i]):
            if abs(delta[i]) > 1e-9:
                print(f"  {delta[i]:+.2f}  {questions[i][:90]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5, help="results scored per question")
    ap.add_argument("--depth", type=int, default=10, help="candidates Jev rescores")
    ap.add_argument("--check", action="store_true", help="self-check, no model, no network")
    a = ap.parse_args()
    if a.check:
        return self_check()

    corpus = pd.read_parquet(os.path.join(DATA, "corpus.parquet"))
    queries = pd.read_parquet(os.path.join(DATA, "queries.parquet")).to_dict("records")
    qrels: dict[str, list[str]] = {}
    for qid, cid in pd.read_parquet(os.path.join(DATA, "answers.parquet"))[["query_id", "chunk_id"]].itertuples(index=False):
        if cid not in qrels.setdefault(qid, []):
            qrels[qid].append(cid)
    passage = {c: f"{doc_label(f)}\n{h or ''}\n{t}".strip()
               for c, f, h, t in zip(corpus.chunk_id, corpus.filename, corpus.heading, corpus.text)}
    key = os.environ.get("TYPESAFE_API_KEY", "")

    print(f"{len(corpus)} chunks, {len(queries)} perguntas, k={a.k}, Jev no top {a.depth}"
          + ("" if key else "  (sem TYPESAFE_API_KEY: só a run sem Jev)"))
    hybrid = Hybrid(corpus)
    runs: dict[str, list[dict]] = {"hybrid (sem jev)": []}
    if key:
        runs["hybrid + jev"] = []
    tokens, t0 = 0, time.time()
    for i, q in enumerate(queries, 1):
        cands = hybrid.search(q["question"], max(a.k, a.depth))
        expected = qrels[q["query_id"]]
        runs["hybrid (sem jev)"].append(score(expected, cands[:a.k]))
        if key:
            top = cands[:a.depth]
            s, used = jev_scores(key, q["question"], [passage[c] for c in top])
            tokens += used
            jev = [c for _, c in sorted(zip(s, top), key=lambda x: -x[0])]
            runs["hybrid + jev"].append(score(expected, (jev + cands[a.depth:])[:a.k]))
        print(f"\r  {i}/{len(queries)}", end="", flush=True)
    print(f"\r  {len(queries)} perguntas em {time.time() - t0:.0f}s"
          + (f", Jev: {tokens:,} tokens ≈ US$ {tokens * 0.042 / 1e6:.4f}" if key else ""))
    report(runs, [q["question"] for q in queries])


def self_check() -> None:
    assert tokenize("Decisão decisões decisao") == ["decis"] * 3
    assert tokenize("Lei 8.666/93") == ["lei", "8666/93"]
    assert doc_label("PL_993_2026.pdf") == "PL 993/2026"
    assert rrf(["a", "b", "c"], ["c", "a", "b"])[0] == "a"
    m = score(["x", "y"], ["x", "z", "y"])
    assert m["recall"] == 1.0 and m["mrr"] == 1.0 and round(m["ndcg"], 3) == 0.920, m
    assert score(["x"], ["a", "b"]) == {"ndcg": 0.0, "recall": 0.0, "mrr": 0.0, "hit": 0.0}
    # jev: request shape, 529 retried, scores in passage order
    import io
    sent, replies = [], []
    def fake(req, timeout):  # noqa: E306
        sent.append(json.loads(req.data))
        r = replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return io.BytesIO(json.dumps(r).encode())
    real, real_sleep = urllib.request.urlopen, time.sleep
    urllib.request.urlopen, time.sleep = fake, lambda s: None
    replies[:] = [urllib.error.HTTPError("u", 529, "busy", {}, io.BytesIO(b"")),
                  {"answers": {"answers": {"type": "noul", "noul": 0.8}}, "usage": {"input_tokens": 7}}]
    assert jev_scores("k", "q?", ["p"]) == ([0.8], 7)
    assert sent[-1] == {"model": JEV_MODEL, "state": {"query": "q?", "passage": "p"}, "questions": JEV_QUESTIONS}
    urllib.request.urlopen, time.sleep = real, real_sleep
    print("ok")


if __name__ == "__main__":
    main()
