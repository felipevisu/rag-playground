"""Resolve answer passages to chunks.

Each `answer[i]` in questions-and-answers.json is verbatim text from the PDF.
Locate it in the document text (same parse the chunker saw), then pick the
minimal set of chunks whose [start, end) spans cover it.

    exact match on normalized text  ->  fuzzy fallback  ->  unresolved (build aborts)
"""

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from chunkers import Config, parse_pdf

FUZZY_MIN_RATIO = 0.85   # share of the passage's chars that must align
FUZZY_MIN_BLOCK = 8      # shorter aligned runs are coincidence
MIN_NEW_CHARS = 20       # a chunk must add this much uncovered passage text to count


@dataclass
class Hit:
    chunk_id: str
    part: int
    coverage: float


def normalize(s: str) -> tuple[str, list[int]]:
    """Casefold, strip accents, collapse whitespace. Returns (text, map) where
    map[i] is the index in `s` that produced text[i]."""
    out, idx, prev_space = [], [], True
    for i, ch in enumerate(s):
        d = unicodedata.normalize("NFKD", ch)
        d = "".join(c for c in d if not unicodedata.combining(c)).casefold()
        if not d:
            continue
        if d.isspace():
            if not prev_space:
                out.append(" ")
                idx.append(i)
            prev_space = True
        else:
            out.extend(d)
            idx.extend([i] * len(d))
            prev_space = False
    if out and out[-1] == " ":
        out.pop(), idx.pop()
    return "".join(out), idx


def locate(passage: str, doc_norm: str, doc_map: list[int]) -> tuple[int, int] | None:
    """(start, end) of `passage` in the original document, or None."""
    p, _ = normalize(passage)
    if not p:
        return None
    j = doc_norm.find(p)
    if j >= 0:
        return doc_map[j], doc_map[j + len(p) - 1] + 1
    # fuzzy: align passage against the document, keep long runs only
    blocks = [
        b for b in SequenceMatcher(None, doc_norm, p, autojunk=False).get_matching_blocks()
        if b.size >= FUZZY_MIN_BLOCK
    ]
    if not blocks:
        return None
    matched = sum(b.size for b in blocks)
    lo, hi = blocks[0].a, blocks[-1].a + blocks[-1].size
    if matched / len(p) < FUZZY_MIN_RATIO or hi - lo > 1.5 * len(p):
        return None
    return doc_map[lo], doc_map[hi - 1] + 1


def cover(span: tuple[int, int], chunks: list[dict]) -> list[tuple[dict, float]]:
    """Minimal chunks covering `span`, each with its coverage fraction.
    Greedy by overlap size; a chunk only counts if it adds MIN_NEW_CHARS."""
    lo, hi = span
    cands = []
    for c in chunks:
        a, b = max(lo, c["start"]), min(hi, c["end"])
        if b - a > 0:
            cands.append((b - a, a, b, c))
    cands.sort(key=lambda x: -x[0])
    covered = bytearray(hi - lo)
    picked = []
    for n, a, b, c in cands:
        new = (b - a) - sum(covered[a - lo:b - lo])
        if new < MIN_NEW_CHARS and picked:
            continue
        covered[a - lo:b - lo] = b"\x01" * (b - a)
        picked.append((c, n / (hi - lo)))
    return picked


def resolve(pdf_dir: Path, cfg: Config, corpus: list[dict], qa: list[dict]) -> tuple[list[dict], list[str]]:
    """qa rows: {query_id, filename, answer: [passages]} -> (answers rows, unresolved messages)."""
    by_file: dict[str, list[dict]] = {}
    for c in corpus:
        by_file.setdefault(c["filename"], []).append(c)

    docs: dict[str, tuple[str, list[int]]] = {}
    rows, unresolved = [], []
    for q in qa:
        fname = q["filename"]
        if fname not in by_file:
            unresolved.append(f'{q["query_id"]}: no chunks for {fname}')
            continue
        if fname not in docs:
            text, _ = parse_pdf(pdf_dir / fname, cfg.strip_footer)
            docs[fname] = normalize(text)
        doc_norm, doc_map = docs[fname]
        for i, passage in enumerate(q["answer"]):
            span = locate(passage, doc_norm, doc_map)
            if span is None:
                unresolved.append(f'{q["query_id"]}[{i}] ({fname}): "{passage[:70]}…"')
                continue
            for c, cov in cover(span, by_file[fname]):
                rows.append({"query_id": q["query_id"], "chunk_id": c["chunk_id"],
                             "part": i, "coverage": round(cov, 3)})
    return rows, unresolved
