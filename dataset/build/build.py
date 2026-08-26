#!/usr/bin/env python3
"""Build the evaluation dataset: corpus.parquet, queries.parquet, answers.parquet.

Runs the same parse + chunk pipeline as week01's worker (docling
DocumentConverter -> HybridChunker), then resolves the hand-written ground truth
in source/ground-truth.json against the chunks it just produced.

Ground truth stores *anchors* -- verbatim text lifted from the PDF -- not chunk
ids. Chunk ids move whenever the chunker, its tokenizer or the corpus changes;
anchors do not. Resolution happens here, at build time.

Usage:
  python build.py                # full build
  python build.py --reuse-corpus # re-resolve against out/corpus.parquet, no docling
  python build.py --self-check   # run the resolver asserts, no docling needed
"""

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "source" / "pdfs"
GROUND_TRUTH = ROOT / "source" / "ground-truth.json"
OUT = ROOT / "out"

# Kept identical to week01/services/worker/main.py so the chunks here match the
# ones the worker writes to Postgres.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 512

# An anchor resolves when the chunks picked together cover this much of it.
# Anchors are ~220 chars, so 0.70 tolerates a chunk boundary shaving the ends
# while staying far above the noise floor of repeated legal boilerplate.
COVER_MIN = 0.70
# A single chunk must contribute at least this much to be considered at all.
COVER_PARTIAL = 0.25
# Enough coverage that looking for another chunk is pointless.
COVER_DONE = 0.95
# Shortest run of characters counted as a real match. Docling reflows list
# markers and line breaks between versions, which shatters one long run into
# several; anything below this is coincidence between two legal texts.
MIN_BLOCK = 8


# ── anchor resolution ──────────────────────────────────────────────────────

def norm(s: str) -> str:
    """Whitespace-collapsed, case-folded, accent-stripped form for matching."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).strip().casefold()


def spans(anchor: str, text: str) -> list[tuple[int, int]]:
    """Ranges of `anchor` present in `text`, as matching blocks >= MIN_BLOCK.

    All blocks, not just the longest: docling changes how it renders list
    markers and line breaks between versions, so an anchor lifted from an older
    run survives as several aligned runs rather than one.
    """
    if anchor in text:
        return [(0, len(anchor))]
    blocks = SequenceMatcher(None, anchor, text, autojunk=False).get_matching_blocks()
    return [(b.a, b.a + b.size) for b in blocks if b.size >= MIN_BLOCK]


def covered_chars(spans_: list[tuple[int, int]]) -> set[int]:
    return {i for lo, hi in spans_ for i in range(lo, hi)}


def resolve(anchor: str, chunks: list[str]) -> list[int]:
    """Indices of the chunks that hold `anchor`, best first. [] when unresolved.

    An anchor split across a chunk boundary is covered by no single chunk, so we
    union the anchor *characters* each chunk accounts for rather than summing
    their coverage -- summing would let boilerplate repeated across unrelated
    chunks add up to a false match.
    """
    a = norm(anchor)
    if not a:
        return []
    scored = []
    for i, c in enumerate(chunks):
        chars = covered_chars(spans(a, norm(c)))
        if len(chars) / len(a) >= COVER_PARTIAL:
            scored.append((len(chars), chars, i))
    scored.sort(key=lambda x: x[0], reverse=True)

    covered: set[int] = set()
    hits: list[int] = []
    for _, chars, i in scored:
        if len(chars - covered) / len(a) < 0.05:  # repeats what we already have
            continue
        covered |= chars
        hits.append(i)
        if len(covered) / len(a) >= COVER_DONE:
            break
    return hits if len(covered) / len(a) >= COVER_MIN else []


# ── chunking ───────────────────────────────────────────────────────────────

def chunk_pdfs() -> list[dict]:
    from docling.chunking import HybridChunker
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from transformers import AutoTokenizer

    opts = PdfPipelineOptions()
    opts.do_ocr = False
    opts.do_table_structure = False
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    chunker = HybridChunker(
        tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME),
        max_tokens=MAX_TOKENS,
        merge_peers=True,
    )

    rows = []
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"no PDFs in {PDF_DIR}")
    for n, pdf in enumerate(pdfs, 1):
        print(f"[{n}/{len(pdfs)}] {pdf.name} — converting (1-3 min on CPU)")
        doc = converter.convert(str(pdf)).document
        for i, ch in enumerate(chunker.chunk(doc)):
            pages = sorted({
                p.page_no
                for item in ch.meta.doc_items
                for p in getattr(item, "prov", [])
                if getattr(p, "page_no", None) is not None
            })
            rows.append({
                "chunk_id": f"{pdf.stem}::{i:04d}",
                "filename": pdf.name,
                "chunk_index": i,
                "pages": pages,
                "headings": list(ch.meta.headings or []),
                "text": ch.text,
                "n_chars": len(ch.text),
            })
        print(f"      -> {sum(r['filename'] == pdf.name for r in rows)} chunks")
    return rows


# ── build ──────────────────────────────────────────────────────────────────

def build(reuse_corpus: bool = False) -> int:
    import pandas as pd

    OUT.mkdir(exist_ok=True)
    corpus_path = OUT / "corpus.parquet"
    if reuse_corpus:
        if not corpus_path.exists():
            sys.exit(f"{corpus_path.name} not there yet — run without --reuse-corpus")
        corpus = pd.read_parquet(corpus_path).to_dict("records")
        for r in corpus:  # parquet gives numpy arrays back for list columns
            r["pages"] = list(r["pages"])
            r["headings"] = list(r["headings"])
        print(f"reusing {corpus_path.name}: {len(corpus)} chunks")
    else:
        corpus = chunk_pdfs()
        # The chunks are valid output regardless of whether the ground truth
        # resolves, and re-deriving them costs ~40 min of CPU. Write them now.
        pd.DataFrame(corpus).to_parquet(corpus_path, index=False)
        print(f"wrote {corpus_path.name}: {len(corpus)} rows")

    by_file: dict[str, list[dict]] = {}
    for r in corpus:
        by_file.setdefault(r["filename"], []).append(r)

    gt = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    queries, answers, unresolved = [], [], []

    for doc in gt["documents"]:
        fname = doc["filename"]
        cands = by_file.get(fname)
        if not cands:
            unresolved.append(f"{fname}: no chunks (PDF missing from pdfs/?)")
            continue
        texts = [c["text"] for c in cands]

        for q in doc["questions"]:
            queries.append({
                "query_id": q["query_id"],
                "question": q["question"],
                "type": q["type"],
                "expected_answer": q["expected_answer"],
                "filename": fname,
                "title": doc["title"],
                "n_anchors": len(q["anchors"]),
            })
            seen = set()
            for anchor in q["anchors"]:
                hits = resolve(anchor, texts)
                if not hits:
                    unresolved.append(
                        f'{q["query_id"]} ({fname}): anchor "{anchor[:70]}…"'
                    )
                    continue
                for i in hits:
                    c = cands[i]
                    if c["chunk_id"] in seen:
                        continue
                    seen.add(c["chunk_id"])
                    answers.append({
                        "query_id": q["query_id"],
                        "chunk_id": c["chunk_id"],
                        "relevance": 1,
                        "filename": c["filename"],
                        "chunk_index": c["chunk_index"],
                        "pages": c["pages"],
                    })

    if unresolved:
        print(f"\n{len(unresolved)} anchor(s) did not resolve:", file=sys.stderr)
        for u in unresolved:
            print(f"  - {u}", file=sys.stderr)
        print(
            "\nGround truth is incomplete — refusing to write a silently degraded "
            "dataset. Fix the anchors in questions.json (they must be verbatim "
            "text from the PDF) and re-run.",
            file=sys.stderr,
        )
        return 1

    for name, rows in (("queries", queries), ("answers", answers)):
        path = OUT / f"{name}.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        print(f"wrote {path.name}: {len(rows)} rows")

    write_manifest(corpus, queries, answers)

    docs = len({r["filename"] for r in corpus})
    print(
        f"\n{docs} documents · {len(corpus)} chunks · {len(queries)} queries · "
        f"{len(answers)} query-chunk pairs"
    )
    return 0


def write_manifest(corpus: list, queries: list, answers: list) -> None:
    """Record what produced this build.

    Without it the parquet files are an unattributable blob: the same PDFs and
    the same settings gave 180 chunks on one docling build and 183 on the next,
    and nothing in the output said which was which. Diffing two manifests
    answers that in one step.
    """
    def ver(pkg: str) -> str:
        try:
            return version(pkg)
        except PackageNotFoundError:
            return "not installed"

    h = hashlib.sha256()
    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        h.update(pdf.name.encode())
        h.update(hashlib.sha256(pdf.read_bytes()).digest())

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parser": {"do_ocr": False, "do_table_structure": False},
        "chunker": {
            "class": "HybridChunker",
            "max_tokens": MAX_TOKENS,
            "merge_peers": True,
            "tokenizer": MODEL_NAME,
        },
        "versions": {
            p: ver(p)
            for p in ("docling", "docling-core", "transformers", "torch", "pyarrow")
        },
        "counts": {
            "documents": len({r["filename"] for r in corpus}),
            "chunks": len(corpus),
            "queries": len(queries),
            "qrels": len(answers),
        },
        "corpus_sha256": h.hexdigest(),
    }
    path = OUT / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path.name}")


# ── self-check ─────────────────────────────────────────────────────────────

def self_check() -> int:
    chunks = [
        "Art. 1o Fica instituída a Política Nacional de Justiça Climática.",
        "Parágrafo único. A Política reger-se-á pelos princípios da precaução, "
        "prevenção e equidade intergeracional, bem como pelo reconhecimento.",
        "§ 1o São fontes de recursos do FNARC as dotações da LOA.",
    ]
    # exact hit
    assert resolve("Parágrafo único. A Política reger-se-á", chunks) == [1]
    # whitespace + accent insensitive
    assert resolve("PARAGRAFO   UNICO.  A  POLITICA reger-se-a", chunks) == [1]
    # mid-chunk, not just the head
    assert resolve("equidade intergeracional, bem como pelo reconhecimento", chunks) == [1]
    # anchor straddling a boundary picks up both sides
    split = resolve(
        "prevenção e equidade intergeracional, bem como pelo reconhecimento. "
        "§ 1o São fontes de recursos do FNARC as dotações da LOA.",
        chunks,
    )
    assert set(split) == {1, 2}, split
    # docling re-rendered the list markers -- runs are broken up but aligned
    reflowed = ["- Art. 9o Fica proibida a disposicao final em aterros "
                "sanitarios de: - I - residuos reciclaveis; - II - residuos organicos."]
    assert resolve(
        "Art. 9o Fica proibida a disposicao final em aterros sanitarios de: "
        "I - residuos reciclaveis; II - residuos organicos.",
        reflowed,
    ) == [0]
    # no match at all -> unresolved, never a wrong guess
    assert resolve("dispõe sobre a pesca artesanal em águas interiores", chunks) == []
    print("self-check ok")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true", help="test the resolver only")
    ap.add_argument("--reuse-corpus", action="store_true",
                    help="skip docling, re-resolve against the existing corpus.parquet")
    args = ap.parse_args()
    sys.exit(self_check() if args.self_check else build(args.reuse_corpus))
