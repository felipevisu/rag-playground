"""Configurable PDF chunker.

    PDF ─► parse ─► pieces ─► pack ─► chunks

parse   pypdf per page, optional footer stripping
split   where to cut: fixed | recursive | sentence | legal
pack    measure pieces (chars | words | tokens), split oversize, merge
        undersize, optional overlap
"""

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

SPLITTERS = ("fixed", "recursive", "sentence", "legal")
UNITS = ("chars", "words", "tokens")
# HF model id -> real max sequence length of the embedding model. Offered in the
# UI; the CLI accepts any id. Keep `max` at or under this number.
TOKENIZERS = {
    "sentence-transformers/all-MiniLM-L6-v2": 256,
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 128,
    "intfloat/multilingual-e5-small": 512,
    "intfloat/multilingual-e5-large": 512,
    "BAAI/bge-m3": 8192,
    "Alibaba-NLP/gte-multilingual-base": 8192,
    "nomic-ai/nomic-embed-text-v1.5": 8192,
}

# Câmara PDFs stamp every page with these four lines.
FOOTER = re.compile(
    r"^(\*CD\d+\*.*|Apresentação: .*|Assinado eletronicamente .*|"
    r"Para verificar as? assinaturas?, acesse .*)$",
    re.M,
)
RECURSIVE_SEPS = ["\n\n", "\n", ". ", "; ", " "]
SENTENCE_END = re.compile(r"(?<=[.;:!?])\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ§\"“(])")
LEGAL_HEAD = re.compile(
    r"^(Art\.\s*\d+[ºo°]?[-A-Z]*|§\s*\d+[ºo°]?(?:-[A-Z])?|Parágrafo único|"
    r"CAPÍTULO [IVXL]+|JUSTIFICA(?:ÇÃO|TIVA)|Sala das Sessões.*)",
    re.M,
)


@dataclass
class Config:
    name: str
    splitter: str = "recursive"
    unit: str = "chars"
    max: int = 1000
    min: int = 0          # merge neighbours until at least this big (0 = off)
    overlap: int = 0      # tail of previous chunk repeated at the start of the next
    tokenizer: str = ""   # HF model id; required when unit == "tokens"
    strip_footer: bool = True

    def validate(self) -> None:
        assert self.splitter in SPLITTERS, f"splitter must be one of {SPLITTERS}"
        assert self.unit in UNITS, f"unit must be one of {UNITS}"
        assert self.max > 0 and self.min >= 0 and self.overlap >= 0
        assert self.overlap < self.max, "overlap must be smaller than max"
        assert self.min <= self.max, "min must be <= max"
        if self.unit == "tokens":
            assert self.tokenizer, "unit=tokens needs a tokenizer"
        if self.splitter == "legal":
            assert self.overlap == 0, "legal splitter keeps article boundaries: no overlap"


@dataclass
class Piece:
    text: str
    start: int            # char offset in the document text
    heading: str = ""


@dataclass
class Chunk:
    text: str
    pages: list[int]
    heading: str
    size: int
    start: int = 0        # char span in the parse_pdf() text; lets answers
    end: int = 0          # resolve to chunks by interval intersection


# ── parse ──────────────────────────────────────────────────────────────────

def parse_pdf(path: Path, strip_footer: bool) -> tuple[str, list[int]]:
    """Return (document text, page_starts) — page_starts[i] = offset where page i+1 begins."""
    from pypdf import PdfReader

    text, starts = "", []
    for page in PdfReader(str(path)).pages:
        t = page.extract_text() or ""
        if strip_footer:
            t = FOOTER.sub("", t)
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"\n{3,}", "\n\n", t).strip()
        starts.append(len(text))
        text += t + "\n\n"
    return text, starts


def pages_for(start: int, end: int, page_starts: list[int]) -> list[int]:
    return [
        i + 1
        for i, s in enumerate(page_starts)
        if s < end and (i + 1 == len(page_starts) or page_starts[i + 1] > start)
    ]


# ── measure ────────────────────────────────────────────────────────────────

class Measure:
    def __init__(self, unit: str, tokenizer: str = ""):
        self.unit = unit
        self.tok = None
        if unit == "tokens":
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(tokenizer)

    def __call__(self, text: str) -> int:
        if self.unit == "chars":
            return len(text)
        if self.unit == "words":
            return len(text.split())
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def head(self, text: str, n: int) -> str:
        """First n units of text."""
        if self.unit == "chars":
            return text[:n]
        if self.unit == "words":
            return " ".join(text.split()[:n])
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        return text[: offs[n - 1][1]] if len(offs) > n else text

    def tail(self, text: str, n: int) -> str:
        """Last n units of text."""
        if self.unit == "chars":
            t = text[-n:]
            i = t.find(" ")  # don't start the overlap mid-word
            return t[i + 1:] if 0 <= i < len(t) - 1 else t
        if self.unit == "words":
            return " ".join(text.split()[-n:])
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        return text[offs[-n][0]:] if len(offs) > n else text


# ── split ──────────────────────────────────────────────────────────────────

def split_at(text: str, pattern: re.Pattern, keep_heading: bool = False) -> list[Piece]:
    """Cut `text` right before every match of `pattern`."""
    cuts = [m.start() for m in pattern.finditer(text)]
    if not cuts or cuts[0] != 0:
        cuts.insert(0, 0)
    cuts.append(len(text))
    pieces, heading = [], ""
    for a, b in zip(cuts, cuts[1:]):
        seg = text[a:b]
        if not seg.strip():
            continue
        if keep_heading and (m := pattern.match(seg)):
            heading = m.group(0).strip()
        pieces.append(Piece(seg, a, heading))
    return pieces


def split_recursive(piece: Piece, measure: Measure, max_: int, seps=RECURSIVE_SEPS) -> list[Piece]:
    """Cut an oversize piece at the coarsest separator that gets it under max."""
    if measure(piece.text) <= max_:
        return [piece]
    if not seps:  # nothing left to split on: hard cut
        head = measure.head(piece.text, max_)
        rest = Piece(piece.text[len(head):], piece.start + len(head), piece.heading)
        return [Piece(head, piece.start, piece.heading)] + split_recursive(rest, measure, max_, seps)
    sep, *rest_seps = seps
    out, pos = [], 0
    for part in piece.text.split(sep):
        seg = part + sep
        if pos + len(seg) > len(piece.text):
            seg = piece.text[pos:]
        if seg.strip():
            out += split_recursive(Piece(seg, piece.start + pos, piece.heading), measure, max_, rest_seps)
        pos += len(seg)
    return out


def pieces_for(text: str, cfg: Config, measure: Measure) -> list[Piece]:
    if cfg.splitter == "fixed":
        return [Piece(text, 0)]  # packing does the cutting
    if cfg.splitter == "recursive":
        return [Piece(text, 0)]  # oversize handling does the cutting
    if cfg.splitter == "sentence":
        return split_at(text, SENTENCE_END)
    if cfg.splitter == "legal":
        return split_at(text, LEGAL_HEAD, keep_heading=True)
    raise ValueError(cfg.splitter)


# ── pack ───────────────────────────────────────────────────────────────────

def pack(pieces: list[Piece], cfg: Config, measure: Measure, page_starts: list[int]) -> list[Chunk]:
    """Split oversize pieces, then merge neighbours (same heading) up to max."""
    if cfg.splitter == "fixed":
        step = cfg.max - cfg.overlap
        out, text = [], pieces[0].text
        pos = 0
        while pos < len(text):
            head = measure.head(text[pos:], cfg.max)
            if not head.strip():
                break
            out.append(Chunk(head.strip(), pages_for(pos, pos + len(head), page_starts), "", measure(head),
                             pos, pos + len(head)))
            adv = len(measure.head(text[pos:], step))
            pos += max(adv, 1)
        return out

    atoms: list[Piece] = []
    for p in pieces:
        atoms += split_recursive(p, measure, cfg.max)

    chunks: list[Chunk] = []
    buf: list[Piece] = []

    def flush():
        if not buf:
            return
        text = "".join(p.text for p in buf)
        start, end = buf[0].start, buf[-1].start + len(buf[-1].text)
        chunks.append(Chunk(text.strip(), pages_for(start, end, page_starts), buf[0].heading,
                            measure(text.strip()), start, end))

    for a in atoms:
        same_heading = not buf or a.heading == buf[0].heading
        fits = measure("".join(p.text for p in buf) + a.text) <= cfg.max
        below_min = not buf or measure("".join(p.text for p in buf)) < cfg.min
        if buf and not (fits and (same_heading or below_min)):
            flush()
            carry = []
            if cfg.overlap:
                tail = measure.tail(chunks[-1].text, cfg.overlap)
                carry = [Piece(tail + " ", buf[-1].start + len(buf[-1].text) - len(tail), buf[0].heading)]
            buf = carry
        buf.append(a)
    flush()
    return chunks


# ── entry point ────────────────────────────────────────────────────────────

def chunk_pdf(path: Path, cfg: Config, measure: Measure | None = None) -> list[dict]:
    cfg.validate()
    measure = measure or Measure(cfg.unit, cfg.tokenizer)
    text, page_starts = parse_pdf(path, cfg.strip_footer)
    chunks = pack(pieces_for(text, cfg, measure), cfg, measure, page_starts)
    return [
        {
            "chunk_id": f"{path.stem}::{i:04d}",
            "filename": path.name,
            "chunk_index": i,
            "pages": c.pages,
            "heading": c.heading,
            "text": c.text,
            "n_chars": len(c.text),
            "size": c.size,
            "start": c.start,
            "end": c.end,
        }
        for i, c in enumerate(chunks)
    ]


def chunk_dir(pdf_dir: Path, cfg: Config) -> list[dict]:
    measure = Measure(cfg.unit, cfg.tokenizer)
    rows = []
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        rows += chunk_pdf(pdf, cfg, measure)
    return rows
