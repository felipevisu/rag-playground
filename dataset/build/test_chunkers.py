"""Smallest check that fails if the packing logic breaks. Run: python test_chunkers.py"""

from chunkers import Config, Measure, Piece, pack, pieces_for, pages_for

DOC = (
    "Art. 1º Fica instituída a Política.\n"
    "Parágrafo único. Reger-se-á pelos princípios.\n"
    "Art. 2º Para os fins desta Lei:\n"
    "I - justiça climática: conjunto de medidas;\n"
    "II - racismo ambiental: práticas ou omissões.\n"
    "JUSTIFICAÇÃO\n"
    "O ordenamento jurídico brasileiro conta com a Política."
)
PAGES = [0, DOC.index("Art. 2º")]  # page 2 starts at Art. 2º


def run(**kw):
    cfg = Config(name="t", **kw)
    m = Measure(cfg.unit)
    return pack(pieces_for(DOC, cfg, m), cfg, m, PAGES)


# legal: headings tracked, incisos stay with their article
legal = run(splitter="legal", max=200)
assert [c.heading for c in legal] == ["Art. 1º", "Parágrafo único", "Art. 2º", "JUSTIFICAÇÃO"], legal
assert "II - racismo" in legal[2].text
assert legal[2].pages == [2] and legal[0].pages == [1]

# legal + min: small neighbours merge across headings, first heading kept
merged = run(splitter="legal", max=200, min=150)
assert len(merged) < len(legal)
assert merged[0].heading == "Art. 1º"

# recursive: never exceeds max
for c in run(splitter="recursive", max=60):
    assert c.size <= 60, c

# fixed with overlap: consecutive chunks share text
fx = run(splitter="fixed", max=50, overlap=10)
assert fx[0].text[-10:].strip() in fx[1].text

# sentence: cuts after sentence end
sent = run(splitter="sentence", max=80)
assert sent[0].text.startswith("Art. 1º") and len(sent) > 1
assert all(c.text.rstrip()[-1] in ".;:" for c in sent[:-1]), [c.text for c in sent]

# words unit
w = run(splitter="recursive", unit="words", max=8)
assert all(c.size <= 8 for c in w)

# hard cut when no separator works
m = Measure("chars")
cfg = Config(name="t", max=5)
out = pack([Piece("abcdefghijkl", 0)], cfg, m, [0])
assert [c.text for c in out] == ["abcde", "fghij", "kl"], out

assert pages_for(0, 5, [0, 10, 20]) == [1]
assert pages_for(8, 12, [0, 10, 20]) == [1, 2]

print("ok")
