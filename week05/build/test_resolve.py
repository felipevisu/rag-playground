"""Run: python test_resolve.py"""

from resolve import cover, locate, normalize

DOC = "Art. 1º  Fica instituída\na Política.\nParágrafo único. Reger-se-á pelos princípios da precaução."
n, m = normalize(DOC)

# normalize: whitespace collapsed, accents/case gone, map points back into DOC
assert n == "art. 1o fica instituida a politica. paragrafo unico. reger-se-a pelos principios da precaucao.", n
assert DOC[m[n.index("politica")]:].startswith("Política")

# exact hit despite different whitespace / accents in the passage
s = locate("PARÁGRAFO ÚNICO. Reger-se-á   pelos princípios", n, m)
assert s and DOC[s[0]:s[1]] == "Parágrafo único. Reger-se-á pelos princípios", DOC[s[0]:s[1]] if s else s

# fuzzy: pypdf-style corrupted accents still land on the right span
s = locate("Parágrafo único. Reger-se-a - pelos princí-pios da precauç a3o", n, m)
assert s and DOC[s[0]:s[1]].startswith("Parágrafo único") and "precau" in DOC[s[0]:s[1]], s

# nothing alike -> None, never a wrong guess
assert locate("dispõe sobre a pesca artesanal em águas interiores", n, m) is None

# cover: passage split across two chunks picks both; a 3-char boundary graze is ignored;
# an overlap-duplicate that adds nothing is skipped
chunks = [
    {"chunk_id": "a", "start": 0, "end": 40},
    {"chunk_id": "b", "start": 40, "end": 100},
    {"chunk_id": "c", "start": 97, "end": 200},
    {"chunk_id": "dup", "start": 0, "end": 100},  # simulates overlap duplicate
]
picked = cover((10, 100), chunks)
assert [c["chunk_id"] for c, _ in picked] == ["dup"], picked  # one chunk covers all -> minimal
picked = cover((10, 100), chunks[:3])
assert [c["chunk_id"] for c, _ in picked] == ["b", "a"], picked  # c adds only 3 chars
assert abs(dict((c["chunk_id"], cov) for c, cov in picked)["b"] - 60 / 90) < 1e-6

print("ok")
