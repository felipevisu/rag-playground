#!/usr/bin/env python3
"""
Downloads bills (PL/PLP/PDL/MPV/PRC) from the Chamber of Deputies Open Data API, one folder
per segment, until each segment has TARGET PDFs on disk.

  pdfs/<segment>/<TYPE>_<num>_<year>.pdf
  metadata.csv   (one row per PDF, with a `segmento` column)

Idempotent: PDFs already on disk are skipped, so re-running resumes.

Usage:
  pip install requests
  python download.py                    # all segments
  python download.py --only saude       # one segment
"""

import argparse
import csv
import time
from pathlib import Path

import requests

BASE = "https://dadosabertos.camara.leg.br/api/v2"
HEADERS = {"Accept": "application/json", "User-Agent": "estudo-rag-pls/1.0"}
HERE = Path(__file__).resolve().parent
YEARS = [2026, 2025, 2024, 2023]  # newest first: 2025/26 holds week05's 57 env bills
TARGET = 114                      # 2x week05's meio ambiente corpus
TYPES = "PL,PLP,PDL,MPV,PRC"       # same mix as week05 (which named them all PL_)

# segment folder -> API keyword (full-text over summary/index terms)
SEGMENTS = {
    "meio_ambiente": "meio ambiente",
    "saude": "saúde",
    "educacao": "educação",
}
FIELDS = ["id", "segmento", "sigla", "ementa", "dataApresentacao", "situacao", "urlInteiroTeor", "arquivo"]


def get(path, **params):
    r = requests.get(f"{BASE}/{path}", params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()["dados"]


def bills(keyword):
    """Yield bill summaries for the keyword, newest year first, paging lazily."""
    for year in YEARS:
        page = 1
        while data := get("proposicoes", keywords=keyword, ano=year, siglaTipo=TYPES,
                          itens=100, pagina=page, ordem="ASC", ordenarPor="id"):
            yield from data
            page += 1


def fetch(url, target):
    try:
        r = requests.get(url, headers={"User-Agent": HEADERS["User-Agent"]}, timeout=60)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"  ! {url}: {e}")
        return False
    if not r.content.startswith(b"%PDF"):  # some full texts are .doc/html, skip them
        print(f"  ! not a PDF: {url}")
        return False
    target.write_bytes(r.content)
    return True


def download_segment(segment, keyword, rows):
    folder = HERE / "pdfs" / segment
    folder.mkdir(parents=True, exist_ok=True)
    have = {p.name for p in folder.glob("*.pdf")}
    print(f"\n== {segment} ({keyword}): {len(have)}/{TARGET} on disk")

    for b in bills(keyword):
        if len(have) >= TARGET:
            break
        name = f'{b["siglaTipo"]}_{b["numero"]}_{b["ano"]}.pdf'
        path = folder / name
        key = str(path.relative_to(HERE))
        if name in have and key in rows:
            continue
        d = get(f'proposicoes/{b["id"]}')
        url = d.get("urlInteiroTeor")
        if name not in have:
            if not url or not fetch(url, path):
                continue
            have.add(name)
            print(f"  [{len(have)}/{TARGET}] {name}  {(d.get('ementa') or '')[:80]}")
            time.sleep(0.3)  # be gentle with the API
        rows[key] = {  # a bill can live in several segments
            "id": d["id"],
            "segmento": segment,
            "sigla": f'{d["siglaTipo"]} {d["numero"]}/{d["ano"]}',
            "ementa": (d.get("ementa") or "").strip(),
            "dataApresentacao": d.get("dataApresentacao"),
            "situacao": (d.get("statusProposicao") or {}).get("descricaoSituacao"),
            "urlInteiroTeor": url,
            "arquivo": key,
        }

    if len(have) < TARGET:
        print(f"  ! only {len(have)} available for '{keyword}' in {YEARS}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=SEGMENTS, help="Download a single segment")
    args = ap.parse_args()

    csv_path = HERE / "metadata.csv"
    rows = {}
    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            rows = {r["arquivo"]: r for r in csv.DictReader(f)}

    try:
        for seg, kw in SEGMENTS.items():
            if not args.only or args.only == seg:
                download_segment(seg, kw, rows)
    finally:  # keep metadata for whatever was downloaded, even on Ctrl-C / API error
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(sorted(rows.values(), key=lambda r: (r["segmento"], int(r["id"]))))
        print(f"\nmetadata: {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
