#!/usr/bin/env python3
"""
Downloads bill documents (Projetos de Lei / PLs) from the Brazilian Chamber
of Deputies using their Open Data API.

Flow:
  1. Search bills by year / topic (keyword).
  2. For each bill, fetch details (summary, status, full-text URL).
  3. Download the full text (PDF) and save metadata to a CSV.

Usage:
  pip install requests
  python download.py --year 2024 --topic "meio ambiente" --limit 20

Good first ingestion step for a RAG: you end up with a folder of PDFs plus a
metadata CSV, ready for chunking and indexing.

Note: API query params and response fields stay in Portuguese — that is the
upstream API contract, not a translation gap.
"""

import argparse
import csv
import time
from pathlib import Path

import requests

BASE = "https://dadosabertos.camara.leg.br/api/v2"
HEADERS = {"Accept": "application/json", "User-Agent": "estudo-rag-pls/1.0"}


def search_bills(year, topic, limit):
    """List bills for a year, optionally filtered by keyword (topic)."""
    bills = []
    page = 1
    while len(bills) < limit:
        params = {
            "ano": year,
            "itens": 100,
            "pagina": page,
            "ordem": "ASC",
            "ordenarPor": "id",
        }
        if topic:
            params["keywords"] = topic  # full-text search over summary/index terms
        r = requests.get(f"{BASE}/proposicoes", params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()["dados"]
        if not data:
            break  # no more pages
        bills.extend(data)
        page += 1
    return bills[:limit]


def details(bill_id):
    """Details of a bill: full summary, status and full-text URL."""
    r = requests.get(f"{BASE}/proposicoes/{bill_id}", headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()["dados"]


def download_file(url, target):
    """Download the full text (PDF/doc). Returns True if something was saved."""
    if not url:
        return False
    try:
        r = requests.get(url, headers={"User-Agent": HEADERS["User-Agent"]}, timeout=60)
        r.raise_for_status()
        target.write_bytes(r.content)
        return True
    except requests.RequestException as e:
        print(f"  ! failed to download {url}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(description="Download Chamber of Deputies bills for RAG use.")
    ap.add_argument("--year", type=int, required=True, help="Bill year (e.g. 2024)")
    ap.add_argument("--topic", default="", help='Keyword, e.g. "meio ambiente"')
    ap.add_argument("--limit", type=int, default=20, help="How many bills to download")
    ap.add_argument("--out", default="pls", help="Output folder")
    args = ap.parse_args()

    folder = Path(args.out)
    (folder / "pdfs").mkdir(parents=True, exist_ok=True)

    print(f"Searching up to {args.limit} bills from {args.year}"
          + (f' about "{args.topic}"' if args.topic else "") + " ...")
    bills = search_bills(args.year, args.topic, args.limit)
    print(f"Found {len(bills)} bills.\n")

    rows = []
    for i, b in enumerate(bills, 1):
        d = details(b["id"])
        name = f'PL_{d["numero"]}_{d["ano"]}'
        print(f'[{i}/{len(bills)}] {name} (id {d["id"]})')
        print(f"    summary: {(d.get('ementa') or '').strip()[:100]}")

        full_text_url = d.get("urlInteiroTeor")
        file_path = folder / "pdfs" / f"{name}.pdf"
        saved = download_file(full_text_url, file_path)

        rows.append({
            "id": d["id"],
            "label": f'PL {d["numero"]}/{d["ano"]}',
            "summary": (d.get("ementa") or "").strip(),
            "presented_at": d.get("dataApresentacao"),
            "status": (d.get("statusProposicao") or {}).get("descricaoSituacao"),
            "full_text_url": full_text_url,
            "file": str(file_path) if saved else "",
        })
        time.sleep(0.3)  # be gentle with the API

    if not rows:
        print("\nNothing found — no CSV written.")
        return

    csv_path = folder / "metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nDone. PDFs in: {folder/'pdfs'}")
    print(f"Metadata in: {csv_path}")
    print("Those two are the raw material for your RAG (documents + metadata).")


if __name__ == "__main__":
    main()
