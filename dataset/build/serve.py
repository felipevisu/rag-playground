#!/usr/bin/env python3
"""Local UI for the chunker: pick a config, build, browse chunks, download.

  python serve.py            # http://localhost:8765
"""

import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from build import OUT, ROOT, build
from chunkers import Config, SPLITTERS, TOKENIZERS, UNITS

HERE = Path(__file__).resolve().parent
PORT = 8765


def list_builds() -> list[dict]:
    out = []
    for m in sorted(OUT.glob("*/manifest.json")):
        d = json.loads(m.read_text(encoding="utf-8"))
        out.append({"name": m.parent.name, **d})
    return out


def read_chunks(name: str) -> list[dict]:
    df = pd.read_parquet(OUT / name / "corpus.parquet")
    df["pages"] = df["pages"].apply(lambda p: [int(x) for x in p])
    ans = pd.read_parquet(OUT / name / "answers.parquet")
    by_chunk = ans.groupby("chunk_id")["query_id"].apply(lambda s: sorted(set(s)))
    df["queries"] = df["chunk_id"].map(by_chunk).apply(lambda v: v if isinstance(v, list) else [])
    return json.loads(df.to_json(orient="records", force_ascii=False))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter
        if "/api/" in str(args[0] if args else ""):
            super().log_message(fmt, *args)

    def send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path, ctype: str, download: str | None = None):
        if not path.is_file():
            return self.send_json({"error": "not found"}, 404)
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path == "/":
            return self.send_file(HERE / "ui.html", "text/html; charset=utf-8")
        if url.path == "/api/options":
            return self.send_json({"splitters": SPLITTERS, "units": UNITS, "tokenizers": TOKENIZERS})
        if url.path == "/api/builds":
            return self.send_json(list_builds())
        if url.path == "/api/chunks":
            return self.send_json(read_chunks(q["name"]))
        if url.path.startswith("/out/"):
            name, _, fname = url.path[5:].partition("/")
            if fname == "corpus.jsonl":
                rows = read_chunks(name)
                body = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/jsonl")
                self.send_header("Content-Disposition", f'attachment; filename="{name}-corpus.jsonl"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                return self.wfile.write(body)
            if fname in ("corpus.parquet", "answers.parquet", "manifest.json", "config.yaml"):
                return self.send_file(OUT / name / fname, "application/octet-stream", f"{name}-{fname}")
        self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/api/build":
            return self.send_json({"error": "not found"}, 404)
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            cfg = Config(**json.loads(raw))
            cfg.validate()
            out = build(cfg)
            (ROOT / "configs").mkdir(exist_ok=True)
            (ROOT / "configs" / f"{cfg.name}.yaml").write_bytes((out / "config.yaml").read_bytes())
            self.send_json(json.loads((out / "manifest.json").read_text(encoding="utf-8")))
        except (AssertionError, TypeError, ValueError) as e:
            self.send_json({"error": str(e)}, 400)
        except Exception as e:
            traceback.print_exc()
            self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    host = os.environ.get("HOST", "127.0.0.1")  # docker sets 0.0.0.0
    print(f"chunker ui: http://localhost:{port}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
