#!/usr/bin/env python3
"""Serve the report and run the eval from it.

`python -m http.server` only reads. The report grew a "Rodar" button, so it
needed something that answers POST. This is that same static server plus one
endpoint:

    GET  /api/options                          ->  {"buckets": [...]}
    POST /api/run  {"bucket", "label", "note", "k"}  ->  {"run_id": ...}

run.py is imported, not shelled out to: same process, same runs/ directory, and
the exit-with-a-message paths (API down, dataset missing) come back as SystemExit
and turn into the error the browser shows.
"""

import json
import os
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import run as evaluation

HERE = str(Path(__file__).resolve().parent)
PORT = int(os.environ.get("PORT", 8080))

# One eval at a time: run_id is a second-resolution timestamp, so two concurrent
# runs would race for the same filename, and they would race for the API too.
busy = threading.Lock()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=HERE, **kw)

    def do_GET(self):
        if self.path.rstrip("/") != "/api/options":
            return super().do_GET()
        try:
            buckets = [b for b in evaluation.list_buckets(evaluation.API)
                       if b["status"] == "ready" and b.get("qrels_count")]
            self.reply(200, {"buckets": buckets})
        except SystemExit as e:
            self.reply(502, {"error": str(e.code)})
        except Exception as e:
            self.reply(500, {"error": f"{type(e).__name__}: {e}"})

    def do_POST(self):
        if self.path.rstrip("/") != "/api/run":
            return self.reply(404, {"error": "not found"})
        if not busy.acquire(blocking=False):
            return self.reply(409, {"error": "já tem uma avaliação rodando"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            if not body.get("bucket"):
                return self.reply(400, {"error": "bucket é obrigatório"})
            payload = evaluation.run(
                str(body.get("label", "")),
                str(body.get("note", "")),
                max(1, min(50, int(body.get("k", 5)))),
                evaluation.API,
                str(body["bucket"]),
            )
            self.reply(200, {"run_id": payload["run_id"], "metrics": payload["metrics"]})
        except SystemExit as e:  # run.py's sys.exit("chunks-api unreachable…", sha mismatch…)
            self.reply(502, {"error": str(e.code)})
        except Exception as e:
            self.reply(500, {"error": f"{type(e).__name__}: {e}"})
        finally:
            busy.release()

    def reply(self, code: int, obj: dict):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print(f"eval ui on http://localhost:{PORT}/  (api {evaluation.API})", flush=True)
    ThreadingHTTPServer(("", PORT), Handler).serve_forever()
