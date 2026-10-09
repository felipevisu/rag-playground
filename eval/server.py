#!/usr/bin/env python3
"""Serve the report and run the eval from it.

`python -m http.server` only reads. The report grew a "Rodar" button, so it
needed something that answers POST. This is that same static server plus one
endpoint:

    GET  /api/options                          ->  {"buckets": [...], "descriptions": [...], "rerankers": {...}, "transform": {...}}
    GET  /api/progress                         ->  {"running", "done", "total", "query_id", "started"}
    POST /api/cancel                           ->  stops the run before its next question
    POST /api/run  {"bucket", "k", "rerank", "variants", "docs", "doc_top", "rerank_depth", "rerank_keep", "sample"}  ->  202, runs in a thread;
                                               the outcome lands in /api/progress as "run_id" or "error"
    DELETE /api/runs/<run_id>                  ->  removes runs/<run_id>.json, regenerates index.json

run.py is imported, not shelled out to: same process, same runs/ directory, and
the exit-with-a-message paths (API down, dataset missing) come back as SystemExit
and turn into the error the browser shows. The run doesn't hold the POST open:
an hour-long idle request died to laptop sleep or a dropped port forward
("Failed to fetch") while the run went on unseen.
"""

import json
import os
import re
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import run as evaluation

HERE = str(Path(__file__).resolve().parent)
PORT = int(os.environ.get("PORT", 8080))

# One eval at a time: run_id is a second-resolution timestamp, so two concurrent
# runs would race for the same filename, and they would race for the API too.
busy = threading.Lock()
RUN_ID = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}")  # also keeps the path inside runs/


def run_in_background(*args):
    try:
        evaluation.PROGRESS["run_id"] = evaluation.run(*args)["run_id"]
    except SystemExit as e:  # run.py's sys.exit("chunks-api unreachable…", sha mismatch, cancelado…)
        evaluation.PROGRESS["error"] = str(e.code)
    except Exception as e:
        evaluation.PROGRESS["error"] = f"{type(e).__name__}: {e}"
    finally:
        busy.release()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=HERE, **kw)

    def do_GET(self):
        if self.path.rstrip("/") == "/api/progress":
            return self.reply(200, {**evaluation.PROGRESS, "running": busy.locked()})
        if self.path.rstrip("/") != "/api/options":
            return super().do_GET()
        try:
            everything = evaluation.list_buckets(evaluation.API)
            buckets = [b for b in everything if b["status"] == "ready" and b.get("qrels_count")]
            descriptions = [b for b in everything if b["status"] == "ready" and b.get("kind") == "descriptions"] \
                if evaluation.two_layer_info(evaluation.API) else None  # null before week09
            self.reply(200, {"buckets": buckets, "descriptions": descriptions,
                             "two_layer": evaluation.two_layer_info(evaluation.API),
                             "rerankers": evaluation.list_rerankers(evaluation.API),
                             "rerank_tuning": evaluation.rerank_tuning_info(evaluation.API),
                             "transform": evaluation.transform_info(evaluation.API)})
        except SystemExit as e:
            self.reply(502, {"error": str(e.code)})
        except Exception as e:
            self.reply(500, {"error": f"{type(e).__name__}: {e}"})

    def do_POST(self):
        if self.path.rstrip("/") == "/api/cancel":
            evaluation.PROGRESS["cancel"] = busy.locked()
            return self.reply(200, {"cancel": evaluation.PROGRESS["cancel"]})
        if self.path.rstrip("/") != "/api/run":
            return self.reply(404, {"error": "not found"})
        if not busy.acquire(blocking=False):
            return self.reply(409, {"error": "já tem uma avaliação rodando"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            if not body.get("bucket"):
                busy.release()
                return self.reply(400, {"error": "bucket é obrigatório"})
            args = (
                max(1, min(50, int(body.get("k", 5)))),
                evaluation.API,
                str(body["bucket"]),
                str(body.get("rerank") or ""),
                int(body.get("variants") or 0),
                str(body.get("docs") or ""),
                max(0, min(50, int(body.get("doc_top") or 0))),
                max(0, min(100, int(body.get("rerank_depth") or 0))),
                None if body.get("rerank_keep") in (None, "") else max(0.0, min(10.0, float(body["rerank_keep"]))),
                max(1, min(100, int(body.get("sample") or 100))),
            )
        except Exception as e:  # bad body: nothing started
            busy.release()
            return self.reply(400, {"error": f"{type(e).__name__}: {e}"})
        evaluation.PROGRESS.update(run_id=None, error=None)
        threading.Thread(target=run_in_background, args=args, daemon=True).start()
        self.reply(202, {"started": True})

    def do_DELETE(self):
        run_id = self.path.rstrip("/").removeprefix("/api/runs/")
        if not self.path.startswith("/api/runs/") or not RUN_ID.fullmatch(run_id):
            return self.reply(404, {"error": "not found"})
        path = evaluation.RUNS / f"{run_id}.json"
        if not path.exists():
            return self.reply(404, {"error": f"execução {run_id} não existe"})
        path.unlink()
        evaluation.reindex()
        self.reply(200, {"deleted": run_id})

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
