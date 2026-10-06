#!/usr/bin/env python3
"""Cross-encoder scoring on the Mac's GPU (MPS), outside Docker.

Docker on macOS has no GPU, and its Linux torch skips Apple's Accelerate: bge-reranker
takes ~37 s per question in the container, ~4 s here. chunks-api keeps everything else
and sends only the (query, chunk) pairs, when RERANK_URL points here.

    POST /rerank  {"model": "<hf id>", "max_length": 512, "pairs": [[q, chunk], ...]}  ->  {"scores": [...]}

    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/python serve.py                 # :8001, models from ~/.cache/huggingface
"""

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
from sentence_transformers import CrossEncoder

PORT = int(os.environ.get("PORT", "8001"))
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
_models: dict[tuple[str, int], CrossEncoder] = {}


def model(name: str, max_length: int) -> CrossEncoder:
    if (name, max_length) not in _models:
        print(f"Loading {name} on {DEVICE}…", flush=True)
        # float32 like the container: same scores, runs stay comparable.
        _models[(name, max_length)] = CrossEncoder(name, max_length=max_length, device=DEVICE,
                                                   model_kwargs={"torch_dtype": torch.float32})
    return _models[(name, max_length)]


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/rerank":
            return self.reply(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            scores = model(body["model"], int(body["max_length"])).predict([tuple(p) for p in body["pairs"]])
            self.reply(200, {"scores": [float(s) for s in scores]})
        except Exception as e:
            self.reply(500, {"error": f"{type(e).__name__}: {e}"})

    def reply(self, code: int, payload: dict):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    # Single-threaded on purpose: one GPU, eval asks one question at a time.
    print(f"rerank server on :{PORT} ({DEVICE})", flush=True)
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
