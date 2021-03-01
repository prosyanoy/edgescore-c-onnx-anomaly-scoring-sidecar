"""Deliberately naive Python baseline for the benchmark comparison.

Same API as EdgeScore's /v1/score (NDJSON in, NDJSON out) and the *same ONNX model*, but written the
way a quick prototype would be: a single-threaded stdlib HTTP server, json.loads per line, and one
onnxruntime call per event (no batching, no worker pool, no queue).
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import onnxruntime as ort


def make_handler(sess, meta):
    features, sign, thr = meta["features"], meta["score_sign"], meta["threshold"]
    in_name = meta["input_name"]

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # silence per-request logging
            pass

        def do_GET(self):
            body = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            out = []
            for i, line in enumerate(raw.splitlines()):
                if not line.strip():
                    continue
                ev = json.loads(line)
                row = np.array([[float(ev[f]) for f in features]], dtype=np.float32)
                score = sign * float(sess.run(None, {in_name: row})[0][0][0])
                out.append(json.dumps({"seq": i, "score": round(score, 6), "anomaly": score > thr}))
            body = ("\n".join(out) + "\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8089)
    ap.add_argument("--model-dir", type=Path, default=Path("models"))
    args = ap.parse_args()
    meta = json.loads((args.model_dir / "model.json").read_text())
    sess = ort.InferenceSession(str(args.model_dir / "model.onnx"), providers=["CPUExecutionProvider"])
    HTTPServer(("127.0.0.1", args.port), make_handler(sess, meta)).serve_forever()


if __name__ == "__main__":
    main()
