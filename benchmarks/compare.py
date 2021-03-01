"""Throughput / latency comparison: EdgeScore (C++) vs the naive Python baseline.

Starts each server on localhost, drives it with N client processes that POST NDJSON batches over
keep-alive connections for a fixed duration, and prints a Markdown table. Both servers use the same
ONNX model and the same /v1/score request/response contract.

    python benchmarks/compare.py --seconds 10 --clients 2 --batch 50
"""
from __future__ import annotations

import argparse
import http.client
import json
import multiprocessing as mp
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from telemetry import FEATURES, labeled_mix  # noqa: E402


def make_payload(batch: int, seed: int) -> bytes:
    x, _ = labeled_mix(batch, 0.05, np.random.default_rng(seed))
    return ("\n".join(json.dumps(dict(zip(FEATURES, map(float, r)))) for r in x) + "\n").encode()


def client(port: int, path: str, payload: bytes, batch: int, seconds: float, out: mp.Queue):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    lat, events, rejected = [], 0, 0
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        t0 = time.perf_counter()
        conn.request("POST", path, payload, {"Content-Type": "application/x-ndjson"})
        resp = conn.getresponse()
        resp.read()
        dt = time.perf_counter() - t0
        if resp.status in (200, 202):
            lat.append(dt)
            events += batch
        elif resp.status == 503:
            rejected += batch
            time.sleep(0.01)
        else:
            raise RuntimeError(f"unexpected status {resp.status}")
    out.put((events, lat, rejected))


def wait_ready(port: int, proc: subprocess.Popen, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        if proc.poll() is not None:
            raise RuntimeError("server exited early")
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
            c.request("GET", "/healthz")
            if c.getresponse().status == 200:
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("server did not become ready")


def scrape(port: int) -> dict:
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    c.request("GET", "/metrics")
    text = c.getresponse().read().decode()
    return {m.group(1): float(m.group(2)) for m in re.finditer(r"^(\w+) ([0-9.eE+-]+)$", text, re.M)}


def drive(port, path, batch, seconds, clients):
    payload = make_payload(batch, 5)
    q: mp.Queue = mp.Queue()
    procs = [mp.Process(target=client, args=(port, path, payload, batch, seconds, q)) for _ in range(clients)]
    t0 = time.perf_counter()
    for p in procs:
        p.start()
    results = [q.get() for _ in procs]
    for p in procs:
        p.join()
    elapsed = time.perf_counter() - t0
    events = sum(r[0] for r in results)
    lat = np.array([x for r in results for x in r[1]])
    return {"events": events, "elapsed": elapsed, "eps": events / elapsed, "rejected": sum(r[2] for r in results),
            "req_p50_ms": float(np.percentile(lat, 50) * 1e3), "req_p99_ms": float(np.percentile(lat, 99) * 1e3)}


def run_server(cmd, env, port):
    proc = subprocess.Popen(cmd, env={**os.environ, **env}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)
    try:
        wait_ready(port, proc)
    except Exception:
        stop(proc)
        raise
    return proc


def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--clients", type=int, default=2)
    ap.add_argument("--batch", type=int, default=50, help="events per request")
    ap.add_argument("--workers", type=int, default=2, help="EdgeScore inference threads")
    ap.add_argument("--edgescore-bin", type=Path, default=ROOT / "build" / "edgescore")
    ap.add_argument("--model-dir", type=Path, default=ROOT / "models")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()
    if not args.edgescore_bin.exists():
        sys.exit(f"{args.edgescore_bin} not found: run `make build` first")

    rows = []
    # --- naive Python baseline
    proc = run_server([args.python, str(ROOT / "benchmarks" / "baseline_server.py"), "--port", "18089",
                       "--model-dir", str(args.model_dir)], {}, 18089)
    try:
        print("baseline ...", flush=True)
        r = drive(18089, "/v1/score", args.batch, args.seconds, args.clients)
        rows.append(("Python baseline (1 thread, per-event `session.run`)", r, None))
    finally:
        stop(proc)

    # --- EdgeScore, synchronous scoring endpoint (same contract as the baseline)
    env = {"EDGESCORE_PORT": "18088", "EDGESCORE_MODEL": str(args.model_dir / "model.onnx"),
           "EDGESCORE_WORKERS": str(args.workers), "EDGESCORE_BIND": "127.0.0.1"}
    proc = run_server([str(args.edgescore_bin)], env, 18088)
    try:
        print("edgescore /v1/score ...", flush=True)
        r = drive(18088, "/v1/score", args.batch, args.seconds, args.clients)
        m = scrape(18088)
        rows.append((f"EdgeScore `/v1/score` ({args.workers} workers, batched)", r, m))
    finally:
        stop(proc)

    # --- EdgeScore, HEC async ingest: throughput counted until every accepted event is scored
    proc = run_server([str(args.edgescore_bin)], env, 18088)
    try:
        print("edgescore HEC ...", flush=True)
        r = drive(18088, "/services/collector/event", args.batch, args.seconds, args.clients)
        t_end = time.perf_counter()
        while scrape(18088)["edgescore_events_scored_total"] < r["events"] and time.perf_counter() - t_end < 30:
            time.sleep(0.05)
        r["eps"] = r["events"] / (r["elapsed"] + (time.perf_counter() - t_end))
        rows.append((f"EdgeScore HEC `/services/collector/event` ({args.workers} workers, until drained)", r, scrape(18088)))
    finally:
        stop(proc)

    base = rows[0][1]["eps"]
    print(f"\nbatch={args.batch} events/request, clients={args.clients}, {args.seconds:.0f}s per run, "
          f"{os.cpu_count()} CPUs shared by client and server\n")
    print("| Server | events/s | speed-up | request p50 (ms) | request p99 (ms) | server-side event p50 / p99 (ms) | 503'd events |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name, r, m in rows:
        srv = "n/a" if m is None else f"{m['edgescore_latency_p50_seconds'] * 1e3:.2f} / {m['edgescore_latency_p99_seconds'] * 1e3:.2f}"
        print(f"| {name} | {r['eps']:,.0f} | {r['eps'] / base:.1f}x | {r['req_p50_ms']:.1f} | {r['req_p99_ms']:.1f} | {srv} | {r['rejected']} |")


if __name__ == "__main__":
    main()
