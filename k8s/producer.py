"""Demo log-producing app: emits synthetic host telemetry as JSON log lines on stdout and ships the
same events to the EdgeScore sidecar over the Splunk-HEC-compatible endpoint (pod-local, via localhost).

Stdlib only. Configuration via environment:
  PRODUCER_TARGET        default http://127.0.0.1:8088/services/collector/event
  PRODUCER_RATE          events per second (default 200)
  PRODUCER_ANOMALY_RATE  fraction of injected incidents (default 0.03)
  EDGESCORE_HEC_TOKEN    optional HEC token (sent as `Authorization: Splunk <token>`)
"""
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

TARGET = os.environ.get("PRODUCER_TARGET", "http://127.0.0.1:8088/services/collector/event")
RATE = float(os.environ.get("PRODUCER_RATE", "200"))
ANOMALY_RATE = float(os.environ.get("PRODUCER_ANOMALY_RATE", "0.03"))
TOKEN = os.environ.get("EDGESCORE_HEC_TOKEN", "")
HOST = os.environ.get("HOSTNAME", "demo-app")
BATCH_SECONDS = 0.25


def sample(rng):
    req = max(20.0, rng.gauss(200, 30))
    cpu = min(99.0, max(1.0, 18 + 0.16 * req + rng.gauss(0, 3)))
    mem = min(95.0, max(5.0, rng.gauss(45, 5)))
    lat = 35 + 0.10 * cpu + rng.gammavariate(4.0, 3.0)
    err = rng.betavariate(1.0, 400.0) * 100.0
    out = max(1.0, req * 4 + rng.gauss(0, 25))
    if rng.random() < ANOMALY_RATE:
        kind = rng.randrange(4)
        if kind == 0:
            cpu, lat = rng.uniform(92, 100), lat * rng.uniform(2, 4)
        elif kind == 1:
            lat, err = rng.uniform(300, 1500), rng.uniform(8, 40)
        elif kind == 2:
            req, out = rng.uniform(0, 25), out * 0.05
        else:
            mem, lat = rng.uniform(92, 99), lat + 25
    return {"cpu_pct": round(cpu, 2), "mem_pct": round(mem, 2), "latency_ms": round(lat, 2),
            "error_rate": round(err, 3), "req_rate": round(req, 1), "bytes_out_kb": round(out, 1)}


def ship(batch):
    body = "\n".join(json.dumps({"time": time.time(), "host": HOST, "sourcetype": "telemetry", "event": e})
                     for e in batch).encode()
    req = urllib.request.Request(TARGET, body, {"Content-Type": "application/json"})
    if TOKEN:
        req.add_header("Authorization", "Splunk " + TOKEN)
    try:
        urllib.request.urlopen(req, timeout=5).read()
        return True
    except urllib.error.HTTPError as e:
        print(json.dumps({"level": "warn", "msg": "sidecar rejected batch", "status": e.code}), flush=True)
    except OSError as e:
        print(json.dumps({"level": "warn", "msg": "sidecar unreachable", "error": str(e)}), flush=True)
    return False


def main():
    rng = random.Random(1234)
    per_batch = max(1, int(RATE * BATCH_SECONDS))
    sent = failed = 0
    next_tick = time.monotonic()
    while True:
        batch = [sample(rng) for _ in range(per_batch)]
        for e in batch:
            print(json.dumps({"level": "info", "msg": "telemetry", **e}), flush=True)
        if ship(batch):
            sent += len(batch)
        else:
            failed += len(batch)
        if (sent + failed) % (per_batch * 20) == 0:
            print(json.dumps({"level": "info", "msg": "producer stats", "sent": sent, "failed": failed}), flush=True)
        next_tick += BATCH_SECONDS
        time.sleep(max(0.0, next_tick - time.monotonic()))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
