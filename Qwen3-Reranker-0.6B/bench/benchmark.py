"""Latency benchmark for the reranker API.

Usage: python bench/benchmark.py [--url http://localhost:8904] [--requests 20] [--quick]
"""
import argparse
import csv
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

WORDS = ("data model query search rank index token vector cloud server latency network "
         "python system memory cache batch score document passage retrieval embedding "
         "pipeline context answer question language neural transformer").split()
LENGTHS = {"short": 25, "medium": 150, "long": 450}  # approx words per doc


def make_text(n, rng):
    return " ".join(rng.choice(WORDS) for _ in range(n))


def make_payload(n_docs, length, rng):
    return {"query": make_text(12, rng), "documents": [make_text(LENGTHS[length], rng) for _ in range(n_docs)]}


def pct(vals, p):
    s = sorted(vals)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def one_call(client, url, payload):
    t0 = time.perf_counter()
    try:
        r = client.post(f"{url}/rerank", json=payload, timeout=300)
        r.raise_for_status()
        return (time.perf_counter() - t0) * 1000, r.json()["latency_ms"], True
    except Exception:
        return (time.perf_counter() - t0) * 1000, 0.0, False


def run_scenario(url, n_docs, length, conc, n_req, warmup=3):
    rng = random.Random(42)
    payloads = [make_payload(n_docs, length, rng) for _ in range(n_req)]
    with httpx.Client() as client:
        for _ in range(warmup):
            one_call(client, url, payloads[0])
        t0 = time.perf_counter()
        with ThreadPoolExecutor(conc) as ex:
            res = list(ex.map(lambda p: one_call(client, url, p), payloads))
        wall = time.perf_counter() - t0
    ok = [r for r in res if r[2]]
    lat = [r[0] for r in ok] or [0]
    srv = [r[1] for r in ok] or [0]
    return {
        "docs": n_docs, "length": length, "concurrency": conc, "requests": n_req,
        "p50_ms": round(pct(lat, 50), 1), "p95_ms": round(pct(lat, 95), 1), "p99_ms": round(pct(lat, 99), 1),
        "mean_ms": round(statistics.mean(lat), 1), "min_ms": round(min(lat), 1), "max_ms": round(max(lat), 1),
        "server_mean_ms": round(statistics.mean(srv), 1),
        "req_per_s": round(len(ok) / wall, 2), "docs_per_s": round(len(ok) * n_docs / wall, 1),
        "error_rate": round(1 - len(ok) / n_req, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8904")
    ap.add_argument("--requests", type=int, default=20)
    ap.add_argument("--quick", action="store_true", help="smaller scenario grid")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    a = ap.parse_args()

    health = httpx.get(f"{a.url}/health", timeout=10).json()
    print("Server:", health)

    docs_set = [1, 10, 50] if a.quick else [1, 10, 50, 100]
    lengths = ["short", "medium"] if a.quick else ["short", "medium", "long"]
    conc_set = [1, 4] if a.quick else [1, 4, 8]

    rows = []
    # Grid 1: docs x length at concurrency 1 (pure model latency)
    for length in lengths:
        for n in docs_set:
            if length == "long" and n == 100 and not a.quick:
                reqs = max(5, a.requests // 4)
            else:
                reqs = a.requests
            rows.append(run_scenario(a.url, n, length, 1, reqs))
            print(rows[-1])
    # Grid 2: concurrency at fixed 10 docs / medium
    for c in conc_set[1:]:
        rows.append(run_scenario(a.url, 10, "medium", c, a.requests * 2))
        print(rows[-1])

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    cols = list(rows[0].keys())
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    out.with_suffix(".md").write_text(f"Server: {health}\n\n" + "\n".join(md) + "\n")
    print(f"\nSaved {out.with_suffix('.csv')} and {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
