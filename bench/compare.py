"""Start each backend on port 8902, run the same scenarios, print/save a comparison table."""
import os, subprocess, sys, time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from benchmark import run_scenario

ROOT = Path(__file__).parent.parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"
URL = "http://localhost:8902"
BACKENDS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["torch", "onnx", "onnx-int8"]
SCEN = [(1, "medium"), (10, "medium"), (50, "medium")]
REQS = 3

rows = []
for b in BACKENDS:
    env = dict(os.environ, RERANK_BACKEND=b, RERANK_PORT="8902", RERANK_LOG_FILE=str(ROOT / "logs" / f"bench_{b}.jsonl"))
    p = subprocess.Popen([str(PY), str(ROOT / "app.py")], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    try:
        while True:
            try:
                if httpx.get(f"{URL}/health", timeout=2).json().get("status") == "ok":
                    break
            except Exception:
                pass
            time.sleep(2)
        load = time.time() - t0
        for n, ln in SCEN:
            r = run_scenario(URL, n, ln, 1, REQS, warmup=1)
            r["backend"] = b; r["start_s"] = round(load, 1)
            rows.append(r); print(b, n, ln, r["p50_ms"], "ms", flush=True)
    finally:
        p.terminate(); p.wait()

base = {r["docs"]: r["p50_ms"] for r in rows if r["backend"] == BACKENDS[0]}
lines = ["| backend | docs | p50 ms | p95 ms | docs/s | speedup vs " + BACKENDS[0] + " | start s |", "|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['backend']} | {r['docs']} | {r['p50_ms']} | {r['p95_ms']} | {r['docs_per_s']} | {base[r['docs']]/r['p50_ms']:.2f}x | {r['start_s']} |")
out = Path(__file__).parent / "compare.md"
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
