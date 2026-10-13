"""Compare PyTorch vs ONNX fp32 vs fused ONNX vs ONNX int8 scores on handwritten relevance data."""
import sys, time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from convert import MODEL_NAME, REVISION, load_native_model  # noqa: E402

M = Path(__file__).parent / "models"

# (query, relevant doc, [distractors])
DATA = [
 ("What is the capital of France?", "Paris is the capital and largest city of France.",
  ["Berlin is the capital of Germany.", "The Eiffel Tower was completed in 1889.", "France is famous for its wine and cheese."]),
 ("How do I reset my router?", "To reset a router, hold the reset button for 10 seconds until the lights blink.",
  ["Routers forward packets between networks.", "Wi-Fi 6 offers higher throughput than Wi-Fi 5.", "Use a surge protector to guard electronics."]),
 ("symptoms of vitamin D deficiency", "Low vitamin D can cause fatigue, bone pain, muscle weakness and mood changes.",
  ["Vitamin C is found in citrus fruits.", "Sunlight exposure helps the body make vitamin D.", "Iron deficiency causes anemia."]),
 ("python list comprehension syntax", "A list comprehension looks like [expr for item in iterable if cond] and builds a new list.",
  ["Python was created by Guido van Rossum.", "Tuples in Python are immutable sequences.", "Use pip to install Python packages."]),
 ("who wrote Pride and Prejudice", "Pride and Prejudice is an 1813 novel by Jane Austen.",
  ["Charles Dickens wrote Great Expectations.", "The novel is set in rural England.", "Jane Eyre was written by Charlotte Bronte."]),
 ("how does photosynthesis work", "Photosynthesis converts light, water and carbon dioxide into glucose and oxygen inside chloroplasts.",
  ["Cellular respiration releases energy from glucose.", "Plants need nitrogen for growth.", "Chlorophyll gives leaves their green color."]),
 ("best way to learn guitar chords", "Beginners should start with open chords like G, C and D and practice smooth transitions daily.",
  ["The guitar has six strings in standard tuning.", "Fender made the Stratocaster in 1954.", "Pianos have 88 keys."]),
 ("what causes inflation", "Inflation occurs when demand outpaces supply or when money supply grows faster than output, raising general prices.",
  ["The central bank sets interest rates.", "Unemployment measures people without jobs.", "Stock markets fluctuate daily."]),
 ("treatment for a sprained ankle", "For a sprained ankle use rest, ice, compression and elevation, and see a doctor if swelling persists.",
  ["Ankle bones include the talus and tibia.", "Running shoes should be replaced every 500 miles.", "A broken wrist needs a cast."]),
 ("how to boil an egg", "Place eggs in boiling water for 9 to 12 minutes, then cool them in ice water before peeling.",
  ["Chickens lay about one egg per day.", "Eggs are a good source of protein.", "Bake bread at 220 degrees Celsius."]),
]
pairs = [(q, d) for q, rel, ds in DATA for d in [rel] + ds]
queries = [q for q, _ in pairs]; docs = [d for _, d in pairs]
tok = AutoTokenizer.from_pretrained(MODEL_NAME, revision=REVISION)

def sig(x): return 1 / (1 + np.exp(-x))

def run_torch():
    m = load_native_model()
    out = []
    with torch.inference_mode():
        for i in range(0, len(pairs), 16):
            e = tok(queries[i:i+16], docs[i:i+16], padding=True, truncation=True, max_length=512, return_tensors="pt")
            out += m(**e).logits.view(-1).tolist()
    return np.array(out)

def run_onnx(path):
    so = ort.SessionOptions(); so.intra_op_num_threads = 8
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    out = []
    for i in range(0, len(pairs), 16):
        e = tok(queries[i:i+16], docs[i:i+16], padding=True, truncation=True, max_length=512, return_tensors="np")
        feed = {"input_ids": e["input_ids"].astype(np.int64), "attention_mask": e["attention_mask"].astype(np.int64)}
        out += s.run(None, {x.name: feed[x.name] for x in s.get_inputs()})[0].reshape(-1).tolist()
    return np.array(out)

def spearman(a, b):
    ra, rb = a.argsort().argsort(), b.argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])

def per_query(s):  # (spearman avg, top1 idx list)
    s = s.reshape(len(DATA), 4)
    return s.argmax(1)

ref = run_torch()
res = {"onnx_fp32": run_onnx(M / "model_fp32.onnx"), "onnx_opt": run_onnx(M / "model_opt.onnx"),
       "onnx_int8": run_onnx(M / "model_int8.onnx")}
# Jina's own published int8 ONNX, if downloaded for comparison (see README)
if (M / "jina_official" / "model_int8.onnx").exists():
    res["jina_official_int8"] = run_onnx(M / "jina_official" / "model_int8.onnx")
ref_top = per_query(ref)
print(f"torch top-1 correct (relevant doc ranked first): {(ref_top==0).sum()}/{len(DATA)}")
ok = True
for k, s in res.items():
    d = np.abs(sig(s) - sig(ref))
    sp = spearman(s, ref)
    agree = (per_query(s) == ref_top).mean()
    correct = (per_query(s) == 0).sum()
    print(f"{k}: spearman={sp:.4f} top1_agree_with_torch={agree:.0%} top1_correct={correct}/{len(DATA)} "
          f"mean|dscore|={d.mean():.4f} max|dscore|={d.max():.4f}")
    if k == "onnx_int8" and (sp < 0.98 or agree < 0.95): ok = False
print("PASS" if ok else "FAIL (int8 below bar: spearman>=0.98, top-1 agreement>=95%)")
