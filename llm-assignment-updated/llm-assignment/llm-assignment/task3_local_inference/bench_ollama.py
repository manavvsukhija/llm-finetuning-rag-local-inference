"""Benchmark two Ollama model tags on the same prompts.
  ollama pull <tagA>; ollama pull <tagB>    (check exact tags on ollama.com/library/qwen2.5)
  python bench_ollama.py qwen2.5:1.5b-instruct-fp16 qwen2.5:1.5b-instruct-q4_K_M
Writes bench_summary.csv and outputs_<tag>.json (read the outputs side by side for the quality comparison)."""
import sys, json, csv, subprocess, urllib.request, statistics, re

def api(model, prompt, num_predict=200):
    req = urllib.request.Request("http://localhost:11434/api/generate", method="POST",
        data=json.dumps({"model": model, "prompt": prompt, "stream": False,
                         "options": {"temperature": 0, "num_predict": num_predict}}).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())

def sh(cmd):
    try: return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception: return ""

prompts = [p.strip() for p in open("prompts.txt", encoding="utf-8") if p.strip()]
rows = []
for model in sys.argv[1:]:
    api(model, "Hi", 5)                                    # warm-up: loads the model
    ps = sh(["ollama", "ps"]); gpu = sh(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"])
    speeds, outs = [], []
    for p in prompts:
        r = api(model, p); tps = r["eval_count"] / (r["eval_duration"] / 1e9)
        speeds.append(tps); outs.append({"prompt": p, "output": r["response"], "tokens": r["eval_count"], "tok_per_s": round(tps, 1)})
    json.dump(outs, open("outputs_" + re.sub(r"[^\w.-]", "_", model) + ".json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    rows.append({"model": model, "mean_tok_per_s": round(statistics.mean(speeds), 1), "min": round(min(speeds), 1),
                 "max": round(max(speeds), 1), "gpu_memory_used_MiB_after_load": gpu, "ollama_ps": ps.replace("\n", " | ")})
    print(rows[-1])
    sh(["ollama", "stop", model])                     # unload so the next model's memory reading is not inflated
with open("bench_summary.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
