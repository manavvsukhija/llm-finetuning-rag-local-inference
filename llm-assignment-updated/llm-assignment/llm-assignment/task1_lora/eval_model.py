"""Evaluate on test.jsonl. Modes: zeroshot (base), fewshot (base + 3 examples), lora (adapter).
  python eval_model.py --mode zeroshot
  python eval_model.py --mode fewshot
  python eval_model.py --mode lora --adapter lora_r16
Metrics: valid_json, exact_match (whole object equal), field_acc (share of gold fields correct)."""
import json, argparse, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
PROMPT = "### Request:\n{text}\n### JSON:\n"
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--mode", choices=["zeroshot", "fewshot", "lora"], required=True)
ap.add_argument("--adapter", default=None)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16).cuda()
if a.mode == "lora":
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter)
model.eval()

test = [json.loads(l) for l in open("test.jsonl", encoding="utf-8")]
train = [json.loads(l) for l in open("train.jsonl", encoding="utf-8")]
shots = ""
if a.mode == "fewshot":
    seen_intents, picked = set(), []
    for r in train:
        i = json.loads(r["target"])["intent"]
        if i not in seen_intents: seen_intents.add(i); picked.append(r)
    shots = "".join(PROMPT.format(text=r["text"]) + r["target"] + "\n\n" for r in picked)

def gen(text):
    ids = tok(shots + PROMPT.format(text=text), return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=100, do_sample=False, pad_token_id=tok.pad_token_id)
    s = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return s.split("###")[0].split("\n\n")[0].strip()

def parse(s):
    for cand in (s, s.split("\n")[0]):
        try:
            o = json.loads(cand)
            if isinstance(o, dict): return o
        except Exception: pass
    return None

valid = exact = 0; field = 0.0; preds = []
for r in test:
    pred_s = gen(r["text"]); gold = json.loads(r["target"]); p = parse(pred_s)
    if p is not None:
        valid += 1
        field += sum(p.get(k) == v for k, v in gold.items()) / len(gold)
        exact += p == gold
    preds.append({"text": r["text"], "gold": gold, "pred_raw": pred_s, "heldout_template": r["heldout_template"],
                  "correct": p == gold})
n = len(test)
res = {"mode": a.mode, "n": n, "valid_json": valid / n, "exact_match": exact / n, "field_acc": field / n}
for h in (False, True):
    sub = [x for x in preds if x["heldout_template"] == h]
    res[f"exact_match_{'heldout' if h else 'seen'}_templates"] = sum(x["correct"] for x in sub) / max(1, len(sub))
print(json.dumps(res, indent=2))
tag = a.mode if a.mode != "lora" else f"lora_{a.adapter}"
json.dump({"metrics": res, "predictions": preds}, open(f"results_{tag}.json", "w"), indent=2, ensure_ascii=False)
print("failures to inspect:", [x["text"] for x in preds if not x["correct"]][:5])
