"""RAG from scratch: pypdf + sentence-transformers + FAISS. No LangChain / LlamaIndex.
  python rag.py eval --pdf doc.pdf --questions questions.json --sizes 100 300 --k 1 3 5
  python rag.py ask  --pdf doc.pdf --q "your question" --size 200
questions.json: [{"question": "...", "gold_page": 12, "gold_snippet": "30-60 words copied from the source passage"}]
A retrieved chunk counts as a HIT if >= 70% of the snippet's distinct words appear in it."""
import re, json, argparse, urllib.request
import numpy as np, faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

EMB = "BAAI/bge-small-en-v1.5"
QPREFIX = "Represent this sentence for searching relevant passages: "   # bge query instruction

def load_words(pdf):
    words, pages = [], []
    for i, p in enumerate(PdfReader(pdf).pages, 1):
        for w in (p.extract_text() or "").split():
            words.append(w); pages.append(i)
    return words, pages

def chunk(words, pages, size, overlap_frac=0.2):
    step = max(1, int(size * (1 - overlap_frac))); out = []
    for s in range(0, len(words), step):
        out.append({"text": " ".join(words[s:s + size]), "p0": pages[s], "p1": pages[min(s + size, len(words)) - 1]})
        if s + size >= len(words): break
    return out

def toks(s): return set(re.findall(r"[a-z0-9]+", s.lower()))

def is_hit(chunk_text, snippet):
    sn = toks(snippet)
    return len(sn & toks(chunk_text)) / max(1, len(sn)) >= 0.7

def build_index(chunks, model):
    emb = model.encode([c["text"] for c in chunks], normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    idx = faiss.IndexFlatIP(emb.shape[1]); idx.add(np.asarray(emb, dtype="float32"))
    return idx

def search(q, idx, model, k):
    qv = model.encode([QPREFIX + q], normalize_embeddings=True)
    sims, ids = idx.search(np.asarray(qv, dtype="float32"), k)
    return list(zip(ids[0].tolist(), sims[0].tolist()))

def call_llm(prompt, model="qwen2.5:1.5b-instruct-q4_K_M"):
    """Plug in any LLM. Default: a local Ollama server. Replace with an API call if you prefer."""
    req = urllib.request.Request("http://localhost:11434/api/generate", method="POST",
        data=json.dumps({"model": model, "prompt": prompt, "stream": False}).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())["response"]

def evaluate(a):
    words, pages = load_words(a.pdf); model = SentenceTransformer(EMB)
    qs = json.load(open(a.questions)); report = {}
    print(f"{len(words)} words, ~{pages[-1]} pages, {len(qs)} questions")
    for size in a.sizes:
        chunks = chunk(words, pages, size); idx = build_index(chunks, model)
        hits = {k: 0 for k in a.k}; fails = []
        for q in qs:
            res = search(q["question"], idx, model, max(a.k))
            ok = [is_hit(chunks[i]["text"], q["gold_snippet"]) for i, _ in res]
            for k in a.k: hits[k] += any(ok[:k])
            if not any(ok[:max(a.k)]):
                fails.append({"question": q["question"], "gold_page": q.get("gold_page"),
                              "top3": [{"page": chunks[i]["p0"], "score": round(s, 3), "text": chunks[i]["text"][:200]} for i, s in res[:3]]})
        report[size] = {"n_chunks": len(chunks), **{f"recall@{k}": hits[k] / len(qs) for k in a.k}, "failures": fails}
        print(size, {k: v for k, v in report[size].items() if k != "failures"})
    json.dump(report, open("rag_results.json", "w"), indent=2)
    print("Saved rag_results.json (look at the 'failures' lists for your 2 failure cases)")

def ask(a):
    words, pages = load_words(a.pdf); model = SentenceTransformer(EMB)
    chunks = chunk(words, pages, a.size); idx = build_index(chunks, model)
    ctx = [chunks[i] for i, _ in search(a.q, idx, model, 5)]
    context = "\n\n".join(f"[p.{c['p0']}] {c['text']}" for c in ctx)
    prompt = f"Answer using ONLY the context. Cite page numbers. If it is not in the context, say so.\n\nContext:\n{context}\n\nQuestion: {a.q}\nAnswer:"
    print(call_llm(prompt))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval"); e.add_argument("--pdf", required=True); e.add_argument("--questions", required=True)
    e.add_argument("--sizes", type=int, nargs="+", default=[100, 300]); e.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    q = sub.add_parser("ask"); q.add_argument("--pdf", required=True); q.add_argument("--q", required=True); q.add_argument("--size", type=int, default=200)
    a = ap.parse_args(); evaluate(a) if a.cmd == "eval" else ask(a)
