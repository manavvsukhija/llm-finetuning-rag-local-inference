# Worked calculations, concept notes, run log and interview prep

Author: Manav Sukhija. The first part holds the worked calculations and concept notes. Sections 4–8, after the divider, record what my own runs produced and what I learned from them.

> The assignment says "in your own words" for several answers. The primers below are to
> make sure you understand each idea. Close this file and write those answers yourself.

## Task 1 (d): GPU memory to fine-tune a 7B model (7e9 parameters)

**(a) Full fine-tuning, AdamW, mixed precision**
| item | bytes/param | GB |
|---|---|---|
| fp16 weights | 2 | 14 |
| fp16 gradients | 2 | 14 |
| fp32 master weights | 4 | 28 |
| Adam m (fp32) | 4 | 28 |
| Adam v (fp32) | 4 | 28 |
| **total** | **16** | **112** (+ activations, roughly 5-20 GB) |

So about 115-130 GB: needs several 80 GB GPUs (or FSDP / ZeRO sharding).

**(b) LoRA** (base frozen in fp16)
- Base weights: 7e9 x 2 B = 14 GB. No gradients or optimizer state for the base.
- LoRA params, r=16 on all 7 linear layers of a Llama-7B-style model (hidden 4096, MLP 11008, 32 layers):
  - attention q,k,v,o: 4 x r x (4096+4096) = 4 x 16 x 8192 = 524,288
  - MLP gate, up, down: 3 x r x (4096+11008) = 3 x 16 x 15,104 = 724,992
  - per layer 1,249,280 x 32 layers = **39,976,960 ~ 40M trainable params**
- Trainable state: 40M x (4 weights + 4 grads + 8 Adam) = 40M x 16 B = **0.64 GB**
- Activations with gradient checkpointing, small batch, ~512 tokens: ~2-4 GB
- **Total ~ 14 + 0.64 + 3 ~ 17-19 GB** (fits a 24 GB card)

**(c) QLoRA** (base in 4-bit NF4 + double quantization)
- Base: 7e9 x 0.5 B = 3.5 GB, plus ~0.1 GB quantization constants = ~3.6 GB
- LoRA state: 0.64 GB (same adapters as above)
- Activations (checkpointing) ~2-4 GB, CUDA context / dequant buffers ~1 GB
- **Total ~ 3.6 + 0.64 + 3 + 1 ~ 8-9 GB** (range 6-10 GB; fits a 12-16 GB card)

Say clearly in the report that these are estimates and that activations depend on batch size and sequence length.

## Task 1: LoRA settings and the reasoning to adapt (defaults in train_lora.py)
- rank r=16, alpha=32 (alpha/r = 2 scaling): narrow task, small model; r=8 is probably enough, r=16 gives headroom. If you have time, compare r=8 vs r=16 and report it.
- target modules: all linear layers (q,k,v,o,gate,up,down). Attention-only LoRA often underperforms on format-learning tasks.
- lr 2e-4 (typical LoRA range 1e-4 to 3e-4, higher than full fine-tuning), cosine schedule, 3 epochs, effective batch 16 on 1,200 examples ~ 225 steps.
- dropout 0.05 as light regularization. Prompt tokens are masked so the loss is only on the JSON.
- Be honest in the report: what you actually tried, not just what you planned.

## Task 1: where to look for your 3 failures
Open `results_lora_*.json`, filter `correct == false`. Compare failures on `heldout_template == true`
vs seen templates (by design the held-out templates, unseen names and unseen tasks are where it should break).
Check for: float vs int amounts ("697.0"), wrong category for an item, copying errors on unseen names, dropped words in the task text.

## Task 2 primers
- **BM25**: a lexical scoring function (term frequency with saturation, inverse document frequency, length normalization). Embeddings capture meaning but can blur exact tokens such as IDs, names, numbers and rare terms; BM25 matches those exactly. Hybrid search merges both rankings (for example reciprocal rank fusion).
- **Re-ranker**: a cross-encoder reads the query and one candidate passage together and outputs a relevance score. It is more accurate than comparing two independent embeddings but too slow for the whole corpus, so it re-orders the top 20-50 candidates from the first stage and you keep the best 5.
- **Chunk size**: small chunks are precise but can lose context; large chunks keep context but dilute the embedding with unrelated text. Decide from YOUR numbers and explain with your failure cases.
- Typical fixes for failures: hybrid BM25, re-ranker, query rewriting, bigger overlap, adding section titles to chunks.

## Task 3 primers
- **KV cache**: when generating, every new token attends to the keys and values of all earlier tokens. Storing K and V per layer avoids recomputing them each step, at the cost of memory that grows linearly with context length and with the number of concurrent users.
  Size per token = 2 (K and V) x layers x kv_heads x head_dim x bytes.
- **Why quantization saves memory**: weights are stored with fewer bits (16 -> 4 is ~4x smaller) plus small per-block scale factors, and are dequantized on the fly during compute. Decoding is memory-bandwidth bound, so it is often faster too. Quality can drop slightly, which is what your 10-prompt comparison should show.

## Task 3: back-of-envelope plan, 20 users on one 24 GB GPU
Assumptions (state yours): Llama-3-8B-class model (32 layers, 8 KV heads via GQA, head_dim 128), 4-bit AWQ/GPTQ weights, fp16 KV cache, ~2,048 tokens of context per user, served with vLLM (continuous batching + PagedAttention), GPU bandwidth ~900 GB/s.
- KV per token = 2 x 32 x 8 x 128 x 2 B = 131,072 B = 128 KiB
- KV for 20 users = 20 x 2,048 x 128 KiB ~ 5.4 GB
- Weights: 8e9 x 0.5 B ~ 4 GB, ~5.5 GB with scales and embeddings kept in higher precision
- Overhead (CUDA context, activations, fragmentation) ~ 2 GB
- **Total ~ 5.5 + 5.4 + 2 ~ 13 GB: fits with headroom.** (fp16 weights would be 16 + 5.4 + 2 = 23+ GB: too tight.)
- Speed: each decode step reads weights once for the whole batch plus all KV: (5.5 + 5.4 GB) / 900 GB/s ~ 12 ms -> ~80 steps/s upper bound, so roughly 25-40 tok/s per user in practice.
- Operational notes: cap max sequence length, queue extra requests, watch time-to-first-token under prefill bursts.

## Report structure (final, 4 pages)
1. **Task 1** (pages 1–2): task and data; results table (zero-shot / 3-shot / LoRA / QLoRA); LoRA settings and why; 3 failures and why; 7B memory calculation.
2. **Task 2** (pages 2–3): corpus and pipeline; recall@k for 100- vs 300-word chunks; why 300 won; 2 failures and fixes; end-to-end example; BM25 and the re-ranker. The full list of 15 questions is in `task2_rag/questions.json`.
3. **Task 3** (pages 3–4): tok/s and memory table; why the numbers make sense; the 10-prompt quality comparison; KV cache; quantisation; the 24 GB serving plan; limitations.

---

## 4. Task 1: what my runs showed
Run on Colab, Tesla T4 (15 GB), with Qwen2.5-0.5B-Instruct. Files are `task1_lora/results_*.json` and `log_*.txt`.

| Model | Valid JSON | Exact | Field acc. | Seen (37) | New phrasings (63) | Train time | Peak mem |
|---|---|---|---|---|---|---|---|
| zero-shot | 6% | 0% | 0% | 0% | 0% | – | – |
| 3-shot | 100% | 16% | 37% | 24% | 11% | – | – |
| LoRA r16 | 100% | 77% | 83% | 100% | 63% | 107 s | 4.87 GB |
| QLoRA r16 | 100% | 100% | 100% | 100% | 100% | 216 s | 2.01 GB |

**Exact match by phrasing template** (correct / total):

| Template | 3-shot | LoRA | QLoRA |
|---|---|---|---|
| seen in training | 9/37 | 37/37 | 37/37 |
| "I need a nudge to {task} {when}, please" | 2/11 | **0/11** | 11/11 |
| "Don't let me forget to {task} {when}" | 3/7 | **3/7** | 7/7 |
| "{item} cost me {amt} bucks" | 2/16 | **9/16** | 16/16 |
| "Drop {name} a note saying {msg}" | 0/12 | 11/12 | 12/12 |
| "Paid ₹{amt} for {item} just now" | 0/8 | 8/8 | 8/8 |
| "WhatsApp {name} - {msg}" | 0/9 | 9/9 | 9/9 |

**What I take from it:**

- **Zero-shot:** the model writes prose instead of JSON, and it cannot know the key names or the category list. This also made it slow (5 min for 100 requests), because it kept generating to the 100-token limit.
- **3-shot:** one example per intent fixes the format (100% valid JSON), but values are often wrong, for example the wrong category or the item copied wrongly.
- **LoRA's failures follow two shortcuts:**
    - The word "remind" signals a reminder. Without it, the model falls back to `send_message` (15 of 18 failures).
    - The item is "whatever comes last". When the item comes first, the model copies the whole sentence (7 of 16 failures).
- **QLoRA's 100% is one run.** Both runs reached a training loss of about 1e-4 by epoch 1, so they memorised the training templates. Which shortcut each run settles on is largely chance (seed, quantisation noise, the 8-bit optimiser). I would not claim QLoRA is better without 3 or more seeds each.
- **Epochs:** the loss was already 4.6e-4 at epoch 0.8, so 1 epoch would likely have given the same result in a third of the time.

**Parameter count check for Qwen2.5-0.5B:** 24 layers, hidden size 896, MLP size 4,864, and k/v projections of width 128 (2 KV heads × 64).

| Module | in + out | × r = 16 |
|---|---|---|
| q_proj, o_proj | 896 + 896 | 28,672 each |
| k_proj, v_proj | 896 + 128 | 16,384 each |
| gate, up, down | 896 + 4,864 | 92,160 each |

That is 366,592 per layer × 24 layers = **8,798,208**, exactly what peft printed.

**Measured memory:**

- **LoRA (4.87 GB peak):** the base model was loaded in fp32, which is 0.49 B × 4 B ≈ 2 GB. The rest is activations at batch 8, fp16 autocast copies and the adapters' optimiser state (8.8 M × 16 B ≈ 0.14 GB).
- **QLoRA (2.01 GB peak):** the linear weights are 4-bit, and gradient checkpointing keeps activations small. The cost is 2× the training time, because the weights are dequantised on every forward and backward pass and activations are recomputed.

## 5. Task 2: what my runs showed
The corpus is the Llama 2 paper (`doc.pdf`). On Windows, pypdf extracted 30,912 words; on Linux, a different pypdf version extracted 39,051. Some words get merged ("Weuseaweightdecayof"), which hurts both search and scoring.

| # | Question (short) | Page | Top-5 hit, 100 / 300 words |
|---|---|---|---|
| 1 | Tokens used for pretraining, and why | 5 | ✗ / ✗ |
| 2 | Changes vs Llama 1; sizes released | 4 | ✗ / ✓ |
| 3 | Optimizer and LR schedule | 5 | ✓ / ✓ |
| 4 | GPU hours and CO2 | 7 | ✓ / ✓ |
| 5 | Why SFT collection stopped; how many | 9 | ✗ / ✗ |
| 6 | Which tokens get loss in SFT; epochs | 9 | ✓ / ✓ |
| 7 | How annotators rated preference strength | 10 | ✓ / ✓ |
| 8 | Why two reward models | 10 | ✗ / ✓ |
| 9 | Reward-model loss | 11 | ✓ / ✓ |
| 10 | Rejection sampling vs PPO | 14 | ✓ / ✓ |
| 11 | PPO batch / clip settings | 15 | ✗ / ✓ |
| 12 | What Ghost Attention fixes | 16 | ✓ / ✓ |
| 13 | Safety context distillation | 23 | ✓ / ✓ |
| 14 | Who did red teaming | 28 | ✗ / ✓ |
| 15 | Training-data cutoff | 77 | ✓ / ✓ |

**Notes:**

- **Q11 and Q14 at 100 words** are metric misses, not retrieval misses: the right passage was rank 1 but split across two chunks, so neither chunk alone reached the 70% threshold.
- **Q1 retrieved** the acknowledgements page (a list of names), Table 1 and the model card. Name-list pages are "hubs" that look vaguely similar to many queries.
- **Q5 shares little wording** with its passage: the question says "supervised fine-tuning", the passage says "SFT".
- **End-to-end `ask`:** for "Why two reward models?" (200-word chunks), the answer used an unrelated GPT-4 evaluation passage and cited no pages. A small 4-bit model answers confidently from whatever it is given.

**Fixes I would try, in order:**

1. Prefix each chunk with its section heading.
2. Drop the references and acknowledgements pages from the index.
3. Use hybrid BM25 + embeddings with reciprocal rank fusion.
4. Add a cross-encoder re-ranker (for example bge-reranker) over the top 30.
5. Score hits at page level as well as by snippet overlap.

## 6. Task 3: what my runs showed
Laptop CPU only (`ollama ps` reports 100% CPU), with a 4,096-token context, temperature 0 and at most 200 new tokens.

| Setting | Mean tok/s | Min–max | Memory |
|---|---|---|---|
| fp16 | 9.1 | 6.1–10.0 | 3.3 GB |
| q4_K_M | 22.1 | 11.9–26.1 | 1.2 GB |

**Sanity checks:**

- **fp16 size:** 1.54 B × 2 B = 3.1 GB of weights, matching the 3.3 GB measured.
- **q4_K_M size:** about 4.85 bits per weight ≈ 0.93 GB. The KV cache (28 KiB per token × 4,096 = 112 MiB) and buffers make up the rest of the 1.2 GB.
- **Bandwidth:** decoding reads all the weights once per token. 3.1 GB × 9.1 tok/s ≈ 28 GB/s and 0.93 GB × 22.1 ≈ 21 GB/s, both typical laptop RAM bandwidth. So generation here is memory-bandwidth bound, which is why 4-bit is faster.
- **Slowest prompt:** prompt 5 (15 tokens) was the slowest for both, because the fixed cost of each request is spread over few tokens.

| # | Prompt | fp16 | q4_K_M | Verdict |
|---|---|---|---|---|
| 1 | Sky is blue, 3 sentences | correct, 4 sentences | correct, 2 sentences | both miss the count |
| 2 | Iterative Fibonacci + docstring | correct, cut at 200 tokens | correct, cut at 200 tokens | tie |
| 3 | Romeo and Juliet, exactly 2 sentences | 3 sentences | 1 sentence | both miss |
| 4 | 120 km in 1.5 h → m/s | correct steps, cut before 22.2 m/s | same | tie (cap) |
| 5 | Translate to French | correct | identical | tie |
| 6 | 5 differences process vs thread | repetitive, no error | "a thread can be part of multiple processes" (wrong) | fp16 better |
| 7 | Sentence → JSON | correct | identical | tie |
| 8 | Polite decline email | good | good | tie |
| 9 | Capital of Australia, why not Sydney | Canberra; "established 1908" (site chosen 1908, founded 1913); no real why | Canberra; circular why | both weak |
| 10 | Overfitting analogy | muddled | clearer (memorising answers) | q4 slightly better |

## 7. Problems hit during the runs, and fixes
- **Windows encoding:** `rag.py` opens files with Windows' default cp1252 encoding, which can't read UTF-8 characters such as β and curly quotes. Fixed by saving `questions.json` with `\u` escapes. `bench_ollama.py` now writes its outputs as UTF-8.
- **Colab, peft vs torchao:** peft raised "incompatible torchao 0.10.0" when creating LoRA layers on the fp32 model. Uninstalling torchao fixed it (`python -m pip uninstall -y torchao`). The QLoRA run didn't hit this error; its training failed for the separate `warmup_ratio` reason below.
- **Colab, transformers 5:** `TrainingArguments` no longer accepts `warmup_ratio`. Replaced with `warmup_steps=7`.
- **Ollama memory reading:** Ollama keeps the previous model loaded for 5 minutes, so the second model's reading would include both. `bench_ollama.py` now runs `ollama stop` after each model.
- **Snippet length vs chunk size:** with 100-word chunks and 20-word overlap, a snippet longer than about 45 words can never reach 70% overlap with a single chunk. I kept snippets to 22–44 words and checked each one against `rag.py`'s own chunking before running.

## 8. Likely interview questions and short answers
- **Why mask the prompt tokens?** So the loss and gradients come only from the JSON the model must produce. Otherwise it also learns to reproduce the request text, which wastes capacity and skews the loss.
- **What does alpha do?** The LoRA update is scaled by alpha / r. With alpha = 2r the scale is 2, and keeping that ratio fixed means changing r doesn't change the effective learning rate much.
- **Why is the LoRA learning rate higher than for full fine-tuning?** Only a small number of parameters change, and B starts at zero, so the update starts at exactly zero and needs a larger step to move.
- **Why is QLoRA slower?** The 4-bit weights are dequantised to fp16 on every forward and backward pass, and gradient checkpointing recomputes activations.
- **Is QLoRA really better here (100% vs 77%)?** Not proven. It is one run each and both memorised the training set. I would run 3+ seeds and report the mean and spread.
- **Why exact match rather than field accuracy?** A downstream system needs the whole object right. Field accuracy (83% for LoRA) is reported too, and shows partial credit.
- **Why did 300-word chunks win?** Questions often need two facts that sit together. Bigger chunks keep them together and mean fewer distractors. Part of the 100-word deficit was the metric splitting passages.
- **Why FAISS `IndexFlatIP`?** It is exact search, and with only 129–387 vectors there is no need for an approximate index. Inner product on normalised vectors equals cosine similarity.
- **What does BM25 add?** Exact matching of rare tokens such as numbers, IDs and names, which embeddings blur. Hybrid ranking with reciprocal rank fusion combines the two.
- **What does a re-ranker do?** A cross-encoder scores the query and a passage jointly. It is more accurate but slower, so it is used only on the top 20–50 candidates.
- **Why is 4-bit faster on CPU?** Decoding is memory-bandwidth bound, and 4-bit reads about 3.3× fewer bytes of weights per token.
- **What is in the KV cache, and how big is it?** The keys and values for every past token, per layer. Its size is 2 × layers × kv_heads × head_dim × bytes per token, which is 28 KiB per token for Qwen2.5-1.5B and 128 KiB per token for an 8B model with GQA.
- **What limits 20 users on 24 GB?** Mostly KV-cache memory as context grows. With 4-bit weights and 2,048 tokens each, the total is about 13 GB; fp16 weights would need about 23 GB. Throughput is bounded by bandwidth: about 12 ms per decode step.
