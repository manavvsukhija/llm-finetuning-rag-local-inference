# LLM engineering assignment: LoRA fine-tuning, RAG from scratch, local inference

**Author:** Manav Sukhija 

Every number below comes from my own runs. The raw outputs and logs are in the folders listed under each task.

## Results at a glance

| Task | Headline result |
| --- | --- |
| 1 – LoRA / QLoRA | Exact match on 100 test requests rose from 16% (base model, 3-shot) to **77% with LoRA** and **100% with QLoRA**. Both are single runs. |
| 2 – RAG from scratch | Recall@5 of **0.87 with 300-word chunks**, against 0.60 with 100-word chunks, on 15 hand-written questions. |
| 3 – Local inference | 4-bit q4_K_M ran **2.4× faster** (22.1 vs 9.1 tok/s) and was **2.75× smaller** (1.2 vs 3.3 GB) than fp16, with 1 factual error in 10 prompts. |

## Repository layout

```
LLM_Assignment_Report_Manav_Sukhija.pdf   4-page report
NOTES.md                                  worked calculations, concept notes, run log and interview prep
task1_lora/
  gen_data.py, train_lora.py, eval_model.py, requirements.txt
  task1_colab.ipynb                       the notebook actually used on Colab (scripts inlined)
  results_*.json                          metrics + every prediction, one file per run
  log_*.txt, gpu_info.txt                 training / eval logs and GPU name
  train.jsonl, test.jsonl                 generated data (1,200 / 100), recreated exactly by gen_data.py (seed 7)
task2_rag/
  rag.py, requirements.txt
  doc.pdf                                 corpus: Llama 2 paper, arXiv 2307.09288 (77 pages)
  questions.json                          15 questions with gold page + gold snippet
  rag_results.json                        recall@k per chunk size + retrieved top-3 for every miss
task3_local_inference/
  bench_ollama.py, prompts.txt
  bench_summary.csv                       tok/s and memory per setting
  outputs_<model>.json                    the 10 answers from each setting, for side-by-side reading
```

## Task 1 – LoRA / QLoRA (`task1_lora/`)

**How it was run:** on Google Colab with a Tesla T4 (15 GB). Upload `task1_colab.ipynb`, choose Runtime → T4 GPU, then Runtime → Run all. The whole run takes about 25 minutes. The equivalent commands are:

```
pip install -r requirements.txt
pip uninstall -y torchao                            # Colab ships torchao 0.10, which current peft rejects
python gen_data.py                                  # train.jsonl (1,200) + test.jsonl (100)
python eval_model.py --mode zeroshot                # base model, no examples
python eval_model.py --mode fewshot                 # base model, 3 examples in the prompt
python train_lora.py --out lora_r16                 # LoRA, r=16
python eval_model.py --mode lora --adapter lora_r16
python train_lora.py --qlora --out qlora_r16        # QLoRA, 4-bit NF4 base
python eval_model.py --mode lora --adapter qlora_r16
```

**What the task is:** Qwen2.5-0.5B-Instruct turns a plain-English request into fixed JSON with one of three intents: `set_reminder`, `add_expense` or `send_message`. Of the 100 test requests, 37 use phrasings seen in training. The other 63 use 6 unseen phrasings and include unseen names and tasks.

| Model | Valid JSON | Exact match | Field acc. | Exact: seen / new phrasings | Train time | Peak GPU mem |
| --- | --- | --- | --- | --- | --- | --- |
| Base, zero-shot | 6% | 0% | 0% | 0% / 0% | – | – |
| Base, 3-shot | 100% | 16% | 37% | 24% / 11% | – | – |
| LoRA r=16 | 100% | **77%** | 83% | 100% / 63% | 107 s | 4.87 GB |
| QLoRA r=16 | 100% | **100%** | 100% | 100% / 100% | 216 s | 2.01 GB |

**LoRA settings:**

- Rank r = 16 with alpha = 32.
- Applied to all 7 linear layers: q, k, v, o, gate, up, down.
- Learning rate 2e-4 on a cosine schedule, with 7 warmup steps.
- 3 epochs at effective batch size 16 (8 per step × 2 gradient accumulation), which is 225 steps.
- Dropout 0.05.
- Prompt tokens are masked, so the loss is computed on the JSON answer only.
- This trains 8,798,208 parameters, 1.75% of the model.

## Task 2 – RAG from scratch (`task2_rag/`)

**How it was run:** on a Windows laptop, CPU only. The `ask` command needs Ollama running with `qwen2.5:1.5b-instruct-q4_K_M` pulled.

```
pip install -r requirements.txt
python rag.py eval --pdf doc.pdf --questions questions.json --sizes 100 300 --k 1 3 5
python rag.py ask  --pdf doc.pdf --q "Why does Llama 2-Chat use two reward models?"
```

**The pipeline**, with no LangChain or LlamaIndex:

1. pypdf extracts the text.
2. The words are cut into fixed-size windows with 20% overlap.
3. bge-small-en-v1.5 embeds each window.
4. A FAISS `IndexFlatIP` index finds the closest windows by exact cosine similarity.
5. A local Qwen model writes the final answer from the top 5 windows.

| Chunk size | Chunks | Recall@1 | Recall@3 | Recall@5 | Questions missed in the top 5 |
| --- | --- | --- | --- | --- | --- |
| 100 words | 387 | 0.33 | 0.53 | 0.60 | 1, 2, 5, 8, 11, 14 |
| 300 words | 129 | 0.60 | 0.73 | **0.87** | 1, 5 |

A retrieved chunk counts as a hit if it contains at least 70% of the gold snippet's distinct words.

## Task 3 – Local inference (`task3_local_inference/`)

**How it was run:** on the same Windows laptop, CPU only, with Ollama. Each request used temperature 0 and at most 200 new tokens.

```
ollama pull qwen2.5:1.5b-instruct-fp16
ollama pull qwen2.5:1.5b-instruct-q4_K_M
python bench_ollama.py qwen2.5:1.5b-instruct-fp16 qwen2.5:1.5b-instruct-q4_K_M
```

| Setting | Mean tok/s | Min–max tok/s | Memory (`ollama ps`) |
| --- | --- | --- | --- |
| fp16 | 9.1 | 6.1–10.0 | 3.3 GB |
| q4_K_M (4-bit) | 22.1 | 11.9–26.1 | 1.2 GB |

## Fixes made to the starter code

- **`train_lora.py`:** `warmup_ratio=0.03` became `warmup_steps=7` (3% of 225 steps). transformers 5 removed `warmup_ratio`.
- **Colab:** uninstalled `torchao`. Colab's version 0.10 made the current peft refuse to create LoRA layers.
- **`bench_ollama.py`:** each model is unloaded (`ollama stop`) before the next one is measured, so the second model's memory reading doesn't include the first.
- **`bench_ollama.py`:** the output files are written as UTF-8, because Windows' default encoding crashes on some model output.
- **`questions.json`:** saved as ASCII with `\u` escapes, because `rag.py` opens files with Windows' default encoding.

## Known limitations

- **Task 1:** each Task 1 number comes from one training run (one seed).
- **Task 2:** the word-overlap hit test misses passages that are split across two chunks. pypdf also merges some words on this PDF.
- **Task 3:** quality was judged by reading the outputs, not with a graded rubric.

See the report and `NOTES.md` for details.
