"""LoRA / QLoRA fine-tuning. Prompt tokens are masked so loss is only on the JSON answer.
Examples:
  python train_lora.py --model Qwen/Qwen2.5-0.5B-Instruct --out lora_r16
  python train_lora.py --model Qwen/Qwen2.5-1.5B-Instruct --qlora --out qlora_r16"""
import json, argparse, torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
                          DataCollatorForSeq2Seq, BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

PROMPT = "### Request:\n{text}\n### JSON:\n"
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--out", default="lora_out")
ap.add_argument("--r", type=int, default=16)
ap.add_argument("--alpha", type=int, default=32)
ap.add_argument("--dropout", type=float, default=0.05)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--epochs", type=float, default=3)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--grad_acc", type=int, default=2)
ap.add_argument("--qlora", action="store_true")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None: tok.pad_token = tok.eos_token

def encode(row):
    p = tok(PROMPT.format(text=row["text"]), add_special_tokens=False)["input_ids"]
    t = tok(row["target"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    return {"input_ids": p + t, "attention_mask": [1] * (len(p) + len(t)), "labels": [-100] * len(p) + t}

rows = [json.loads(l) for l in open("train.jsonl", encoding="utf-8")]
ds = Dataset.from_list(rows).map(encode, remove_columns=["text", "target", "heldout_template"])

if a.qlora:
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model)
else:
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).cuda()

cfg = LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=a.dropout, task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, cfg)
model.print_trainable_parameters()

args = TrainingArguments(output_dir=a.out, per_device_train_batch_size=a.bs,
    gradient_accumulation_steps=a.grad_acc, learning_rate=a.lr, num_train_epochs=a.epochs,
    lr_scheduler_type="cosine", warmup_steps=7,  # 7 = 3% of 225 steps (warmup_ratio was removed in transformers 5)
    logging_steps=20, save_strategy="no",
    fp16=torch.cuda.is_available(), report_to="none",
    optim="paged_adamw_8bit" if a.qlora else "adamw_torch")
Trainer(model=model, args=args, train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)).train()
model.save_pretrained(a.out); tok.save_pretrained(a.out)
print("peak GPU memory (GB):", torch.cuda.max_memory_allocated() / 1e9)
