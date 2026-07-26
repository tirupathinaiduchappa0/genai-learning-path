# Lesson 15 — Fine-Tuning LLMs (LLaMA / Falcon)

## 1. The Question

> "How would you fine-tune a large language model like LLaMA or Falcon? Cover data prep, full vs parameter-efficient methods (LoRA/QLoRA), compute, and evaluation."

---

## 2. Theory — first, should you even fine-tune?

Decision order (cheapest → most expensive):

1. **Prompt engineering** — often enough; zero training.
2. **RAG** — inject knowledge at query time (best for *facts / changing knowledge*).
3. **Fine-tuning** — best for *behavior, style, format, tone, domain skills* the model won't reliably follow from prompts.

Rule of thumb: **RAG for knowledge, fine-tuning for behavior.** They're complementary — you can fine-tune a model that also uses RAG.

Fine-tuning is worth it when: you need a consistent output format/style, a specialized domain skill, lower latency/cost via a smaller specialized model, or you have thousands of good examples.

---

## 3. Types of fine-tuning

| Method | What it updates | Cost | When |
|---|---|---|---|
| **Full fine-tune** | All weights | Very high (multi-GPU, 100s of GB) | Lots of data + compute, big domain shift |
| **LoRA** | Small low-rank adapter matrices; base frozen | Low | Most cases |
| **QLoRA** | LoRA on a **4-bit quantized** base | Very low (single GPU) | Default for open models on limited hardware |
| **Prefix/Prompt tuning** | Learned soft prompts | Lowest | Lightweight task adaptation |

### Why LoRA/QLoRA (PEFT) dominates
Full fine-tuning a 7B model needs ~14 GB just for fp16 weights, plus gradients + optimizer states (~4×) → 60–80 GB+. **LoRA** freezes the base and trains tiny rank-`r` matrices injected into attention layers — <1% of parameters. **QLoRA** additionally loads the base in **4-bit** (NF4), slashing memory so a 7B–13B model fine-tunes on a single consumer/A100 GPU, with ~full-fine-tune quality.

---

## 4. The workflow

```
1. Define the objective        (what behavior/skill?)
2. Build the dataset           (instruction/response pairs, high quality)
3. Format to a chat template   (system/user/assistant)
4. Pick base model + method    (Llama-3-8B + QLoRA)
5. Train (HF Trainer + peft)   (few epochs, watch val loss)
6. Evaluate                    (held-out set + task metrics + human)
7. Merge / serve adapters      (vLLM, TGI)
8. Iterate
```

### Data is everything
- **Quality > quantity.** A few thousand clean, consistent examples beat a noisy huge set.
- Format as instruction pairs; keep a **held-out eval set**.
- Watch for leakage, duplicates (Lesson 01!), and label noise.
- Balance classes / task types; include hard/edge cases.

---

## 5. Hands-on — QLoRA with HuggingFace + PEFT

```python
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset

model_id = "meta-llama/Meta-Llama-3-8B"

# 1. 4-bit quantization config (the "Q" in QLoRA)
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# 2. LoRA adapter config (the "LoRA" part)
lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()   # e.g. "0.5% of params trainable"

# 3. Data (expects a 'text' field already in chat format)
dataset = load_dataset("json", data_files="train.jsonl", split="train")

# 4. Train
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=TrainingArguments(
        output_dir="./out",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        bf16=True,
        logging_steps=10,
    ),
)
trainer.train()
model.save_pretrained("./llama3-qlora-adapter")   # tiny adapter, not full model
```

Serving: load base + adapter, or `merge_and_unload()` to bake the adapter into the weights, then serve via vLLM/TGI.

---

## 6. Evaluation

- **Held-out loss / perplexity** — sanity, but not sufficient.
- **Task metrics** — accuracy/F1 for classification, ROUGE/BERTScore or LLM-judge for generation.
- **Regression check** — did general capability degrade (catastrophic forgetting)?
- **Human eval** — for tone/quality on real prompts.
- **A/B** vs the base model + prompt-only baseline — prove the fine-tune actually helped.

---

## 7. Real-time / production notes

- **Alignment stage:** SFT first; for preference/tone, add **DPO** (simpler than RLHF) or RLHF.
- **Catastrophic forgetting:** low LR, few epochs, mix in some general data.
- **Adapters are portable:** ship multiple task adapters over one base model → cheap multi-task serving.
- **Cost:** QLoRA a 7–8B model fits on a single 24–48 GB GPU; full fine-tune needs a cluster.
- **Governance:** version datasets + adapters + eval results; document data provenance and licenses (Llama/Falcon license terms).

---

## 8. Interview script

"First I ask whether fine-tuning is even the right tool — RAG for knowledge, fine-tuning for behavior, style, or format. If it's behavior, I default to QLoRA: load the base in 4-bit and train small low-rank LoRA adapters on the attention projections, which gets near full-fine-tune quality on a single GPU. Data quality is the whole game — a few thousand clean, consistent instruction pairs with a held-out eval set. I train a few epochs with a low learning rate to avoid catastrophic forgetting, evaluate with task metrics plus human review against a prompt-only baseline, then serve the adapter via vLLM. For preference alignment I'd add DPO. I version datasets, adapters, and eval results for governance."
