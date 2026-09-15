from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import os

MODEL_ID   = "EleutherAI/pythia-1b"
CACHE_DIR  = os.path.join(os.path.dirname(__file__), "models", "pythia-1b")

CHECKPOINTS = {
    "step1000":   "step1000",
    "step10000":  "step10000",
    "step50000":  "step50000",
    "step143000": "step143000",
}

QUESTIONS = [
    {"prompt": "World War II ended in",
     "expected": ["1945"], "era": "pre-2000"},
    {"prompt": "The Berlin Wall fell in",
     "expected": ["1989"], "era": "pre-2000"},
    {"prompt": "The United States declared independence in",
     "expected": ["1776"], "era": "pre-2000"},

    {"prompt": "Donald Trump was elected US president in",
     "expected": ["2016"], "era": "2016-2020"},
    {"prompt": "The United Kingdom voted to leave the EU in",
     "expected": ["2016", "Brexit"], "era": "2016-2020"},

    {"prompt": "COVID-19 was declared a pandemic by the WHO in",
     "expected": ["2020", "March"], "era": "2020-2021"},
    {"prompt": "Joe Biden was inaugurated as US president in",
     "expected": ["2021", "January"], "era": "2020-2021"},

    {"prompt": "ChatGPT was released by OpenAI in",
     "expected": ["2022"], "era": "post-cutoff"},
    {"prompt": "The 2022 FIFA World Cup was held in",
     "expected": ["Qatar"], "era": "post-cutoff"},
    {"prompt": "Elon Musk acquired Twitter in",
     "expected": ["2022"], "era": "post-cutoff"},
]


def load_checkpoint(revision: str):
    print(f"  Downloading/loading {MODEL_ID} @ {revision} → {CACHE_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=revision,
        cache_dir=CACHE_DIR,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=revision,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model.eval().to(device)
    return tokenizer, model, device


def generate(tokenizer, model, device, prompt: str, max_new_tokens: int = 30) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def check(response: str, expected: list[str]) -> bool:
    r = response.lower()
    return any(e.lower() in r for e in expected)


# ── 主循环 ──────────────────────────────────────────────────────────────
results = {}

for label, revision in CHECKPOINTS.items():
    print(f"Loading {label}...")
    tokenizer, model, device = load_checkpoint(revision)
    results[label] = []

    for q in QUESTIONS:
        resp = generate(tokenizer, model, device, q["prompt"])
        ok   = check(resp, q["expected"])
        results[label].append({**q, "response": resp, "correct": ok})

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print(f"  done.")

# ── 按问题分组展示 ───────────────────────────────────────────────────────
ERAS = ["pre-2000", "2016-2020", "2020-2021", "post-cutoff"]
CKPT_LABELS = list(CHECKPOINTS.keys())

for era in ERAS:
    era_questions = [q for q in QUESTIONS if q["era"] == era]
    print(f"\n{'━'*70}")
    print(f"  {era}")
    print(f"{'━'*70}")

    for q in era_questions:
        print(f"\n  Q: {q['prompt']}")
        for label in CKPT_LABELS:
            row = next(r for r in results[label] if r["prompt"] == q["prompt"])
            status = "✓" if row["correct"] else "✗"
            resp_clean = row['response'].replace('\n', ' ')[:80]
            print(f"    [{label:<12}] {status}  {resp_clean}")

# ── 得分汇总表 ───────────────────────────────────────────────────────────
print(f"\n\n{'─'*70}")
print(f"  SCORE SUMMARY")
print(f"{'─'*70}")
print(f"  {'':15}", end="")
for era in ERAS:
    print(f"  {era:<14}", end="")
print()
print(f"  {'─'*65}")
for label in CKPT_LABELS:
    print(f"  {label:<15}", end="")
    for era in ERAS:
        subset = [r for r in results[label] if r["era"] == era]
        n = len(subset)
        c = sum(1 for r in subset if r["correct"])
        bar = "✓" * c + "✗" * (n - c)
        print(f"  {c}/{n} {bar:<12}", end="")
    print()

out_path = os.path.join(os.path.dirname(__file__), "pythia_cutoff_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n结果已保存 → {out_path}")
