from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import os

BASE_DIR = os.path.dirname(__file__)

MODELS = {
    "gpt2-xl-2019":    {"id": "gpt2-xl",                    "cache": os.path.join(BASE_DIR, "models", "gpt2-xl"),    "cutoff": "early 2019"},
    "pythia-1b-2020":  {"id": "EleutherAI/pythia-1b",       "cache": os.path.join(BASE_DIR, "models", "pythia-1b"),  "cutoff": "late 2020"},
    "opt-1.3b-2022":   {"id": "facebook/opt-1.3b",          "cache": os.path.join(BASE_DIR, "models", "opt-1.3b"),   "cutoff": "early 2022"},
}

# Test questions stratified by time period
QUESTIONS = [
    # Should all be answered correctly (high-frequency pre-cutoff facts)
    {"prompt": "World War II ended in",
     "expected": ["1945"], "era": "pre-2000"},
    {"prompt": "The Berlin Wall fell in",
     "expected": ["1989"], "era": "pre-2000"},
    {"prompt": "The United States declared independence in",
     "expected": ["1776"], "era": "pre-2000"},

    # gpt2 might not know, pythia/opt should know
    {"prompt": "Donald Trump was elected US president in",
     "expected": ["2016"], "era": "2016-2020"},
    {"prompt": "The United Kingdom voted to leave the EU in",
     "expected": ["2016"], "era": "2016-2020"},
    {"prompt": "COVID-19 was declared a pandemic by the WHO in",
     "expected": ["2020", "March"], "era": "2020"},

    # 2019: gpt2 boundary, pythia/opt should know
    {"prompt": "Notre-Dame Cathedral in Paris caught fire in",
     "expected": ["2019"], "era": "2019"},
    {"prompt": "Boeing 737 Max was grounded worldwide in",
     "expected": ["2019"], "era": "2019"},

    # 2020: gpt2 doesn't know, pythia/opt should know
    {"prompt": "COVID-19 was declared a pandemic by the WHO in",
     "expected": ["2020"], "era": "2020"},
    {"prompt": "George Floyd was killed by police in Minneapolis in",
     "expected": ["2020"], "era": "2020"},
    {"prompt": "Kobe Bryant died in a helicopter crash in",
     "expected": ["2020"], "era": "2020"},

    # 2021: pythia boundary, opt should know
    {"prompt": "Joe Biden was inaugurated as US president in",
     "expected": ["2021"], "era": "2021"},
    {"prompt": "The Taliban took control of Kabul in",
     "expected": ["2021"], "era": "2021"},
    {"prompt": "The COVID-19 vaccine was first approved in",
     "expected": ["2021"], "era": "2021"},

    # post-cutoff 2022: none of them know, see who fabricates most convincingly
    {"prompt": "ChatGPT was released by OpenAI in",
     "expected": ["2022"], "era": "post-cutoff"},
    {"prompt": "The 2022 FIFA World Cup was held in",
     "expected": ["Qatar"], "era": "post-cutoff"},
    {"prompt": "Elon Musk acquired Twitter in",
     "expected": ["2022"], "era": "post-cutoff"},

    # post-cutoff 2023-2025: none of them know, major breaking news, see who fabricates the most outlandishly
    {"prompt": "Silicon Valley Bank collapsed in",
     "expected": ["2023", "March"], "era": "post-cutoff"},
    {"prompt": "Hamas launched a surprise attack on Israel in",
     "expected": ["2023", "October"], "era": "post-cutoff"},
    {"prompt": "Sam Altman was briefly fired and reinstated as OpenAI CEO in",
     "expected": ["2023", "November"], "era": "post-cutoff"},
    {"prompt": "A Japan Airlines plane collided with a coast guard aircraft at Haneda Airport in",
     "expected": ["2024", "January"], "era": "post-cutoff"},
    {"prompt": "Donald Trump survived an assassination attempt at a rally in",
     "expected": ["2024", "July"], "era": "post-cutoff"},
    {"prompt": "Joe Biden withdrew from the 2024 presidential race in",
     "expected": ["2024", "July"], "era": "post-cutoff"},
    {"prompt": "Donald Trump won the 2024 US presidential election in",
     "expected": ["2024", "November"], "era": "post-cutoff"},
    {"prompt": "Bashar al-Assad's government in Syria fell in",
     "expected": ["2024", "December"], "era": "post-cutoff"},
    {"prompt": "The Chinese AI startup DeepSeek released a model that shook global markets in",
     "expected": ["2025", "January"], "era": "post-cutoff"},
    {"prompt": "A magnitude 7.7 earthquake struck Myanmar in",
     "expected": ["2025", "March"], "era": "post-cutoff"},

    # More widely known big news items, added as a supplement
    {"prompt": "The Titan submersible imploded during a dive to the Titanic wreck in",
     "expected": ["2023", "June"], "era": "post-cutoff"},
    {"prompt": "Barbie and Oppenheimer both premiered in theaters in",
     "expected": ["2023", "July"], "era": "post-cutoff"},
    {"prompt": "A massive wildfire devastated Lahaina, Maui, Hawaii in",
     "expected": ["2023", "August"], "era": "post-cutoff"},
    {"prompt": "A total solar eclipse crossed North America in",
     "expected": ["2024", "April"], "era": "post-cutoff"},
    {"prompt": "The Summer Olympics were held in Paris in",
     "expected": ["2024"], "era": "post-cutoff"},
    {"prompt": "A CrowdStrike software update caused a massive global IT outage in",
     "expected": ["2024", "July"], "era": "post-cutoff"},
    {"prompt": "Pope Francis died in",
     "expected": ["2025", "April"], "era": "post-cutoff"},
    {"prompt": "TikTok briefly went dark in the United States due to a ban in",
     "expected": ["2025", "January"], "era": "post-cutoff"},
]


def load_model(model_id, cache_dir):
    print(f"  Loading {model_id} ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model.eval().to(device)
    return tokenizer, model, device


def generate(tokenizer, model, device, prompt, max_new_tokens=40):
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


def check(response, expected):
    r = response.lower()
    return any(e.lower() in r for e in expected)


# ── main loop ─────────────────────────────────────────────────────────
results = {}

for label, cfg in MODELS.items():
    print(f"\n{'='*60}")
    print(f"  Model : {label}  (cutoff: {cfg['cutoff']})")
    print(f"{'='*60}")
    tokenizer, model, device = load_model(cfg["id"], cfg["cache"])
    results[label] = []

    for q in QUESTIONS:
        resp = generate(tokenizer, model, device, q["prompt"])
        ok   = check(resp, q["expected"])
        results[label].append({**q, "response": resp, "correct": ok})

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("  done.")

# ── output grouped by question ───────────────────────────────────────────
ERAS = ["pre-2000", "2016-2020", "2019", "2020", "2021", "post-cutoff"]
MODEL_LABELS = list(MODELS.keys())

for era in ERAS:
    era_qs = [q for q in QUESTIONS if q["era"] == era]
    if not era_qs:
        continue
    print(f"\n{'━'*70}")
    print(f"  [{era}]")
    print(f"{'━'*70}")
    for q in era_qs:
        answer = ", ".join(q["expected"])
        print(f"\n  Q: {q['prompt']} (ans: {answer})")
        for label in MODEL_LABELS:
            row = next(r for r in results[label] if r["prompt"] == q["prompt"])
            resp_clean = row["response"].replace("\n", " ")[:80]
            print(f"    [{label:<18}]  {resp_clean}")

# ── score summary ────────────────────────────────────────────────────────
print(f"\n\n{'─'*70}")
print(f"  SCORE SUMMARY")
print(f"{'─'*70}")
print(f"  {'Model':<22} {'cutoff':<14}", end="")
for era in ERAS:
    print(f"  {era:<12}", end="")
print()
print(f"  {'─'*68}")

for label in MODEL_LABELS:
    cutoff = MODELS[label]["cutoff"]
    print(f"  {label:<22} {cutoff:<14}", end="")
    for era in ERAS:
        subset = [r for r in results[label] if r["era"] == era]
        if not subset:
            print(f"  {'—':<12}", end="")
            continue
        n = len(subset)
        print(f"  {n} Q{'s' if n>1 else '':<9}", end="")
    print()

# ── save results ─────────────────────────────────────────────────────────
out_path = os.path.join(BASE_DIR, "level3_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nResults saved → {out_path}")
