from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from itertools import product
import torch
import json

# ── config ───────────────────────────────────────────────────────────────────
MODEL_NAME   = "mistralai/Mistral-7B-Instruct-v0.2"
LOCAL_PATH   = "./models/mistral-7b"
TAG_CONFIG   = "level4_tags_config.json"
NUM_RUNS     = 5          # candidates generated per combo (27 combos x 5 = 135, more reasonable than the previous 3 outlets)
MAX_TOKENS   = 40         # 30 was too short before and often cut headlines off; bumped up
TEMPERATURE  = 0.9
TOP_P        = 0.95
MAX_RETRIES  = 3          # retries if a generated result looks truncated
OUTPUT_RAW   = "level4_config.json"

# ── load tag config ────────────────────────────────────────────────────────
with open(TAG_CONFIG, "r", encoding="utf-8") as f:
    tag_cfg = json.load(f)

DIMS      = tag_cfg["dimensions"]
TEMPLATE  = tag_cfg["prompt_template"]
FACTS     = tag_cfg["example_scenario"]["facts"]
EVENT_ID  = tag_cfg["example_scenario"]["id"]

# ── load model ────────────────────────────────────────────────────────────
quant_config = BitsAndBytesConfig(load_in_4bit=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=LOCAL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto",
    cache_dir=LOCAL_PATH
)
model.eval()
print("Model loaded.\n")


# ── generation functions ───────────────────────────────────────────────────
def build_prompt(facts: str, audience_frag: str, tone_frag: str, engagement_frag: str) -> str:
    body = TEMPLATE.format(
        facts=facts,
        audience_fragment=audience_frag,
        tone_fragment=tone_frag,
        engagement_fragment=engagement_frag,
    )
    return f"[INST] {body} [/INST]"


def looks_truncated(text: str) -> bool:
    """Roughly detect whether a headline got cut off: no trailing punctuation, ends on a preposition/conjunction/half a word, etc."""
    if not text:
        return True
    text = text.strip()
    if text[-1] not in ".!?\u2014\u201d\"'":
        # Allow the case with no terminal punctuation, but if the last word looks short/is a function word, it was likely cut off
        last_word = text.split()[-1].lower().strip(".,!?\"'")
        weak_endings = {
            "a", "an", "the", "of", "to", "in", "on", "at", "as", "for",
            "with", "and", "or", "amid", "over", "under", "-",
        }
        if last_word in weak_endings or len(last_word) <= 2:
            return True
    return False


def generate_one(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(
        output[0][input_len:],
        skip_special_tokens=True,
    ).strip()
    first_line = generated.split("\n")[0].strip().strip('"')
    return first_line


def generate_headlines(prompt: str, n: int) -> list[str]:
    headlines = []
    for _ in range(n):
        headline = generate_one(prompt)
        # If it looks truncated, use a remaining retry to get another one
        retries = 0
        while looks_truncated(headline) and retries < MAX_RETRIES:
            headline = generate_one(prompt)
            retries += 1
        headlines.append(headline)
    return headlines


# ── main collection loop: iterate all 27 tag combinations ────────────────
results = {
    "id": EVENT_ID,
    "facts": FACTS,
    "combos": [],
}

combo_list = list(product(DIMS["audience"], DIMS["tone"], DIMS["engagement"]))
print(f"=== Event: {EVENT_ID} | {len(combo_list)} combos ===\n")

for audience, tone, engagement in combo_list:
    combo_id = f"{audience['id']}_{tone['id']}_{engagement['id']}"
    print(f"  [{combo_id}]")

    prompt = build_prompt(
        FACTS,
        audience["prompt_fragment"],
        tone["prompt_fragment"],
        engagement["prompt_fragment"],
    )

    headlines = generate_headlines(prompt, NUM_RUNS)
    for i, h in enumerate(headlines):
        print(f"    {i+1}. {h}")

    results["combos"].append({
        "id": combo_id,
        "audience": audience["id"],
        "tone": tone["id"],
        "engagement": engagement["id"],
        "instruction_fragments": {
            "audience": audience["prompt_fragment"],
            "tone": tone["prompt_fragment"],
            "engagement": engagement["prompt_fragment"],
        },
        "generated": headlines,
        "selected": [],  # ← fill in the manually picked ones here
    })

# ── save ─────────────────────────────────────────────────────────────────────
with open(OUTPUT_RAW, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nSaved to {OUTPUT_RAW}")
print("Review 'generated' per combo, fill in 'selected'.")