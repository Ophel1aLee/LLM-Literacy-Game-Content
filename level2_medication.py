from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import torch.nn.functional as F
from contextlib import contextmanager

MODEL_NAME  = "gpt2-large"
LOCAL_PATH  = "./models/gpt2-large"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
CANDIDATES  = ["safe", "unsafe", "dangerous", "risky", "harmful", "recommended"]
BAR_WIDTH   = 30

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME, cache_dir=LOCAL_PATH)
model = GPT2LMHeadModel.from_pretrained(
    MODEL_NAME,
    cache_dir=LOCAL_PATH,
    attn_implementation="eager"
)
model.eval().to(DEVICE)


NUM_LAYERS = len(model.transformer.h)


def find_token_index(text: str, keyword: str) -> int:
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(text)["input_ids"])
    for i, t in enumerate(tokens):
        if keyword in t.lower().replace("ġ", ""):
            return i
    return -1


@contextmanager
def intervene_attention(layer_idx: int, target_idx: int, multiplier: float):
    attn = model.transformer.h[layer_idx].attn

    def hook(module, inp, out):
        modified = out.clone()
        modified[:, target_idx, :] *= multiplier  # amplify the Q/K/V of "not"
        return modified

    handle = attn.c_attn.register_forward_hook(hook)  # c_attn is the QKV projection layer
    try:
        yield
    finally:
        handle.remove()

def get_attention_weights(sentence: str, intervene_cfg: dict | None = None) -> list[dict]:
    inputs = tokenizer(sentence, return_tensors="pt").to(DEVICE)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        if intervene_cfg:
            with intervene_attention(**intervene_cfg):
                outputs = model(**inputs, output_attentions=True)
        else:
            outputs = model(**inputs, output_attentions=True)

    last_row = outputs.attentions[-1][0].mean(dim=0)[-1].float().cpu().tolist()

    SPECIAL = {"<|endoftext|>"}
    filtered = [
        (tok.replace("Ġ", "").replace("Ċ", ""), w)
        for tok, w in zip(tokens, last_row)
        if tok not in SPECIAL
    ]
    max_w = max(w for _, w in filtered) or 1.0
    return [{"text": t or "_", "weight": round(w / max_w, 3)} for t, w in filtered]


def get_completion_probs(
    sentence: str, intervene_cfg: dict | None = None
) -> list[tuple[str, float]]:
    prompt = f"{sentence}. So for pregnant women, this medication is "
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    actual_cfg = None
    if intervene_cfg:
        idx = find_token_index(prompt, "not")
        if idx != -1:
            actual_cfg = {**intervene_cfg, "target_idx": idx}

    with torch.no_grad():
        if actual_cfg:
            with intervene_attention(**actual_cfg):
                logits = model(**inputs).logits[0, -1, :]
        else:
            logits = model(**inputs).logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)

    return [
        (word, probs[tokenizer.encode(" " + word, add_special_tokens=False)[0]].item())
        for word in CANDIDATES
    ]

def get_yn_probs(
    sentence: str,
    intervene_cfg: dict | None = None,
    question: str = "Is this medication safe for pregnant women?",
) -> dict[str, float]:
    prompt = f"Fact: {sentence}\nQuestion: {question} \nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    actual_cfg = None
    if intervene_cfg:
        idx = find_token_index(prompt, "not")
        if idx != -1:
            actual_cfg = {**intervene_cfg, "target_idx": idx}

    with torch.no_grad():
        if actual_cfg:
            with intervene_attention(**actual_cfg):
                logits = model(**inputs).logits[0, -1, :]
        else:
            logits = model(**inputs).logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)

    yes_id = tokenizer.encode(" Yes")[0]
    no_id  = tokenizer.encode(" No")[0]
    return {
        "Yes": probs[yes_id].item(),
        "No":  probs[no_id].item(),
    }


def print_yn_comparison(sentence: str, before: dict, after: dict | None):
    print(f"\n  Sentence: \"{sentence}\"")

    def yn_ratio(d):
        total = d["Yes"] + d["No"]
        return {"Yes": d["Yes"] / total, "No": d["No"] / total}

    b_norm = yn_ratio(before)
    print(f"  {'':6} {'Before':>7}  {'':<{BAR_WIDTH}}", end="")
    print(f"   {'After':>7}  " if after else "")
    print("  " + "─" * 60)

    a_norm = yn_ratio(after) if after else None
    for word in ["Yes", "No"]:
        pb    = b_norm[word]
        bar_b = ("█" * int(pb * BAR_WIDTH)).ljust(BAR_WIDTH)
        if a_norm:
            pa    = a_norm[word]
            bar_a = "█" * int(pa * BAR_WIDTH)
            sign  = "↑" if pa - pb > 0.02 else ("↓" if pb - pa > 0.02 else " ")
            print(f"  {word:<6} {pb:>7.1%}  {bar_b}   {pa:>7.1%}  {bar_a}  {sign}")
        else:
            print(f"  {word:<6} {pb:>7.1%}  {bar_b}")

def print_attention_comparison(before: list[dict], after: list[dict]):
    COL = BAR_WIDTH + 10
    print(f"\n  {'Token':<14} {'Before':>6}  {'':<{BAR_WIDTH}}   {'After':>6}")
    print("  " + "─" * (14 + 6 + BAR_WIDTH + 6 + BAR_WIDTH + 10))
    for b, a in zip(before, after):
        bar_b = ("█" * int(b["weight"] * BAR_WIDTH)).ljust(BAR_WIDTH)
        bar_a = "█" * int(a["weight"] * BAR_WIDTH)
        marker = " ◄" if abs(a["weight"] - b["weight"]) > 0.15 else ""
        print(f"  {b['text']:<14} {b['weight']:>6.3f}  {bar_b}   {a['weight']:>6.3f}  {bar_a}{marker}")


def print_prob_comparison(sentence: str, before: list[tuple], after: list[tuple] | None):
    sorted_b = sorted(before, key=lambda x: x[1], reverse=True)
    max_b    = sorted_b[0][1] or 1.0

    print(f"\n\"{sentence}\"")

    if after:
        after_dict = dict(after)
        max_a = max(p for _, p in after) or 1.0

        print(f"  {'Word':<14} {'Before':>9}  {'Bar (before)':<{BAR_WIDTH}}   {'After':>9}  Bar (after)")
        print("  " + "─" * (14 + 9 + BAR_WIDTH + 9 + BAR_WIDTH + 10))
        for word, pb in sorted_b:
            pa    = after_dict.get(word, 0.0)
            bar_b = ("█" * int((pb / max_b) * BAR_WIDTH)).ljust(BAR_WIDTH)
            bar_a = "█" * int((pa / max_a) * BAR_WIDTH)
            delta = pa - pb
            sign  = ("↑" if delta > 0 else "↓") if abs(delta) > 1e-5 else " "
            print(f"  {word:<14} {pb:>9.5f}  {bar_b}   {pa:>9.5f}  {bar_a}  {sign}")
    else:
        print(f"  (no 'not'")
        max_b = sorted_b[0][1] or 1.0
        print(f"  {'Word':<14} {'Prob':>9}  Bar")
        print("  " + "─" * 48)
        for word, prob in sorted_b:
            bar = "█" * int((prob / max_b) * BAR_WIDTH)
            print(f"  {word:<14} {prob:>9.5f}  {bar}")

import json

def get_top_attention_tokens(token_weights: list[dict], n: int = 3) -> list[str]:
    sorted_tokens = sorted(token_weights, key=lambda x: x["weight"], reverse=True)
    return [t["text"] for t in sorted_tokens[:n]]


def build_display(
    token_weights: list[dict],
    yn: dict,
    is_correct: bool,
    negative_label: str = "NOT safe for pregnant women",
    positive_label: str = "Safe for pregnant women",
) -> dict:
    top = get_top_attention_tokens(token_weights)
    yes_pct = round(yn["Yes"] * 100, 1)
    no_pct  = round(yn["No"]  * 100, 1)

    if is_correct:
        label = negative_label
        note  = f"Given this input, the model predicts: No ({no_pct}%)"
    else:
        label = positive_label
        note  = f"Given this input, the model predicts: Yes ({yes_pct}%)"

    return {
        "top_attention_tokens": top,
        "prediction_label": label,
        "prediction_note": note
    }


def build_level2_config(
    sentence: str,
    multipliers: list[float],
    scenario_id: str = "medication_negation",
    question: str = "Is this medication safe for pregnant women?",
    negative_label: str = "NOT safe for pregnant women",
    positive_label: str = "Safe for pregnant women",
    hint_before: str = "Try increasing the weight of 'not'.",
    hint_after: str = "The output changed — but we still don't know if the model truly understands negation.",
    level_summary: str = (
        "The model processed every word, including 'not'. "
        "But its understanding of negation is fragile. "
        "Attention weights shifted, and so did the output — "
        "yet the model has no real grasp of logic or meaning. "
        "It follows statistical patterns, not reasoning."
    ),
) -> dict:
    not_idx = find_token_index(sentence, "not")

    states = []
    correction_threshold_weight = None

    prev_correct = None
    for m in multipliers:
        iv = dict(layer_idx=NUM_LAYERS - 1, target_idx=not_idx, multiplier=m) if not_idx != -1 else None
        weights = get_attention_weights(sentence, iv)
        yn      = get_yn_probs(sentence, iv, question=question)

        is_correct = yn["No"] > yn["Yes"]
        not_weight = next((w["weight"] for w in weights if w["text"] == "not"), None)

        # find the flip point
        if prev_correct is not None and is_correct and not prev_correct:
            correction_threshold_weight = not_weight

        states.append({
            "state_id": f"multiplier_{str(m).replace('.', '_')}x",
            "multiplier": m,
            "token_weights": weights,
            "output": {
                "yes_prob": round(yn["Yes"], 3),
                "no_prob":  round(yn["No"],  3),
                "is_correct": is_correct
            },
            "display": build_display(weights, yn, is_correct, negative_label, positive_label)
        })

        prev_correct = is_correct

    return {
        "level_id": 2,
        "scenario_id": scenario_id,
        "display_sentence": sentence + ".",
        "tokens": build_token_list(sentence),
        "intervention_states": states,
        "correction_threshold": {
            "not_weight": correction_threshold_weight,
            "description": "When 'not' attention weight exceeds this value, output flips to correct"
        },
        "ui_strings": {
            "panel_header": "Tokens with highest attention:",
            "output_header": "Given this input, the model predicts:",
            "hint_before": hint_before,
            "hint_after": hint_after
        },
        "level_summary": level_summary,
        "metadata": {
            "source": MODEL_NAME,
            "layer": "last",
            "multipliers_tested": multipliers,
            "version": "1.1"
        }
    }


def build_token_list(sentence: str) -> list[dict]:
    """Token classification, skipping the first one (attention sink)"""
    FUNCTION  = {"is", "for", "a", "an", "the", "of", "in", "on", "at"}
    NEGATION  = {"not", "never", "no", "neither", "nor"}

    tokens_raw = tokenizer.convert_ids_to_tokens(tokenizer(sentence)["input_ids"])
    tokens = [t.replace("Ġ", "").replace("Ċ", "").lower() for t in tokens_raw[1:]]  # skip the first one

    result = []
    for i, t in enumerate(tokens):
        if t in NEGATION:
            category = "negation"
        elif t in FUNCTION:
            category = "function"
        else:
            category = "content"

        result.append({
            "id": i,
            "text": t,
            "category": category,
            "default_weight": 0.0,   # filled in later from intervention_states[0]
            "draggable": category != "function",
            "is_key_token": category == "negation"
        })
    return result

if __name__ == "__main__":
    SENTENCE   = "This medication is not safe for pregnant women"
    MULTIPLIER = 30
    MULTIPLIERS = [1.0, 3.0, 6.0, 10.0, 15.0, 20.0, 30.0]
    LAYER_IDX  = NUM_LAYERS - 1

    config = build_level2_config(SENTENCE, MULTIPLIERS)

    not_idx = find_token_index(SENTENCE, "not")

    print(f"{SENTENCE}")
    print(f"'not' position : index {not_idx}  |  layer: {LAYER_IDX}  |  multiplier: {MULTIPLIER}x")

    intervene_cfg = dict(layer_idx=LAYER_IDX, target_idx=not_idx, multiplier=MULTIPLIER)

    # ── 1: attention weight comparison ─────────────────────────────────────
    w_before = get_attention_weights(SENTENCE)
    w_after  = get_attention_weights(SENTENCE, intervene_cfg)
    print_attention_comparison(w_before, w_after)

    # ── 2: prediction probability comparison ───────────────────────────────
    test_cases = [
        "This medication is not safe for pregnant women",
        "This medication is safe for pregnant women",
    ]


    for s in test_cases:
        ni = find_token_index(s, "not")
        iv = dict(layer_idx=LAYER_IDX, target_idx=ni, multiplier=MULTIPLIER) if ni != -1 else None

        b = get_yn_probs(s)
        a = get_yn_probs(s, iv) if iv else None
        print_yn_comparison(s, b, a)
        print()

    # fill default_weight from the multiplier=1.0 group
    baseline_weights = {w["text"]: w["weight"] for w in config["intervention_states"][0]["token_weights"]}
    for token in config["tokens"]:
        token["default_weight"] = baseline_weights.get(token["text"], 0.0)

    with open("level2_config_medication.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("Done → level2_config_medication.json")
    print(f"Correction threshold (not_weight): {config['correction_threshold']['not_weight']}")