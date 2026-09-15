"""
Quick probe: "This medication is not harmless for pregnant women"
双重否定: not + harmless = 有害
测试 not 和 harmless 两个拉杆各自及联合的影响
"""
from level2_attention import (
    get_yn_probs, get_attention_weights,
    sweep_two_tokens, print_yn_grid_both,
    find_token_index, NUM_LAYERS, BAR_WIDTH,
)

SENTENCE = "This medication is not harmless for pregnant women"
LAYER    = NUM_LAYERS - 1
MULTS    = [1.0, 3.0, 6.0, 10.0, 20.0, 30.0]

# 正确答案：句子 = 有害，所以 "Is it safe?" 答案应是 No
CORRECT_ANSWER = "No"

print(f"Sentence : {SENTENCE}")
print(f"Logic    : not + harmless = HARMFUL → correct answer is '{CORRECT_ANSWER}'")
print(f"Tokens   : not@{find_token_index(SENTENCE, 'not')}  "
      f"harmless@{find_token_index(SENTENCE, 'harmless')}\n")

# ── 1. baseline ──────────────────────────────────────────────────────────────
yn_base = get_yn_probs(SENTENCE)
total   = yn_base["Yes"] + yn_base["No"]
print("── Baseline ──")
for k in ["Yes", "No"]:
    pct = yn_base[k] / total
    bar = "█" * int(pct * BAR_WIDTH)
    tag = " ← CORRECT" if k == CORRECT_ANSWER else ""
    print(f"  {k:<4} {pct:>7.1%}  {bar}{tag}")

# ── 2. spot: 单独放大 not / harmless / 两者 ──────────────────────────────────
combos = [
    ({},                               "baseline"),
    ({"not": 10.0},                    "not×10"),
    ({"not": 30.0},                    "not×30"),
    ({"harmless": 10.0},               "harmless×10"),
    ({"harmless": 30.0},               "harmless×30"),
    ({"not": 10.0, "harmless": 10.0},  "not×10 + harmless×10"),
    ({"not": 30.0, "harmless": 30.0},  "not×30 + harmless×30"),
    ({"not": 30.0, "harmless": 0.1},   "not×30 + harmless×0.1 (suppress harmless)"),
    ({"not": 0.1,  "harmless": 30.0},  "not×0.1 + harmless×30 (suppress not)"),
]

print("\n── Spot comparison (normalized Yes / No) ──")
print(f"  {'Config':<42}  {'Yes':>6}  {'No':>6}  Correct?")
print("  " + "─" * 65)
for mults, label in combos:
    yn    = get_yn_probs(SENTENCE, mults or None)
    total = yn["Yes"] + yn["No"]
    yes_n = yn["Yes"] / total
    no_n  = yn["No"]  / total
    ok    = "✓" if yn["No"] > yn["Yes"] else "✗"
    print(f"  {ok}  {label:<40}  {yes_n:>6.1%}  {no_n:>6.1%}")

# ── 3. 2-D sweep: not × harmless ─────────────────────────────────────────────
print("\n── 2-D sweep: not_mult × harmless_mult ──")

# sweep_two_tokens 目前写死了 "pregnant"，这里直接手动跑
import torch, torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from level2_attention import model, tokenizer, intervene_attention_multi, DEVICE

def yn_probs_raw(sentence, token_multipliers):
    prompt  = f"Fact: {sentence}\nQuestion: Is this medication safe for pregnant women? \nAnswer:"
    inputs  = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    pairs   = []
    for kw, mult in token_multipliers.items():
        idx = find_token_index(prompt, kw)
        if idx != -1:
            pairs.append((idx, mult))
    with torch.no_grad():
        if pairs:
            with intervene_attention_multi(layer_idx=LAYER, interventions=pairs):
                logits = model(**inputs).logits[0, -1, :]
        else:
            logits = model(**inputs).logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)
    yes_id = tokenizer.encode(" Yes")[0]
    no_id  = tokenizer.encode(" No")[0]
    return {"Yes": probs[yes_id].item(), "No": probs[no_id].item()}

# build grid
grid = {}
for nm in MULTS:
    grid[nm] = {}
    for hm in MULTS:
        mults = {}
        if nm != 1.0: mults["not"]      = nm
        if hm != 1.0: mults["harmless"] = hm
        yn = yn_probs_raw(SENTENCE, mults)
        grid[nm][hm] = {
            "Yes": round(yn["Yes"], 4),
            "No":  round(yn["No"],  4),
            "is_correct": yn["No"] > yn["Yes"],
        }

# print No-prob table
print("\n  [No probability — higher = model leans 'No' = correct]\n")
col_label = "not \\ harm"
pm_cols   = "".join("  h={:<5}".format(hm) for hm in MULTS)
header    = f"  {col_label:<12}" + pm_cols
print(header)
print("  " + "─" * len(header))
for nm in MULTS:
    row = "  n={:<10}".format(nm)
    for hm in MULTS:
        row += "  {:.4f}".format(grid[nm][hm]["No"])
    print(row)

print("\n  [Correctness (✓ = No > Yes = model correctly says 'harmful')]\n")
print(header)
print("  " + "─" * len(header))
for nm in MULTS:
    row = "  n={:<10}".format(nm)
    for hm in MULTS:
        row += "  ✓      " if grid[nm][hm]["is_correct"] else "  ✗      "
    print(row)

# ── 4. attention weights for interesting corners ──────────────────────────────
print("\n── Attention weights at key corners ──")
corners = [
    ({},                    "baseline"),
    ({"not": 30.0},         "not×30"),
    ({"harmless": 30.0},    "harmless×30"),
    ({"not": 0.1, "harmless": 30.0}, "suppress not + harmless×30"),
]
for mults, label in corners:
    w = get_attention_weights(SENTENCE, mults or None)
    print(f"\n  [{label}]")
    for tok in w:
        bar = "█" * int(tok["weight"] * 20)
        print(f"    {tok['text']:<14} {tok['weight']:.3f}  {bar}")
