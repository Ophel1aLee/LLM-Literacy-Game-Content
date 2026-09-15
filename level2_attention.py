from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from contextlib import contextmanager
from itertools import combinations

MODEL_NAME  = "gpt2-large"
LOCAL_PATH  = "./models/gpt2-large"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
CANDIDATES  = ["safe", "unsafe", "dangerous", "risky", "harmful", "recommended"]
BAR_WIDTH   = 30

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=LOCAL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    cache_dir=LOCAL_PATH,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    attn_implementation="eager",
)
model.eval().to(DEVICE)

NUM_LAYERS = len(model.transformer.h)


# ── token utilities ───────────────────────────────────────────────────────────

def find_token_index(text: str, keyword: str) -> int:
    """
    Returns the index of the first token whose cleaned text contains `keyword`.
    Falls back to multi-token span matching for BPE-split words.
    """
    token_ids = tokenizer(text)["input_ids"]
    tokens    = tokenizer.convert_ids_to_tokens(token_ids)
    cleaned   = [t.lower().replace("ġ", "").replace("Ġ", "") for t in tokens]

    for i, t in enumerate(cleaned):
        if keyword in t:
            return i

    for window in range(2, 5):
        for i in range(len(cleaned) - window + 1):
            if keyword in "".join(cleaned[i:i + window]):
                return i

    return -1


def debug_tokenization(text: str, keywords: list[str]) -> None:
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(text)["input_ids"])
    print(f"\n  Tokens: {[t.replace('Ġ', '') for t in tokens]}")
    for kw in keywords:
        idx = find_token_index(text, kw)
        found = tokens[idx] if idx != -1 else "NOT FOUND"
        print(f"  '{kw}' → index {idx}  ({found})")


# ── intervention helpers ──────────────────────────────────────────────────────

@contextmanager
def intervene_attention_multi(layer_idx: int, interventions: list[tuple[int, float]]):
    """Scale QKV projections for the given (token_idx, multiplier) pairs.
    GPT-J uses separate q_proj / k_proj / v_proj instead of a fused c_attn.
    """
    attn = model.transformer.h[layer_idx].attn

    def make_hook():
        def hook(module, inp, out):
            modified = out.clone()
            for token_idx, multiplier in interventions:
                modified[:, token_idx, :] *= multiplier
            return modified
        return hook

    handle = attn.c_attn.register_forward_hook(make_hook())
    try:
        yield
    finally:
        handle.remove()


def _build_interventions(
    prompt: str,
    token_multipliers: dict[str, float],
    layer_idx: int,
) -> dict | None:
    """
    Convert {keyword: multiplier} → {layer_idx, interventions[(idx, mult), ...]}.
    Skips tokens not found in the prompt.
    """
    pairs = []
    for keyword, mult in token_multipliers.items():
        idx = find_token_index(prompt, keyword)
        if idx != -1:
            pairs.append((idx, mult))
    if not pairs:
        return None
    return {"layer_idx": layer_idx, "interventions": pairs}


# ── model query functions ─────────────────────────────────────────────────────

def get_attention_weights(
    sentence: str,
    token_multipliers: dict[str, float] | None = None,
    layer_idx: int | None = None,
) -> list[dict]:
    if layer_idx is None:
        layer_idx = NUM_LAYERS - 1

    inputs = tokenizer(sentence, return_tensors="pt").to(DEVICE)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    cfg    = _build_interventions(sentence, token_multipliers or {}, layer_idx)

    with torch.no_grad():
        if cfg:
            with intervene_attention_multi(**cfg):
                outputs = model(**inputs, output_attentions=True)
        else:
            outputs = model(**inputs, output_attentions=True)

    last_row = outputs.attentions[-1][0].mean(dim=0)[-1].float().cpu().tolist()
    SPECIAL  = {"<|endoftext|>"}
    filtered = [
        (tok.replace("Ġ", "").replace("Ċ", ""), w)
        for tok, w in zip(tokens, last_row)
        if tok not in SPECIAL
    ]
    max_w = max(w for _, w in filtered) or 1.0
    return [{"text": t or "_", "weight": round(w / max_w, 3)} for t, w in filtered]


def get_yn_probs(
    sentence: str,
    token_multipliers: dict[str, float] | None = None,
    layer_idx: int | None = None,
    question: str = "Should pregnant women take this medication?",
) -> dict[str, float]:
    if layer_idx is None:
        layer_idx = NUM_LAYERS - 1

    prompt = f"Fact: {sentence}\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    cfg    = _build_interventions(prompt, token_multipliers or {}, layer_idx)

    with torch.no_grad():
        if cfg:
            with intervene_attention_multi(**cfg):
                logits = model(**inputs).logits[0, -1, :]
        else:
            logits = model(**inputs).logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)

    yes_id = tokenizer.encode(" Yes")[0]
    no_id  = tokenizer.encode(" No")[0]
    return {"Yes": probs[yes_id].item(), "No": probs[no_id].item()}


def get_completion_probs(
    sentence: str,
    token_multipliers: dict[str, float] | None = None,
    layer_idx: int | None = None,
) -> list[tuple[str, float]]:
    if layer_idx is None:
        layer_idx = NUM_LAYERS - 1

    prompt = f"{sentence}. So for pregnant women, this medication is "
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    cfg    = _build_interventions(prompt, token_multipliers or {}, layer_idx)

    with torch.no_grad():
        if cfg:
            with intervene_attention_multi(**cfg):
                logits = model(**inputs).logits[0, -1, :]
        else:
            logits = model(**inputs).logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)

    return [
        (word, probs[tokenizer.encode(" " + word, add_special_tokens=False)[0]].item())
        for word in CANDIDATES
    ]


# ── 2-D pairwise sweep ────────────────────────────────────────────────────────

def sweep_two_tokens(
    sentence: str,
    token_a: str,
    token_b: str,
    multipliers_a: list[float],
    multipliers_b: list[float],
    layer_idx: int | None = None,
    question: str = "Should pregnant women take this medication?",
) -> dict:
    """
    2-D grid keyed by (mult_a, mult_b).
    grid[ma][mb] = {"Yes", "No", "is_correct"}
    """
    if layer_idx is None:
        layer_idx = NUM_LAYERS - 1

    grid: dict[float, dict[float, dict]] = {}
    for ma in multipliers_a:
        grid[ma] = {}
        for mb in multipliers_b:
            mults = {}
            if ma != 1.0:
                mults[token_a] = ma
            if mb != 1.0:
                mults[token_b] = mb
            yn = get_yn_probs(sentence, mults or None, layer_idx, question=question)
            grid[ma][mb] = {
                "Yes": round(yn["Yes"], 4),
                "No":  round(yn["No"],  4),
                "is_correct": yn["No"] > yn["Yes"],
            }
    return grid


def print_yn_grid(
    grid: dict,
    multipliers_a: list[float],
    multipliers_b: list[float],
    token_a: str,
    token_b: str,
    metric: str = "No",
):
    col_label = "{} \\ {}".format(token_a[:5], token_b[:5])
    pm_cols   = "".join("  {:<7}".format("b={:.0f}".format(mb)) for mb in multipliers_b)
    header    = "  {:<14}".format(col_label) + pm_cols
    print(header)
    print("  " + "─" * len(header))
    for ma in multipliers_a:
        row = "  {:<14}".format("a={:.0f}".format(ma))
        for mb in multipliers_b:
            cell = grid[ma][mb]
            if metric == "correct":
                val = "  ✓      " if cell["is_correct"] else "  ✗      "
            else:
                val = "  {:.4f}".format(cell[metric])
            row += val
        print(row)


def print_yn_grid_both(
    grid: dict,
    multipliers_a: list[float],
    multipliers_b: list[float],
    token_a: str,
    token_b: str,
):
    print("\n  [No-prob  (higher = model says 'No' = correct)]\n")
    print_yn_grid(grid, multipliers_a, multipliers_b, token_a, token_b, metric="No")
    print("\n  [Correctness  (✓ = No > Yes)]\n")
    print_yn_grid(grid, multipliers_a, multipliers_b, token_a, token_b, metric="correct")


def sweep_all_pairs(
    sentence: str,
    intervention_tokens: list[str],
    multipliers: list[float],
    layer_idx: int | None = None,
    question: str = "Should pregnant women take this medication?",
) -> dict[tuple[str, str], dict]:
    """Run sweep_two_tokens for every pair in intervention_tokens."""
    results = {}
    for tok_a, tok_b in combinations(intervention_tokens, 2):
        print(f"\n── 2-D sweep: {tok_a} × {tok_b} ──")
        grid = sweep_two_tokens(sentence, tok_a, tok_b, multipliers, multipliers, layer_idx, question=question)
        print_yn_grid_both(grid, multipliers, multipliers, tok_a, tok_b)
        results[(tok_a, tok_b)] = grid
    return results


# ── spot comparison ───────────────────────────────────────────────────────────

def print_spot_comparison(
    sentence: str,
    combos: list[tuple[dict[str, float], str]],
):
    print("\n── Spot YN comparison ──")
    for mults, label in combos:
        yn    = get_yn_probs(sentence, mults or None)
        total = yn["Yes"] + yn["No"]
        yes_n = yn["Yes"] / total
        no_n  = yn["No"]  / total
        tag   = "✓" if yn["No"] > yn["Yes"] else "✗"
        print(f"  {tag}  {label:<36}  Yes={yes_n:.3%}  No={no_n:.3%}")


# ── JSON config builder ───────────────────────────────────────────────────────

def get_top_attention_tokens(token_weights: list[dict], n: int = 3) -> list[str]:
    return [t["text"] for t in sorted(token_weights, key=lambda x: x["weight"], reverse=True)[:n]]


def build_display(
    token_weights: list[dict],
    yn: dict,
    is_correct: bool,
    negative_label: str = "NOT safe for pregnant women",
    positive_label: str = "Safe for pregnant women",
) -> dict:
    top     = get_top_attention_tokens(token_weights)
    yes_pct = round(yn["Yes"] * 100, 1)
    no_pct  = round(yn["No"]  * 100, 1)
    label   = negative_label if is_correct else positive_label
    note    = "Model predicts: {} ({}%)".format(
        "No" if is_correct else "Yes",
        no_pct if is_correct else yes_pct,
    )
    return {"top_attention_tokens": top, "prediction_label": label, "prediction_note": note}


def build_token_list(sentence: str, key_tokens: set[str] | None = None) -> list[dict]:
    FUNCTION = {"is", "for", "a", "an", "the", "of", "in", "on", "at", "to"}
    MODAL    = {"may", "might", "could", "should", "would", "can"}
    tokens_raw = tokenizer.convert_ids_to_tokens(tokenizer(sentence)["input_ids"])
    tokens     = [t.replace("Ġ", "").replace("Ċ", "").lower() for t in tokens_raw[1:]]
    result = []
    for i, t in enumerate(tokens):
        if t in MODAL:
            cat = "modal"
        elif t in FUNCTION:
            cat = "function"
        else:
            cat = "content"
        result.append({
            "id": i, "text": t, "category": cat,
            "default_weight": 0.0,
            "draggable": True,
            "is_key_token": (key_tokens is not None and t in key_tokens),
        })
    return result


def build_config(
    sentence: str,
    intervention_tokens: list[str],
    multipliers: list[float],
    layer_idx: int | None = None,
    question: str = "Should pregnant women take this medication?",
    scenario_id: str = "medication_causal",
    negative_label: str = "NOT safe for pregnant women",
    positive_label: str = "Safe for pregnant women",
    panel_header: str = "Tokens with highest attention:",
    output_header: str = "Given this input, the model predicts:",
    hint: str = "Try adjusting 'not' or 'safe' — do they pull in opposite directions?",
) -> dict:
    """
    Build a JSON config with one intervention state per combination of
    (token → multiplier). Each token is swept independently (others at 1×).
    For the JSON we store single-token sweeps; pairwise grids are in sweep_grids.
    """
    if layer_idx is None:
        layer_idx = NUM_LAYERS - 1

    # single-token sweep states
    states = []
    for token in intervention_tokens:
        for mult in multipliers:
            mults   = {token: mult} if mult != 1.0 else {}
            weights = get_attention_weights(sentence, mults or None, layer_idx)
            yn      = get_yn_probs(sentence, mults or None, layer_idx, question=question)
            correct = yn["No"] > yn["Yes"]
            states.append({
                "state_id":   "{}_{}x".format(token, str(mult).replace(".", "_")),
                "token":      token,
                "multiplier": mult,
                "token_weights": weights,
                "output": {
                    # 5 decimals: at high multipliers both probs can shrink into the
                    # same 0.001 bucket at 3dp, which makes the Yes/No bar in Unity
                    # (normalized as yes/(yes+no)) collapse back to a misleading 50/50
                    # even though the model still has a real, decisive preference.
                    "yes_prob":   round(yn["Yes"], 5),
                    "no_prob":    round(yn["No"],  5),
                    "is_correct": correct,
                },
                "display": build_display(weights, yn, correct, negative_label, positive_label),
            })

    # pairwise grids
    sweep_grids = {}
    for tok_a, tok_b in combinations(intervention_tokens, 2):
        grid = sweep_two_tokens(sentence, tok_a, tok_b, multipliers, multipliers, layer_idx, question=question)
        key  = "{}_x_{}".format(tok_a, tok_b)
        sweep_grids[key] = {
            str(ma): {str(mb): grid[ma][mb] for mb in multipliers}
            for ma in multipliers
        }

    # baseline for default_weight
    baseline_state  = next(s for s in states if s["multiplier"] == 1.0)
    baseline_w_map  = {w["text"]: w["weight"] for w in baseline_state["token_weights"]}
    tokens          = build_token_list(sentence, key_tokens=set(intervention_tokens))
    for tok in tokens:
        tok["default_weight"] = baseline_w_map.get(tok["text"], 0.0)

    return {
        "level_id": 2,
        "scenario_id": scenario_id,
        "display_sentence": sentence + ".",
        "tokens": tokens,
        "intervention_tokens": intervention_tokens,
        "intervention_states": states,
        "sweep_grids": sweep_grids,
        "ui_strings": {
            "panel_header": panel_header,
            "output_header": output_header,
            "hint": hint,
        },
        "metadata": {
            "source":    MODEL_NAME,
            "layer":     layer_idx,
            "multipliers_tested": multipliers,
            "version":   "3.0",
        },
    }


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    SENTENCE             = "This medication is not safe for pregnant women"
    INTERVENTION_TOKENS  = ["not", "safe"]
    MULTIPLIERS          = [1.0, 3.0, 6.0, 10.0, 20.0, 30.0]
    LAYER_IDX            = NUM_LAYERS - 1

    # ── 0. tokenization check ────────────────────────────────────────────────
    print(f"Sentence : {SENTENCE}")
    debug_tokenization(SENTENCE, INTERVENTION_TOKENS)

    yn_prompt = f"Fact: {SENTENCE}\nQuestion: Is this medication safe for pregnant women? \nAnswer:"
    print(f"\n── YN prompt tokenization ──")
    debug_tokenization(yn_prompt, INTERVENTION_TOKENS)

    print(f"Layer    : {LAYER_IDX}")

    # ── 1. baseline attention ────────────────────────────────────────────────
    print("\n── Baseline attention weights ──")
    w_base = get_attention_weights(SENTENCE)
    for w in w_base:
        bar = "█" * int(w["weight"] * BAR_WIDTH)
        print(f"  {w['text']:<14} {w['weight']:>6.3f}  {bar}")

    # ── 2. attention shift per token×30 ──────────────────────────────────────
    print("\n── Attention: each token ×30 vs baseline ──")
    w_interventions = {
        tok: get_attention_weights(SENTENCE, {tok: 30.0})
        for tok in INTERVENTION_TOKENS
    }
    header_cols = "".join("  {:>10}".format(t + "×30") for t in INTERVENTION_TOKENS)
    print(f"\n  {'Token':<14} {'Base':>6}" + header_cols)
    print("  " + "─" * (14 + 8 + 12 * len(INTERVENTION_TOKENS)))
    for i, b in enumerate(w_base):
        row = f"  {b['text']:<14} {b['weight']:>6.3f}"
        for tok in INTERVENTION_TOKENS:
            a      = w_interventions[tok][i]
            mark   = " ◄" if abs(a["weight"] - b["weight"]) > 0.15 else "  "
            row   += "  {:>8.3f}{}".format(a["weight"], mark)
        print(row)

    # ── 3. pairwise 2-D sweeps ───────────────────────────────────────────────
    sweep_all_pairs(SENTENCE, INTERVENTION_TOKENS, MULTIPLIERS, LAYER_IDX)

    # ── 4. spot comparisons ──────────────────────────────────────────────────
    combos = [
        ({},                                            "baseline"),
        ({"not":  30.0},                                "not×30"),
        ({"safe": 30.0},                                "safe×30"),
        ({"not": 30.0, "safe": 30.0},                   "not×30 + safe×30"),
        ({"not": 10.0, "safe": 10.0},                   "not×10 + safe×10"),
    ]
    print_spot_comparison(SENTENCE, combos)

    # ── 5. build and save JSON config ─────────────────────────────────────────
    print("\n── Building level2_config.json … ──")
    config = build_config(SENTENCE, INTERVENTION_TOKENS, MULTIPLIERS, LAYER_IDX)

    with open("level2_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("Done → level2_config.json")
    print("Total intervention states : {}".format(len(config["intervention_states"])))
    print("Pairwise sweep grids      : {}".format(list(config["sweep_grids"].keys())))
