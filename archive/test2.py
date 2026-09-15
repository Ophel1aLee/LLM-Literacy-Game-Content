"""
Fact or Fiction — Level 1 Token Probability Scorer (GPT-2)
GPT-2 体积小，无需量化，Windows 直接跑

依赖：pip install transformers torch
模型会自动从 HuggingFace 下载（约 500MB），无需 API key
"""

import math, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "gpt2-large"   # 可选: gpt2 / gpt2-medium / gpt2-large / gpt2-xl


# ── 第一关场景 ──────────────────────────────────────────────────
LEVEL1 = {
    "level": 1,
    "full_sentence": (
        "Many people know that Albert Einstein was a poor student "
        "who failed math in school, which shows that even geniuses struggle early in life."
    ),
    "display_text": "Einstein failed math ___ , which shows that ___ .",
    "slots": [
        {
            "slot_index": 0,
            "context": "Many people know that Albert Einstein was a poor student who failed math",
            "candidates": ["in school", "in his early years", "in university", "he didn't"],
            "correct_candidate": "he didn't",
        },
        {
            "slot_index": 1,
            "context": (
                "Many people know that Albert Einstein was a poor student who failed math "
                "in school, which shows that"
            ),
            "candidates": [
                "even geniuses struggle",
                "failure builds character",
                "teachers misjudge talent",
                "this is a myth",
            ],
            "correct_candidate": "this is a myth",
        },
    ],
}


# ── 模型加载 ────────────────────────────────────────────────────
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[GPT-2] 加载模型 {MODEL_NAME}（device: {device}）...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    print("[GPT-2] 加载完成\n")
    return tokenizer, model, device


# ── 打分 ────────────────────────────────────────────────────────
def score_candidate(tokenizer, model, device, context: str, candidate: str) -> float:
    """计算 P(candidate | context)，返回 candidate 各 token log prob 均值。"""
    ctx_ids  = tokenizer.encode(context)
    full_ids = tokenizer.encode(context + " " + candidate)
    cand_ids = full_ids[len(ctx_ids):]

    if not cand_ids:
        return float("-inf")

    input_ids = torch.tensor([full_ids]).to(device)

    with torch.no_grad():
        outputs   = model(input_ids)
        logits    = outputs.logits[0]                                  # (seq_len, vocab)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    cand_logprobs = [
        log_probs[len(ctx_ids) - 1 + i, token_id].item()
        for i, token_id in enumerate(cand_ids)
    ]
    return sum(cand_logprobs) / len(cand_logprobs)


# ── 单词槽打分 ──────────────────────────────────────────────────
def score_slot(slot, tokenizer, model, device) -> dict:
    context    = slot["context"]
    candidates = slot["candidates"]

    print(f"  Context: ...{context[-60:]!r}")

    raw_scores = {
        cand: score_candidate(tokenizer, model, device, context, cand)
        for cand in candidates
    }

    for cand, lp in raw_scores.items():
        print(f"    {cand!r:30s}  avg_logprob = {lp:.4f}")

    linear     = {k: math.exp(v) for k, v in raw_scores.items()}
    total      = sum(linear.values())
    normalized = {k: v / total for k, v in linear.items()}

    scored = sorted(
        [
            {
                "token":           cand,
                "probability":     round(normalized[cand], 4),
                "raw_log_prob":    round(raw_scores[cand], 4),
                "is_correct":      cand == slot.get("correct_candidate"),
            }
            for cand in candidates
        ],
        key=lambda x: -x["probability"],
    )

    print(f"\n  → 归一化概率：")
    for c in scored:
        bar  = "█" * int(c["probability"] * 40)
        flag = "  ✓ CORRECT" if c["is_correct"] else ""
        print(f"    {c['token']!r:30s}  {c['probability']:.3f}  {bar}{flag}")

    return {
        "slot_index":        slot["slot_index"],
        "correct_candidate": slot.get("correct_candidate"),
        "candidates":        scored,
    }


# ── 主流程 ──────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(f"Fact or Fiction — GPT-2 logprob scorer ({MODEL_NAME})")
    print("=" * 60 + "\n")

    tokenizer, model, device = load_model()

    slot_results = []
    for slot in LEVEL1["slots"]:
        print(f"[Slot {slot['slot_index']}]")
        result = score_slot(slot, tokenizer, model, device)
        slot_results.append(result)
        print()

    output = {
        "level":         LEVEL1["level"],
        "full_sentence": LEVEL1["full_sentence"],
        "display_text":  LEVEL1["display_text"],
        "model_used":    MODEL_NAME,
        "slots":         slot_results,
    }

    out_path = "level1_config.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"✓ 配置已保存 → {out_path}")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()