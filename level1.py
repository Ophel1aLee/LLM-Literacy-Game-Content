import math, json, sys

sys.stdout.reconfigure(encoding="utf-8")

MODEL_PATH = "./models/mistral-7b-v0.2"

TEST_CASES = [
    {
        "id": "tongue_map",
        "description": "Tongue map myth",
        "context": "The tip of the tongue is most sensitive to",
        "candidates": [
            "sweet flavors",
            "all flavors equally",
            "touch and pressure",
            "salty flavors",
        ],
        "correct_candidate": "all flavors equally",
    },
    {
        "id": "brain_10pct",
        "description": "Brain 10% usage myth",
        "context": "The human brain's usage is typically",
        "candidates": [
            "10% of its capacity",
            "nearly 100% over the course of a day",
            "50% at any given moment",
            "30% during active tasks",
        ],
        "correct_candidate": "nearly 100% over the course of a day",
    },
    {
        "id": "goldfish_memory",
        "description": "Goldfish 3-second memory myth",
        "context": "Goldfish have a memory span of",
        "candidates": [
            "three seconds",
            "several months",
            "about thirty seconds",
            "a few days",
        ],
        "correct_candidate": "several months",
    },
    {
        "id": "camel_hump",
        "description": "Camel hump stores water myth",
        "context": "The humps on a camel's back are used to store",
        "candidates": [
            "water for long desert journeys",
            "fat as an energy reserve",
            "nutrients absorbed from food",
            "excess body heat",
        ],
        "correct_candidate": "fat as an energy reserve",
    },
    {
        "id": "bull_red",
        "description": "Bulls enraged by red myth",
        "context": "Bulls charge at a matador's cape because",
        "candidates": [
            "they are enraged by the color red",
            "they react to the movement of the cape",
            "the cape's shape resembles a threat",
            "loud crowd noise startles them",
        ],
        "correct_candidate": "they react to the movement of the cape",
    },
    {
        "id": "viking_horns",
        "description": "Viking horned helmet myth",
        "context": "Viking warriors are historically depicted wearing helmets",
        "candidates": [
            "with horns",
            "without horns",
            "made entirely of leather",
            "decorated with feathers",
        ],
        "correct_candidate": "without horns",
    },
    {
        "id": "lightning_twice",
        "description": "Lightning never strikes twice myth",
        "context": "Lightning strikes the same location",
        "candidates": [
            "never more than once",
            "repeatedly, especially tall structures",
            "only during the summer months",
            "only near bodies of water",
        ],
        "correct_candidate": "repeatedly, especially tall structures",
    },
    {
        "id": "knuckle_cracking",
        "description": "Knuckle cracking causes arthritis myth",
        "context": "Cracking your knuckles causes",
        "candidates": [
            "arthritis in the joints",
            "no long-term joint damage",
            "permanent finger swelling",
            "nerve damage over time",
        ],
        "correct_candidate": "no long-term joint damage",
    },
]


def load_hf(model_path: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import torch

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    print("[HF] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        cache_dir=model_path,
    )

    print("[HF] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        cache_dir=model_path,
        quantization_config=bnb_config,
        attn_implementation="eager",
        device_map={"": 0},
    )
    model.eval()
    return tokenizer, model


def score_hf(tokenizer, model, context: str, candidate: str) -> float:
    import torch

    ctx_ids  = tokenizer.encode(context, add_special_tokens=True)
    full_ids = tokenizer.encode(context + " " + candidate, add_special_tokens=True)
    cand_ids = full_ids[len(ctx_ids):]

    if not cand_ids:
        return float("-inf")

    input_ids = torch.tensor([full_ids]).to(model.device)

    with torch.no_grad():
        logits = model(input_ids).logits[0]

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    return sum(log_probs[len(ctx_ids) - 1 + i, tid].item() for i, tid in enumerate(cand_ids)) / len(cand_ids)


def top5_hf(tokenizer, model, context: str, num_tokens: int = 6) -> list:
    import torch

    ctx_ids = tokenizer.encode(context, add_special_tokens=True)
    input_ids = torch.tensor([ctx_ids]).to(model.device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=num_tokens,
            num_beams=5,
            num_return_sequences=5,
            output_scores=True,
            return_dict_in_generate=True,
            early_stopping=True,
        )

    results = []
    for seq, score in zip(outputs.sequences, outputs.sequences_scores):
        text = tokenizer.decode(seq[len(ctx_ids):], skip_special_tokens=True).strip()
        results.append({"token": text, "probability": round(score.exp().item(), 4)})
    return results


def main():
    print("=" * 60)
    print("Fact or Fiction — Local Mistral logprob scorer")
    print(f"Model path: {MODEL_PATH}")
    print("=" * 60)

    tokenizer, model = load_hf(MODEL_PATH)
    score_fn = lambda ctx, cand: score_hf(tokenizer, model, ctx, cand)

    test_summary = []
    configs = []

    for case in TEST_CASES:
        print(f"\n[{case['id']}] {case['description']}")
        print(f"  Context: {case['context']!r}")

        raw_scores = {cand: score_fn(case["context"], cand) for cand in case["candidates"]}
        linear = {k: math.exp(v) for k, v in raw_scores.items()}
        total = sum(linear.values())
        normalized = {k: v / total for k, v in linear.items()}

        ranked = sorted(normalized.items(), key=lambda x: -x[1])
        top_pred = ranked[0][0]
        correct = case["correct_candidate"]
        model_wrong = top_pred != correct

        for cand, prob in ranked:
            flag = "✓ CORRECT" if cand == correct else ""
            warn = "← model picked this!" if cand == top_pred and model_wrong else ""
            print(f"    {cand!r:35s}  {prob:.3f}  {flag} {warn}")

        print(f"\n  Top-5 free continuations:")
        top5 = top5_hf(tokenizer, model, case["context"])
        for i, item in enumerate(top5, 1):
            print(f"    {i}. {item['token']!r:30s}  {item['probability']:.4f}")

        test_summary.append({
            "id": case["id"],
            "description": case["description"],
            "model_top_pick": top_pred,
            "correct": correct,
            "model_wrong": model_wrong,
            "correct_prob": round(normalized[correct], 3),
        })

        configs.append({
            "id": case["id"],
            "description": case["description"],
            "context": case["context"],
            "correct_candidate": correct,
            "model_wrong": model_wrong,
            "candidates": sorted(
                [
                    {
                        "token": cand,
                        "probability": round(normalized[cand], 4),
                        "raw_log_prob": round(raw_scores[cand], 4),
                        "is_correct": cand == correct,
                    }
                    for cand in case["candidates"]
                ],
                key=lambda x: -x["probability"],
            ),
        })

    print("\n" + "=" * 60)
    print("Test summary")
    print("=" * 60)
    for s in test_summary:
        status = "❌ model wrong" if s["model_wrong"] else "✓ model correct"
        print(f"  {status}  [{s['id']}] {s['description']}")
        if s["model_wrong"]:
            print(f"           model picked: {s['model_top_pick']!r}  |  correct answer probability: {s['correct_prob']:.3f}")

    out_path = "level1_config.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(configs, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Config saved → {out_path}")


if __name__ == "__main__":
    main()
