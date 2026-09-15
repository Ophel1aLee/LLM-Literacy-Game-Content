"""Level 2 variant scenario: bridge structural cracks."""
import json
from level2_attention import build_config, NUM_LAYERS, BAR_WIDTH, get_attention_weights, get_yn_probs

SENTENCE            = "This bridge has cracks that were never repaired"
QUESTION            = "Is it safe to drive across this bridge?"
INTERVENTION_TOKENS = ["cracks"]
MULTIPLIERS         = [1.0, 3.0, 6.0, 8.0, 9.5, 10.0, 15.0, 20.0, 30.0]
LAYER_IDX           = NUM_LAYERS - 1

if __name__ == "__main__":
    print(f"Sentence : {SENTENCE}")
    print(f"Question : {QUESTION}")
    print(f"Layer    : {LAYER_IDX}")

    print("\n── Baseline attention weights (ranked) ──")
    w_base = sorted(get_attention_weights(SENTENCE), key=lambda x: -x["weight"])
    for rank, w in enumerate(w_base):
        bar = "█" * int(w["weight"] * BAR_WIDTH)
        print(f"  #{rank}  {w['text']:<12} {w['weight']:>6.3f}  {bar}")

    print("\n── Tipping point: 'cracks' 9.5x -> 10.0x ──")
    for m in [9.5, 10.0]:
        yn = get_yn_probs(SENTENCE, {"cracks": m}, LAYER_IDX, question=QUESTION)
        pred = "Yes (unsafe correctly missed)" if yn["Yes"] > yn["No"] else "No (correctly flags unsafe)"
        print(f"  cracks×{m:<5} Yes={yn['Yes']:.4f} No={yn['No']:.4f}  -> {pred}")

    print("\n── Building level2_config_bridge.json … ──")
    config = build_config(
        SENTENCE,
        INTERVENTION_TOKENS,
        MULTIPLIERS,
        LAYER_IDX,
        question=QUESTION,
        scenario_id="bridge_structural_cracks",
        negative_label="NOT safe to drive across",
        positive_label="Safe to drive across",
        hint="Pull 'cracks' from 9.5x to 10x — the verdict flips right at that edge.",
    )

    with open("level2_config_bridge.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("Done → level2_config_bridge.json")
    print("Total intervention states : {}".format(len(config["intervention_states"])))
