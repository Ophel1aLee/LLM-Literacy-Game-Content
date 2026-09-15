"""Level 2 variant scenario: crane operator not paying attention (crane_operator_incorrigible)."""
import json
from level2_medication import build_level2_config

SENTENCE     = "This crane operator is not paying attention"
QUESTION     = "Is it safe to be near this crane?"
MULTIPLIERS  = [1.0, 3.0, 6.0, 10.0, 15.0, 20.0, 30.0]

NOTE = (
    "Ground truth: the operator isn't paying attention, so it's unsafe (correct "
    "answer: No). The model gets this wrong at baseline and stays wrong all the "
    "way through 30x amplification of 'not' — margin shrinks but never actually "
    "crosses. This is a real, non-fabricated example of attention-weight "
    "amplification failing to correct the model's judgment."
)

if __name__ == "__main__":
    print(f"Sentence : {SENTENCE}")
    print(f"Question : {QUESTION}")

    config = build_level2_config(
        SENTENCE,
        MULTIPLIERS,
        scenario_id="crane_operator_incorrigible",
        question=QUESTION,
        negative_label="NOT safe to be near this crane",
        positive_label="Safe to be near this crane",
        hint_before="Try increasing the weight of 'not'.",
        hint_after="The output shifts, but does it ever actually flip to correct?",
        level_summary=NOTE,
    )
    config["note"] = NOTE

    print("\n── Intervention states ──")
    for s in config["intervention_states"]:
        o = s["output"]
        mark = "✓" if o["is_correct"] else "✗"
        print(f"  {mark}  not×{s['multiplier']:<5} Yes={o['yes_prob']:.3f} No={o['no_prob']:.3f}")

    threshold = config["correction_threshold"]
    print(f"\nCorrection threshold (not_weight): {threshold['not_weight']}")
    if threshold["not_weight"] is None:
        print("  -> Never flips within the tested multiplier range.")

    with open("level2_config_crane.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("\nDone → level2_config_crane.json")
