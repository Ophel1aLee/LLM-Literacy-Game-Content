from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import json
import os

# ── 配置 ────────────────────────────────────────────────────────────────────
MODEL_NAME = "gpt2-large"
LOCAL_PATH = "./models/gpt2-large"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
NUM_RUNS   = 20     # 每个 prompt 生成几条
MAX_TOKENS = 25     # 标题不需要太长
TEMPERATURE = 0.9
TOP_P       = 0.95
OUTPUT_RAW  = "level4_raw.json"

# ── 加载模型 ─────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_NAME} on {DEVICE}...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME, cache_dir=LOCAL_PATH)
model     = GPT2LMHeadModel.from_pretrained(MODEL_NAME, cache_dir=LOCAL_PATH)
model.eval().to(DEVICE)
print("Model loaded.\n")


# ── 事件数据 ─────────────────────────────────────────────────────────────────
# 每个 event 包含 facts 和三种 publication 的 prompt 构建方式
EVENTS = [
    {
        "id": "city_traffic",
        "facts": (
            "City council approved a new downtown traffic policy. "
            "Private vehicles restricted in a 2km zone. "
            "Measure aimed at reducing emissions. "
            "Bus and tram capacity to be increased by 30%."
        ),
        "publications": [
            {
                "id": "suburban",
                "name": "The Daily Standard",
                "readership": "suburban commuters, car owners",
                "prompt_suffix": (
                    "Write a newspaper headline for suburban commuters "
                    "who rely on their cars, using a concerned and critical tone:"
                ),
            },
            {
                "id": "urban",
                "name": "The City Post",
                "readership": "urban residents, environmental advocates",
                "prompt_suffix": (
                    "Write a newspaper headline for urban residents "
                    "who support green initiatives, using an optimistic and forward-looking tone:"
                ),
            },
            {
                "id": "clickbait",
                "name": "The Click Tribune",
                "readership": "everyone, engagement-first",
                "prompt_suffix": (
                    "Write a breaking news headline that maximises clicks "
                    "using emotional language and a sense of urgency:"
                ),
            },
        ],
    },
    # ── 第二个事件（可选，按同样格式扩充）────────────────────────────────────
    # {
    #     "id": "school_policy",
    #     "facts": "...",
    #     "publications": [...],
    # },
]


# ── 生成函数 ─────────────────────────────────────────────────────────────────
def generate_headlines(prompt: str, n: int) -> list[str]:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[1]
    headlines = []
    for _ in range(n):
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
        # 只保留第一行（有时 GPT-2 会续写多行）
        first_line = generated.split("\n")[0].strip()
        headlines.append(first_line)
    return headlines


# ── 主采集循环 ───────────────────────────────────────────────────────────────
results = []

for event in EVENTS:
    print(f"=== Event: {event['id']} ===")
    event_result = {
        "id":    event["id"],
        "facts": event["facts"],
        "publications": [],
    }

    for pub in event["publications"]:
        prompt = f"{event['facts']} {pub['prompt_suffix']}"
        print(f"\n  [{pub['id']}] {pub['name']}")
        print(f"  Prompt: {prompt[:80]}...")

        headlines = generate_headlines(prompt, NUM_RUNS)

        for i, h in enumerate(headlines):
            print(f"    {i+1:>2}. {h}")

        event_result["publications"].append({
            "id":           pub["id"],
            "name":         pub["name"],
            "readership":   pub["readership"],
            "prompt_suffix": pub["prompt_suffix"],
            "generated":    headlines,   # 全部原始输出，手动筛选后填 selected
            "selected":     [],          # ← 手动填入最终选用的 3-5 条
        })

    results.append(event_result)
    print()

# ── 保存原始数据 ──────────────────────────────────────────────────────────────
with open(OUTPUT_RAW, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Raw data saved to {OUTPUT_RAW}")
print("Next step: open the file, review each 'generated' list, and fill in 'selected'.")