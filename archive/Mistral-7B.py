from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
local_path = "./models/mistral-7b"

quant_config = BitsAndBytesConfig(load_in_4bit=True)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=local_path
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto",
    output_attentions=True,
    cache_dir=local_path
)
model.eval()

def get_attention_weights(sentence: str) -> list[dict]:
    inputs = tokenizer(sentence, return_tensors="pt").to("cuda")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = model(**inputs)

    attn_last = outputs.attentions[-1][0]
    avg_heads  = attn_last.mean(dim=0)
    last_row   = avg_heads[-1, :]
    raw        = last_row.float().cpu().tolist()

    # 过滤特殊token再归一化
    SPECIAL = {"<s>", "</s>", "<unk>", "<pad>"}
    filtered = [(token.lstrip("▁ "), w)
                for token, w in zip(tokens, raw)
                if token not in SPECIAL]

    max_w = max(w for _, w in filtered)

    return [
        {"text": text, "weight": round(w / max_w, 3)}
        for text, w in filtered
    ]

def print_weights(words: list[dict]):
    print(f"\n{'Token':<20} {'Weight':>8}")
    print("-" * 35)
    for w in words:
        bar = "█" * int(w["weight"] * 30)
        print(f"{w['text']:<20} {w['weight']:>6.3f}  {bar}")

if __name__ == "__main__":
    sentence = "This medication is not safe for pregnant women"
    print(f"句子: {sentence}")
    words = get_attention_weights(sentence)
    print_weights(words)

import torch.nn.functional as F


def check_understanding(sentence: str):
    # 构造填空题
    prompt = f"{sentence}. Therefore, for them it is "
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model(**inputs)
        # 拿到最后一个 Token 的预测结果（Logits）
        next_token_logits = outputs.logits[0, -1, :]

        # 转化为概率
        probs = F.softmax(next_token_logits, dim=-1)

    # 我们关注的候选词
    candidates = ["safe", "unsafe", "dangerous", "risky"]
    results = []

    for word in candidates:
        # 拿到这个词对应的 ID
        word_id = tokenizer.encode(word, add_special_tokens=False)[-1]
        word_prob = probs[word_id].item()
        results.append((word, word_prob))

    # 打印结果
    print(f"\n测试句子: {sentence}")
    print(f"{'预测词':<12} {'模型给出的概率':>10}")
    print("-" * 30)
    for word, prob in sorted(results, key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob * 50)
        print(f"{word:<12} {prob:>10.4f}  {bar}")


# 跑两个对照组
check_understanding("This medication is not safe for pregnant women")
check_understanding("This medication is safe for pregnant women")