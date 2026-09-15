from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
local_path  = "./models/llama3-8b"

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

    SPECIAL = {"<s>", "</s>", "<unk>", "<pad>",
               "<|begin_of_text|>", "<|end_of_text|>"}  # LLaMA3的特殊token
    filtered = [(token.lstrip("▁Ġ "), w)
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