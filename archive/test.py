from transformers import AutoTokenizer, MistralForCausalLM, BitsAndBytesConfig
import torch
import torch.nn.functional as F
from contextlib import contextmanager

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
LOCAL_PATH = "./models/mistral-7b-v0.2"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BAR_WIDTH  = 30

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=LOCAL_PATH)
model     = MistralForCausalLM.from_pretrained(
    MODEL_NAME,
    cache_dir=LOCAL_PATH,
    quantization_config=bnb_config,
    attn_implementation="eager",
    device_map="auto",
)
model.eval()

NUM_LAYERS = len(model.model.layers)


def find_token_index(text: str, keyword: str) -> int:
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(text)["input_ids"])
    for i, t in enumerate(tokens):
        if keyword in t.lower().replace("▁", ""):
            return i
    return -1


@contextmanager
def intervene_attention(layer_idx: int, target_idx: int, multiplier: float):
    """
    hook 挂在 k_proj（Key 投影层）上。
    Mistral 的 QKV 是三个独立线性层（q_proj / k_proj / v_proj），
    只放大 Key 让 not 更容易被其他 token 关注，比 QKV 全放大更干净。
    """
    k_proj = model.model.layers[layer_idx].self_attn.k_proj

    def hook(module, inp, out):
        modified = out.clone()
        modified[:, target_idx, :] *= multiplier
        return modified

    handle = k_proj.register_forward_hook(hook)
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

    SPECIAL = {"<unk>", "<s>", "</s>"}
    filtered = [
        (tok.replace("▁", "").replace("Ċ", ""), w)
        for tok, w in zip(tokens, last_row)
        if tok not in SPECIAL
    ]
    max_w = max(w for _, w in filtered) or 1.0
    return [{"text": t or "_", "weight": round(w / max_w, 3)} for t, w in filtered]


def get_yn_probs(sentence: str, intervene_cfg: dict | None = None) -> dict[str, float]:
    prompt = f"Fact: {sentence}\nQuestion: Is this medication safe for pregnant women?\nAnswer:"
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

    # Mistral 的 Yes/No tokenization：注意空格前缀
    yes_id = tokenizer.encode(" Yes", add_special_tokens=False)[0]
    no_id  = tokenizer.encode(" No",  add_special_tokens=False)[0]
    return {"Yes": probs[yes_id].item(), "No": probs[no_id].item()}


def print_attention_comparison(before: list[dict], after: list[dict]):
    print(f"\n  {'Token':<14} {'Before':>6}  {'Bar (before)':<{BAR_WIDTH}}   {'After':>6}  Bar (after)")
    print("  " + "─" * 72)
    for b, a in zip(before, after):
        bar_b  = ("█" * int(b["weight"] * BAR_WIDTH)).ljust(BAR_WIDTH)
        bar_a  = "█" * int(a["weight"] * BAR_WIDTH)
        marker = " ◄" if abs(a["weight"] - b["weight"]) > 0.15 else ""
        print(f"  {b['text']:<14} {b['weight']:>6.3f}  {bar_b}   {a['weight']:>6.3f}  {bar_a}{marker}")


def print_yn_comparison(sentence: str, before: dict, after: dict | None):
    def yn_ratio(d):
        total = d["Yes"] + d["No"]
        return {"Yes": d["Yes"] / total, "No": d["No"] / total}

    b_norm = yn_ratio(before)
    a_norm = yn_ratio(after) if after else None

    print(f"\n  句子: \"{sentence}\"")
    print(f"  {'':6} {'Before':>7}  {'':<{BAR_WIDTH}}", end="")
    print(f"   {'After':>7}  " if a_norm else "")
    print("  " + "─" * 68)

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


# ── 9. 主程序 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    SENTENCE   = "This medication is not safe for pregnant women"
    MULTIPLIER = 5.0
    LAYER_IDX  = NUM_LAYERS - 1

    not_idx = find_token_index(SENTENCE, "not")

    print(f"\n{'='*65}")
    print(f"  模型      : Mistral-7B-v0.1 (4-bit)")
    print(f"  句子      : {SENTENCE}")
    print(f"  'not' 位置 : index {not_idx}  |  层: {LAYER_IDX}  |  倍数: {MULTIPLIER}x")
    print(f"{'='*65}")

    intervene_cfg = dict(layer_idx=LAYER_IDX, target_idx=not_idx, multiplier=MULTIPLIER)

    # ── 一：注意力权重对比 ────────────────────────────────────────────────────
    print("\n\n【一】注意力权重对比（干预前 vs 干预后）")
    w_before = get_attention_weights(SENTENCE)
    w_after  = get_attention_weights(SENTENCE, intervene_cfg)
    print_attention_comparison(w_before, w_after)

    # ── 二：Yes / No 概率对比 ─────────────────────────────────────────────────
    print("\n\n【二】Yes / No 概率对比（干预前 vs 干预后）")

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