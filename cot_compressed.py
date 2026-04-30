"""中性 prompt + 压缩知识（去停用词，仅保留内容词）实验。
4B + 8B, cf_medqa + cf_arc, 各50条, max_new_tokens=1024。
"""
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_NUM_TO_LETTER = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "am",
    "and", "but", "or", "nor", "for", "yet", "so", "both", "either", "neither",
    "in", "on", "at", "to", "of", "by", "from", "with", "into", "through",
    "during", "before", "after", "above", "below", "between", "under", "over",
    "about", "against", "along", "among", "around", "upon",
    "that", "which", "who", "whom", "whose", "this", "these", "those",
    "it", "its", "he", "she", "his", "her", "they", "them", "their",
    "we", "our", "you", "your", "i", "me", "my",
    "if", "then", "than", "when", "where", "while", "because", "since",
    "as", "also", "not", "no", "very", "more", "most", "such", "each",
    "every", "all", "any", "some", "many", "much", "few", "several",
    "other", "another", "own", "same", "only", "just", "even",
    "there", "here", "how", "what", "why",
    "up", "out", "off", "down", "away", "back",
    "however", "therefore", "thus", "hence", "moreover", "furthermore",
    "although", "though", "whereas", "meanwhile", "nevertheless",
    "specifically", "particularly", "especially", "additionally",
    "well", "often", "usually", "typically", "generally", "commonly",
    "rather", "quite", "already", "still", "too",
}


def compress_passage(text):
    """去停用词，保留内容词，用空格拼接。"""
    words = text.split()
    content = [w for w in words if w.lower().rstrip(".,;:!?()\"'") not in STOP_WORDS]
    return " ".join(content)


def normalize_options(options, correct_letter):
    first_key = next(iter(options))
    if first_key in _NUM_TO_LETTER:
        return {_NUM_TO_LETTER[k]: v for k, v in options.items()}, _NUM_TO_LETTER.get(correct_letter, correct_letter)
    return options, correct_letter


def extract_answer(text, valid_labels):
    m = re.search(r'[Tt]he answer is\s*([A-E])', text)
    if m and m.group(1) in valid_labels:
        return m.group(1), "answer_is"
    m = re.search(r'[Aa]nswer\s*:\s*([A-E])', text)
    if m and m.group(1) in valid_labels:
        return m.group(1), "answer_colon"
    m = re.search(r'\b([A-E])\s*\.?\s*$', text.strip())
    if m and m.group(1) in valid_labels:
        return m.group(1), "trailing"
    m = re.search(r'(?:option|choice)\s+([A-E])', text, re.IGNORECASE)
    if m and m.group(1) in valid_labels:
        return m.group(1), "option"
    return "?", "no_match"


def load_real_qa(path):
    real_qa = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            opts, cl = normalize_options(d["options"], d["correct_letter"])
            real_qa[d["question"][:100]] = cl
    return real_qa


def load_cf_samples(path, n):
    samples = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            samples.append(json.loads(line))
    return samples


def run_eval(model, tokenizer, model_label, ds_name, cf_samples, real_qa, device):
    results = {"cf": 0, "param": 0, "other": 0}
    pattern_stats = {}
    gen_lengths = []
    compress_ratios = []
    examples_param = []
    examples_other = []

    for idx, sample in enumerate(cf_samples):
        options, cf_correct = normalize_options(sample["options"], sample["correct_letter"])
        labels = sorted(options.keys())
        option_lines = "\n".join(f"{lb}. {options[lb]}" for lb in labels)
        qkey = sample["question"][:100]
        real_correct = real_qa.get(qkey, "?")

        original = sample["passage"]
        compressed = compress_passage(original)
        compress_ratios.append(len(compressed.split()) / max(len(original.split()), 1))

        prompt = f"""/no_think
{compressed}

Question: {sample['question']}
{option_lines}

Let's think step by step, then give the answer.
You MUST end your response with exactly "The answer is X" where X is {', '.join(labels[:-1])}, or {labels[-1]}."""

        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=1024, do_sample=False)
        gen_tokens = out[0][ids.shape[-1]:]
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        gen_lengths.append(len(gen_tokens))

        pred, pattern = extract_answer(gen_text, set(labels))
        pattern_stats[pattern] = pattern_stats.get(pattern, 0) + 1

        if pred == cf_correct:
            results["cf"] += 1
        elif pred == real_correct:
            results["param"] += 1
            if len(examples_param) < 3:
                examples_param.append(f"    [{idx}] cf={cf_correct} real={real_correct} pred={pred} len={len(gen_tokens)}\n         ...{gen_text[-100:]}")
        else:
            results["other"] += 1
            if len(examples_other) < 2:
                examples_other.append(f"    [{idx}] cf={cf_correct} real={real_correct} pred={pred}({pattern}) len={len(gen_tokens)}\n         ...{gen_text[-100:]}")

    total = len(cf_samples)
    avg_len = sum(gen_lengths) / len(gen_lengths)
    avg_ratio = sum(compress_ratios) / len(compress_ratios)
    print(f"\n{'='*60}")
    print(f"{model_label} | {ds_name} | {total} samples | COMPRESSED prompt")
    print(f"{'='*60}")
    print(f"  反事实遵从:     {results['cf']:>3}/{total} = {results['cf']/total:.1%}")
    print(f"  参数化知识:     {results['param']:>3}/{total} = {results['param']/total:.1%}")
    print(f"  其他/提取失败:  {results['other']:>3}/{total} = {results['other']/total:.1%}")
    print(f"  平均生成长度:   {avg_len:.0f} tokens")
    print(f"  知识压缩率:     {avg_ratio:.1%} (保留词比例)")
    print(f"  提取 pattern:   {pattern_stats}")
    if examples_param:
        print(f"\n  --- 坚持参数化知识 ---")
        for e in examples_param:
            print(e)
    if examples_other:
        print(f"\n  --- 其他/提取失败 ---")
        for e in examples_other:
            print(e)

    # 打印一条压缩 vs 原文对比
    if cf_samples:
        s0 = cf_samples[0]
        orig = s0["passage"]
        comp = compress_passage(orig)
        print(f"\n  --- 压缩示例 (sample 0) ---")
        print(f"  原文 ({len(orig.split())} words): {orig[:120]}...")
        print(f"  压缩 ({len(comp.split())} words): {comp[:120]}...")


# 预加载数据
medqa_real = load_real_qa("data/ood/medqa.jsonl")
arc_real = load_real_qa("data/counterfactual/arc_easy.jsonl")
medqa_cf = load_cf_samples("data/counterfactual/cf_medqa_val.jsonl", 50)
arc_cf = load_cf_samples("data/counterfactual/cf_arc_easy_val.jsonl", 50)

# --- 8B ---
print("\n" + "#" * 70)
print("# qwen3-8B — compressed prompt")
print("#" * 70)
tokenizer_8b = AutoTokenizer.from_pretrained("hugglingface_model/qwen3-8B", trust_remote_code=True)
if tokenizer_8b.pad_token is None:
    tokenizer_8b.pad_token = tokenizer_8b.eos_token
model_8b = AutoModelForCausalLM.from_pretrained(
    "hugglingface_model/qwen3-8B", dtype=torch.bfloat16, trust_remote_code=True,
).to("cuda:0").eval()

run_eval(model_8b, tokenizer_8b, "qwen3-8B", "cf_medqa_val", medqa_cf, medqa_real, "cuda:0")
run_eval(model_8b, tokenizer_8b, "qwen3-8B", "cf_arc_easy_val", arc_cf, arc_real, "cuda:0")

del model_8b, tokenizer_8b
torch.cuda.empty_cache()
import gc; gc.collect()

# --- 4B ---
print("\n" + "#" * 70)
print("# qwen3-4B — compressed prompt")
print("#" * 70)
tokenizer_4b = AutoTokenizer.from_pretrained("hugglingface_model/qwen3-4B", trust_remote_code=True)
if tokenizer_4b.pad_token is None:
    tokenizer_4b.pad_token = tokenizer_4b.eos_token
model_4b = AutoModelForCausalLM.from_pretrained(
    "hugglingface_model/qwen3-4B", dtype=torch.bfloat16, trust_remote_code=True,
).to("cuda:0").eval()

run_eval(model_4b, tokenizer_4b, "qwen3-4B", "cf_medqa_val", medqa_cf, medqa_real, "cuda:0")
run_eval(model_4b, tokenizer_4b, "qwen3-4B", "cf_arc_easy_val", arc_cf, arc_real, "cuda:0")

print("\n\nDONE")
