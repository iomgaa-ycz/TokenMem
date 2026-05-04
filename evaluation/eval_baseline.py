"""eval_baseline — No-Memory / VanillaRAG baseline 评测。

统一入口，通过 --method 切换评测模式。
评测方式: CoT + nothink 生成 → regex 提取答案字母。
VanillaRAG 的段落通过 LLMLingua-2 压缩到 --compress-target-token (默认 64) 个 token。
Prompt 使用中性框定（无 "Reference:" 标签）。

使用：
    python -m evaluation.eval_baseline \\
        --model-path hugglingface_model/qwen3-4B \\
        --method vanilla_rag \\
        --dataset medqa \\
        --data-dir data/ood \\
        --output-dir results/baseline \\
        --compress-target-token 64 \\
        --cot-max-new-tokens 2048 \\
        --batch-size 4
"""

import argparse
import json
import logging
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from llmlingua import PromptCompressor

import torch
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers import CONFIG_MAPPING

if "ministral3" not in CONFIG_MAPPING:
    from transformers.models.mistral.configuration_mistral import MistralConfig

    CONFIG_MAPPING.register("ministral3", MistralConfig)

logger = logging.getLogger(__name__)

_NUM_TO_LETTER = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}


# ---------------------------------------------------------------------------
# Passage 压缩 (LLMLingua-2)
# ---------------------------------------------------------------------------

_compressor = None


def _get_compressor() -> "PromptCompressor":
    """懒加载 LLMLingua-2 PromptCompressor 单例。"""
    global _compressor
    if _compressor is None:
        from llmlingua import PromptCompressor

        _compressor = PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            use_llmlingua2=True,
        )
        logger.info("LLMLingua-2 PromptCompressor 已初始化")
    return _compressor


def compress_passage(passage: str, target_token: int = 64) -> str:
    """用 LLMLingua-2 将 passage 压缩到目标 token 数。

    参数：
        passage: 原始知识段落。
        target_token: 压缩目标 token 数。

    返回：
        压缩后的文本字符串。
    """
    compressor = _get_compressor()
    result = compressor.compress_prompt(
        context=[passage],
        instruction="",
        question="",
        target_token=target_token,
    )
    return result["compressed_prompt"]


def _supports_thinking(tokenizer: PreTrainedTokenizerBase) -> bool:
    """检测 tokenizer 是否支持 enable_thinking 参数（Qwen3 系列）。"""
    template = getattr(tokenizer, "chat_template", "") or ""
    return "enable_thinking" in template


# ---------------------------------------------------------------------------
# Prompt 构造
# ---------------------------------------------------------------------------


def normalize_options(options: Dict[str, str], correct_letter: str) -> tuple:
    """归一化选项 key 和 correct_letter。

    将数字 key（1/2/3/4/5）转为字母（A/B/C/D/E），
    同时转换 correct_letter。字母 key 保持不变。

    参数：
        options: 原始选项字典。
        correct_letter: 原始正确答案标识。

    返回：
        (归一化后的 options, 归一化后��� correct_letter)。
    """
    first_key = next(iter(options))
    if first_key in _NUM_TO_LETTER:
        new_options = {_NUM_TO_LETTER[k]: v for k, v in options.items()}
        new_correct = _NUM_TO_LETTER.get(correct_letter, correct_letter)
        return new_options, new_correct
    return options, correct_letter


def build_cot_prompt(
    question: str,
    options: Dict[str, str],
    passage: Optional[str] = None,
) -> str:
    """构造 CoT 用户内容。

    不包含 /no_think 或 chat template 标记 — 这些由 evaluate_cot 处理。
    使用中性 prompt（无 "Reference:" 标签），passage 直接放在开头。
    末尾要求模型以 "The answer is X" 格式输出答案。

    参数：
        question: 题目文本。
        options: 选项字典。
        passage: 压缩后的知识段落，为 None 时构造 no_memory prompt。

    返回：
        格式化后的用户内容字符串。
    """
    labels = sorted(options.keys())
    option_lines = "\n".join(f"{lb}. {options[lb]}" for lb in labels)
    label_list = ", ".join(labels[:-1]) + ", or " + labels[-1]

    parts: list[str] = []
    if passage is not None:
        parts.append(passage)
    parts.append(f"\nQuestion: {question}")
    parts.append(option_lines)
    parts.append(
        f"\nLet's think step by step, then give the answer.\n"
        f'You MUST end your response with exactly "The answer is X" '
        f"where X is {label_list}."
    )
    return "\n".join(parts)


def extract_answer_letter(text: str, valid_labels: set) -> str:
    """从 CoT 生成文本中提取答案字母。

    按优先级依次尝试多种 pattern，返回第一个匹配的有效字母。
    无法提取时返回 "?"。

    参数：
        text: 模型生成的完整文本。
        valid_labels: 合法选项字母集合，如 {"A","B","C","D"}。

    返回：
        提取到的答案字母，或 "?"。
    """
    patterns = [
        r"[Tt]he answer is\s*([A-E])",
        r"[Aa]nswer\s*:\s*([A-E])",
        r"(?:option|choice)\s+([A-E])",
        r"\b([A-E])\s*\.?\s*$",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m and m.group(1) in valid_labels:
            return m.group(1)
    return "?"


def _tokenize_prompt(
    user_content: str,
    tokenizer: PreTrainedTokenizerBase,
    device: str = "cuda:0",
) -> torch.Tensor:
    """将用户内容编码为 input_ids，自动处理 chat template 和 thinking 开关。

    Qwen3 系列: apply_chat_template(enable_thinking=False) 注入空 think 块。
    其他模型: 有 chat_template 则使用，否则直接 tokenize。
    """
    has_chat = bool(getattr(tokenizer, "chat_template", None))
    if not has_chat:
        return tokenizer(user_content, return_tensors="pt").input_ids.to(device)

    kwargs: Dict = dict(
        tokenize=False,
        add_generation_prompt=True,
    )
    if _supports_thinking(tokenizer):
        kwargs["enable_thinking"] = False

    messages = [{"role": "user", "content": user_content}]
    text = tokenizer.apply_chat_template(messages, **kwargs)
    return tokenizer(text, return_tensors="pt").input_ids.to(device)


# ---------------------------------------------------------------------------
# CoT 生成评测
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_cot(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    valid_labels: set,
    device: str = "cuda:0",
    max_new_tokens: int = 2048,
) -> tuple:
    """CoT 生成评测。

    参数：
        model: CausalLM 模型。
        tokenizer: tokenizer。
        prompt: 用户内容（不含 chat template 标记）。
        valid_labels: 合法选项字母集合。
        device: 推理设备。
        max_new_tokens: 最大生成 token 数。

    返回：
        (answer_letter, gen_length, raw_output)
        answer_letter 为 "?" 表示提取失败。
    """
    ids = _tokenize_prompt(prompt, tokenizer, device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
    gen_tokens = out[0][ids.shape[-1] :]
    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    letter = extract_answer_letter(gen_text, valid_labels)
    return letter, len(gen_tokens), gen_text


@torch.no_grad()
def _batch_evaluate_cot(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    valid_labels_list: List[set],
    device: str = "cuda:0",
    max_new_tokens: int = 2048,
) -> List[tuple]:
    """批量 CoT 生成评测。

    参数：
        prompts: 用户内容列表。
        valid_labels_list: 每条样本的合法标签集列表。
        device: 推理设备。
        max_new_tokens: 最大生成 token 数。

    返回：
        [(answer_letter, gen_length, raw_output), ...] 与 prompts 等长。
    """
    has_chat = bool(getattr(tokenizer, "chat_template", None))
    use_thinking_off = _supports_thinking(tokenizer)

    texts: List[str] = []
    for content in prompts:
        if not has_chat:
            texts.append(content)
        else:
            kwargs: Dict = dict(tokenize=False, add_generation_prompt=True)
            if use_thinking_off:
                kwargs["enable_thinking"] = False
            messages = [{"role": "user", "content": content}]
            texts.append(tokenizer.apply_chat_template(messages, **kwargs))

    orig_pad_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    batch = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    tokenizer.padding_side = orig_pad_side

    outputs = model.generate(
        **batch,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    results: List[tuple] = []
    input_len = batch["input_ids"].shape[1]
    for i, out_ids in enumerate(outputs):
        gen_tokens = out_ids[input_len:]
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        letter = extract_answer_letter(gen_text, valid_labels_list[i])
        results.append((letter, len(gen_tokens), gen_text))
    return results


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------


def load_samples_jsonl(path: Path, n_samples: int = -1) -> List[Dict]:
    """从 JSONL 文件加载评测样本。

    参数：
        path: JSONL 文件路径。
        n_samples: 截取样本数，-1 表示全部加载。

    返回：
        样本字典列表。
    """
    samples: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    if n_samples > 0:
        samples = samples[:n_samples]
    return samples


# ---------------------------------------------------------------------------
# Git SHA 工具
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    """获取当前 git HEAD 的短 SHA。

    返回：
        7 位 short SHA 字符串；获取失败时返回 "unknown"。
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# 评测主流程
# ---------------------------------------------------------------------------


def run_evaluation(
    model_path: str,
    method: str,
    dataset: str,
    data_dir: str = "data/ood",
    output_dir: str = "results/baseline",
    n_samples: int = -1,
    device: str = "cuda:0",
    compress_target_token: int = 64,
    cot_max_new_tokens: int = 2048,
    batch_size: int = 1,
) -> Dict:
    """执行 baseline 评测的完整流程。

    参数：
        model_path: HuggingFace 模型路径。
        method: 评测方法，"no_memory" 或 "vanilla_rag"。
        dataset: 数据集名称。
        data_dir: 数据目录。
        output_dir: 结果输出目录。
        n_samples: 评测样本数，-1 表示全部。
        device: 推理设备。
        compress_target_token: LLMLingua-2 压缩目标 token 数。
        cot_max_new_tokens: CoT 生成最大 token 数。
        batch_size: 批量推理 batch size。

    返回：
        包含评测结果的字典。
    """
    # Phase 1: 加载模型
    logger.info("加载模型: %s", model_path)
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        has_text_config = hasattr(config, "text_config")
    except (KeyError, ValueError):
        has_text_config = False
    if has_text_config:
        import importlib

        arch_name = config.architectures[0]
        cls = getattr(importlib.import_module("transformers"), arch_name)
        model = (
            cls.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="sdpa",
            )
            .to(device)
            .eval()
        )
        logger.info("多模态模型 (%s)，使用完整模型进行 text-only 推理", arch_name)
    else:
        model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="sdpa",
            )
            .to(device)
            .eval()
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Phase 2: 加载数据
    data_path = Path(data_dir) / f"{dataset}.jsonl"
    logger.info("加载数据: %s", data_path)
    samples = load_samples_jsonl(data_path, n_samples=n_samples)
    logger.info("评测样本数: %d, batch_size: %d", len(samples), batch_size)

    # Phase 3: 预处理全部 prompt
    prompts: List[str] = []
    labels_list: List[List[str]] = []
    correct_letters: List[str] = []
    sample_ids: List[str] = []

    for sample in samples:
        options, cl = normalize_options(sample["options"], sample["correct_letter"])
        sorted_labels = sorted(options.keys())

        passage = None
        if method == "vanilla_rag":
            raw_passage = sample.get("passage", "")
            passage = compress_passage(raw_passage, target_token=compress_target_token)

        prompt = build_cot_prompt(
            question=sample["question"],
            options=options,
            passage=passage,
        )
        prompts.append(prompt)
        labels_list.append(sorted_labels)
        correct_letters.append(cl)
        sample_ids.append(sample.get("id", ""))

    # Phase 4: 批量推理 + 逐条 JSONL 输出
    correct = 0
    extract_success = 0
    latencies: List[float] = []
    gen_lengths: List[int] = []

    model_name = Path(model_path).name
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{model_name}_{method}_{dataset}.jsonl"

    n_batches = (len(prompts) + batch_size - 1) // batch_size
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for batch_start in tqdm(
            range(0, len(prompts), batch_size),
            desc=f"{method}/{dataset}",
            total=n_batches,
        ):
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]
            batch_valid = [set(ll) for ll in labels_list[batch_start:batch_end]]

            t0 = time.perf_counter()
            if len(batch_prompts) == 1:
                batch_results = [
                    evaluate_cot(
                        model,
                        tokenizer,
                        batch_prompts[0],
                        batch_valid[0],
                        device=device,
                        max_new_tokens=cot_max_new_tokens,
                    )
                ]
            else:
                batch_results = _batch_evaluate_cot(
                    model,
                    tokenizer,
                    batch_prompts,
                    batch_valid,
                    device=device,
                    max_new_tokens=cot_max_new_tokens,
                )
            elapsed = (time.perf_counter() - t0) * 1000

            for j, (pred, gen_len, raw_output) in enumerate(batch_results):
                idx = batch_start + j
                per_sample_ms = elapsed / len(batch_results)
                latencies.append(per_sample_ms)
                gen_lengths.append(gen_len)

                if pred != "?":
                    extract_success += 1
                if pred == correct_letters[idx]:
                    correct += 1

                record = {
                    "id": sample_ids[idx],
                    "pred": pred,
                    "correct": correct_letters[idx],
                    "gen_length": gen_len,
                    "raw_output": raw_output,
                }
                jf.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Phase 5: 汇总结果
    n_total = len(samples)
    accuracy = correct / n_total if n_total > 0 else 0.0
    latency_ms_mean = sum(latencies) / len(latencies) if latencies else 0.0
    avg_gen_len = sum(gen_lengths) / len(gen_lengths) if gen_lengths else 0.0

    result = {
        "model": model_name,
        "model_path": model_path,
        "method": method,
        "dataset": dataset,
        "scoring": "cot_nothink",
        "compress_target_token": compress_target_token
        if method == "vanilla_rag"
        else None,
        "cot_max_new_tokens": cot_max_new_tokens,
        "batch_size": batch_size,
        "n_samples": n_total,
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "extract_success": extract_success,
        "extract_success_rate": round(extract_success / n_total, 4)
        if n_total > 0
        else 0.0,
        "avg_gen_length": round(avg_gen_len, 1),
        "latency_ms_mean": round(latency_ms_mean, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
    }

    out_path = out_dir / f"{model_name}_{method}_{dataset}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(
        "结果已保存: %s (accuracy=%.4f, avg_gen=%.0f)",
        out_path,
        accuracy,
        avg_gen_len,
    )
    return result


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------


def main() -> None:
    """命令行入口，解析参数并启动评测。"""
    parser = argparse.ArgumentParser(description="No-Memory / VanillaRAG baseline 评测")
    parser.add_argument(
        "--model-path", type=str, required=True, help="HuggingFace 模型路径"
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["no_memory", "vanilla_rag"],
        help="评测方法",
    )
    parser.add_argument("--dataset", type=str, required=True, help="数据集名称")
    parser.add_argument("--data-dir", type=str, default="data/ood", help="数据目录")
    parser.add_argument(
        "--output-dir", type=str, default="results/baseline", help="结果输出目录"
    )
    parser.add_argument(
        "--n-samples", type=int, default=-1, help="评测样本数，-1 表示全部"
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="推理设备")
    parser.add_argument(
        "--compress-target-token",
        type=int,
        default=64,
        help="LLMLingua-2 压缩目标 token 数 (默认: 64)",
    )
    parser.add_argument(
        "--cot-max-new-tokens",
        type=int,
        default=2048,
        help="CoT 生成最大 token 数 (默认: 2048)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="批量推理 batch size (默认: 1)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run_evaluation(
        model_path=args.model_path,
        method=args.method,
        dataset=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        device=args.device,
        compress_target_token=args.compress_target_token,
        cot_max_new_tokens=args.cot_max_new_tokens,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
