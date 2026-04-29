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
        --cot-max-new-tokens 1024
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
    """构造 CoT + nothink 格式的 prompt。

    使用中性 prompt（无 "Reference:" 标签），passage 直接放在开头。
    末尾要求模型以 "The answer is X" 格式输出答案。

    参数：
        question: 题目文本。
        options: 选项字典。
        passage: 压缩后的知识段落，为 None 时构造 no_memory prompt。

    返回：
        格式化后的 CoT prompt 字符串。
    """
    labels = sorted(options.keys())
    option_lines = "\n".join(f"{lb}. {options[lb]}" for lb in labels)
    label_list = ", ".join(labels[:-1]) + ", or " + labels[-1]

    parts = ["/no_think"]
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
    max_new_tokens: int = 1024,
) -> tuple:
    """CoT 生成评测，返回 (提取的字母, 生成文本 token 数)。

    参数：
        model: CausalLM 模型。
        tokenizer: tokenizer。
        prompt: CoT 格式的 prompt。
        valid_labels: 合法选项字母集合。
        device: 推理设备。
        max_new_tokens: 最大生成 token 数。

    返回：
        (answer_letter, gen_length) — answer_letter 为 "?" 表示提取失败。
    """
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
    gen_tokens = out[0][ids.shape[-1] :]
    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    letter = extract_answer_letter(gen_text, valid_labels)
    return letter, len(gen_tokens)


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
    cot_max_new_tokens: int = 1024,
) -> Dict:
    """执行 baseline 评测的完整流程。

    参数：
        model_path: HuggingFace 模型路径（本地或远程）。
        method: 评测方法，"no_memory" 或 "vanilla_rag"。
        dataset: 数据集名称（对应 data_dir 下的 {dataset}.jsonl）。
        data_dir: 数据目录。
        output_dir: 结果输出目录。
        n_samples: 评测样本数，-1 表示全部。
        device: 推理设备。
        compress_target_token: LLMLingua-2 压缩目标 token 数。
        cot_max_new_tokens: CoT 生成最大 token 数。

    返回：
        包含评测结果的字典（accuracy, correct, latency 等）。
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
    logger.info("评测样本数: %d", len(samples))

    # Phase 3: 逐条评测
    correct = 0
    extract_success = 0
    latencies: List[float] = []
    gen_lengths: List[int] = []

    for sample in tqdm(samples, desc=f"{method}/{dataset}"):
        options, correct_letter = normalize_options(
            sample["options"], sample["correct_letter"]
        )
        labels = sorted(options.keys())

        passage = None
        if method == "vanilla_rag":
            raw_passage = sample.get("passage", "")
            passage = compress_passage(raw_passage, target_token=compress_target_token)

        prompt = build_cot_prompt(
            question=sample["question"],
            options=options,
            passage=passage,
        )

        t0 = time.perf_counter()
        pred_letter, gen_len = evaluate_cot(
            model,
            tokenizer,
            prompt,
            valid_labels=set(labels),
            device=device,
            max_new_tokens=cot_max_new_tokens,
        )
        latencies.append((time.perf_counter() - t0) * 1000)
        gen_lengths.append(gen_len)

        if pred_letter != "?":
            extract_success += 1
        if pred_letter == correct_letter:
            correct += 1

    # Phase 4: 汇总结果
    n_total = len(samples)
    accuracy = correct / n_total if n_total > 0 else 0.0
    latency_ms_mean = sum(latencies) / len(latencies) if latencies else 0.0
    avg_gen_len = sum(gen_lengths) / len(gen_lengths) if gen_lengths else 0.0

    model_name = Path(model_path).name
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

    # Phase 5: 保存结果
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_{method}_{dataset}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("结果已保存: %s (accuracy=%.4f)", out_path, accuracy)

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
        default=1024,
        help="CoT 生成最大 token 数 (默认: 1024)",
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
    )


if __name__ == "__main__":
    main()
