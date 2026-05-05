"""eval_rag_sft — RAG SFT 评测脚本。

加载 base model + LoRA adapter (PEFT)，将 256-token 知识直接放入 prompt，
使用 CoT + nothink 生成 → regex 提取答案字母。
不使用 LLMLingua-2 压缩，passage 仅按 token 数截断。

使用：
    python -m evaluation.eval_rag_sft \\
        --model-path hugglingface_model/qwen3-8B \\
        --lora-dir checkpoints/qwen3-8b_rag_sft_p2/best \\
        --dataset cf_arc_easy_val \\
        --data-dir data/counterfactual \\
        --output-dir results/rag_sft \\
        --knowledge-max-len 256 \\
        --cot-max-new-tokens 2048 \\
        --batch-size 4
"""

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from evaluation.eval_baseline import (
    _batch_evaluate_cot,
    build_cot_prompt,
    evaluate_cot,
    extract_answer_letter,
    load_samples_jsonl,
    normalize_options,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 知识截断
# ---------------------------------------------------------------------------


def truncate_passage_by_tokens(
    passage: str,
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int = 256,
) -> str:
    """将 passage 按 token 数截断到 max_tokens。

    参数：
        passage: 原始知识段落。
        tokenizer: 用于编码的 tokenizer。
        max_tokens: 最大允许 token 数。

    返回：
        截断后的文本字符串。若原文 token 数 <= max_tokens 则原样返回。
    """
    token_ids = tokenizer.encode(passage, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return passage
    truncated_ids = token_ids[:max_tokens]
    return tokenizer.decode(truncated_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Git SHA 工具
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    """获取当前 git HEAD 的短 SHA。

    返回：
        7 位 short SHA 字符串；获取失败时返回 "unknown"。
    """
    import subprocess

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
    lora_dir: str,
    dataset: str,
    data_dir: str = "data/counterfactual",
    output_dir: str = "results/rag_sft",
    n_samples: int = -1,
    device: str = "cuda:0",
    knowledge_max_len: int = 256,
    cot_max_new_tokens: int = 2048,
    batch_size: int = 4,
) -> Dict:
    """执行 RAG SFT 评测的完整流程。

    加载 base model + LoRA adapter，merge_and_unload 后进行推理。
    passage 按 token 数截断到 knowledge_max_len，直接放入 prompt。

    参数：
        model_path: HuggingFace base 模型路径。
        lora_dir: LoRA adapter 目录路径。
        dataset: 数据集名称。
        data_dir: 数据目录。
        output_dir: 结果输出目录。
        n_samples: 评测样本数，-1 表示全部。
        device: 推理设备。
        knowledge_max_len: knowledge 截断最大 token 数。
        cot_max_new_tokens: CoT 生成最大 token 数。
        batch_size: 批量推理 batch size。

    返回：
        包含评测结果的字典。
    """
    # Phase 1: 加载 base model + LoRA
    logger.info("加载 base model: %s", model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    logger.info("加载 LoRA adapter: %s", lora_dir)
    model = PeftModel.from_pretrained(base_model, lora_dir)
    model = model.merge_and_unload()
    model = model.to(device).eval()
    logger.info("LoRA merge_and_unload 完成，模型已加载到 %s", device)

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

        # 获取知识段落并按 token 截断
        raw_passage = sample.get("passage", sample.get("counterfactual_passage", ""))
        passage = truncate_passage_by_tokens(
            raw_passage, tokenizer, max_tokens=knowledge_max_len
        )

        prompt = build_cot_prompt(
            question=sample["question"],
            options=options,
            passage=passage if passage else None,
        )
        prompts.append(prompt)
        labels_list.append(sorted_labels)
        correct_letters.append(cl)
        sample_ids.append(sample.get("id", ""))

    # Phase 4: 批量推理 + 逐条 JSONL 输出
    correct = 0
    extract_success = 0
    gen_lengths: List[int] = []

    model_name = Path(model_path).name
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{model_name}_rag_sft_{dataset}.jsonl"

    n_batches = (len(prompts) + batch_size - 1) // batch_size
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for batch_start in tqdm(
            range(0, len(prompts), batch_size),
            desc=f"rag_sft/{dataset}",
            total=n_batches,
        ):
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]
            batch_valid = [set(ll) for ll in labels_list[batch_start:batch_end]]

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

            for j, (pred, gen_len, raw_output) in enumerate(batch_results):
                idx = batch_start + j
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
    avg_gen_len = sum(gen_lengths) / len(gen_lengths) if gen_lengths else 0.0

    result = {
        "model": model_name,
        "method": "rag_sft",
        "dataset": dataset,
        "n_samples": n_total,
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "extract_success_rate": round(extract_success / n_total, 4)
        if n_total > 0
        else 0.0,
        "avg_gen_length": round(avg_gen_len, 1),
        "lora_dir": lora_dir,
        "knowledge_max_len": knowledge_max_len,
        "cot_max_new_tokens": cot_max_new_tokens,
        "batch_size": batch_size,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
    }

    out_path = out_dir / f"{model_name}_rag_sft_{dataset}.json"
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


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    返回：
        解析后的参数命名空间。
    """
    parser = argparse.ArgumentParser(
        description="RAG SFT 评测 (LoRA + in-context knowledge)"
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="HuggingFace base 模型路径"
    )
    parser.add_argument(
        "--lora-dir", type=str, required=True, help="LoRA adapter 目录路径"
    )
    parser.add_argument("--dataset", type=str, required=True, help="数据集名称")
    parser.add_argument(
        "--data-dir", type=str, default="data/counterfactual", help="数据目录"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/rag_sft", help="结果输出目录"
    )
    parser.add_argument(
        "--n-samples", type=int, default=-1, help="评测样本数，-1 表示全部"
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="推理设备")
    parser.add_argument(
        "--knowledge-max-len",
        type=int,
        default=256,
        help="knowledge 截断最大 token 数 (默认: 256)",
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
        default=4,
        help="批量推理 batch size (默认: 4)",
    )
    return parser.parse_args()


def main() -> None:
    """命令行入口，解析参数并启动 RAG SFT 评测。"""
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run_evaluation(
        model_path=args.model_path,
        lora_dir=args.lora_dir,
        dataset=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        device=args.device,
        knowledge_max_len=args.knowledge_max_len,
        cot_max_new_tokens=args.cot_max_new_tokens,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
