"""eval_tokenmem — TokenMem 模型评测脚本。

使用训练好的 TokenMemForCausalLM（frozen base LLM + trainable gates）
对 OOD 多选题数据集进行 loglikelihood 打分评测。

与 eval_baseline.py 的核心区别：
1. 模型类：TokenMemForCausalLM + load_gates() 加载训练好的 gate 权重
2. 知识注入：passage 通过 knowledge_input_ids cross-attention 注入（而非放入 prompt 文本）
3. prompt 格式：与 no_memory 相同（无 "Reference:" 前缀）
4. forward 接口：model(input_ids=..., knowledge_input_ids=..., knowledge_attention_mask=...)
5. 结果目录：results/tokenmem/

使用示例：
    python -m evaluation.eval_tokenmem \\
        --model-path hugglingface_model/qwen3-0.6B \\
        --gate-dir checkpoints/qwen3-0.6B/gates \\
        --dataset medqa \\
        --data-dir data/ood \\
        --output-dir results/tokenmem
"""

import argparse
import json
import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from memory_lora.tokenmem_model import TokenMemForCausalLM

logger = logging.getLogger(__name__)

_NUM_TO_LETTER = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}


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
        (归一化后的 options, 归一化后的 correct_letter)。
    """
    first_key = next(iter(options))
    if first_key in _NUM_TO_LETTER:
        new_options = {_NUM_TO_LETTER[k]: v for k, v in options.items()}
        new_correct = _NUM_TO_LETTER.get(correct_letter, correct_letter)
        return new_options, new_correct
    return options, correct_letter


def build_mc_prompt(
    question: str,
    options: Dict[str, str],
) -> str:
    """构造多选题 prompt（TokenMem 版，不接受 passage 参数）。

    passage 通过 cross-attention 注入，prompt 文本中不含 "Reference:" 前缀。
    格式与 eval_baseline.py 的 no_memory 模式完全相同。

    参数：
        question: 问题文本。
        options: 选项字典，键为字母（A/B/C/...），值为选项文本。

    返回：
        格式化后的 prompt 字符串，以 "Answer:" 结尾。
    """
    labels = sorted(options.keys())
    option_lines = "\n".join(f"{lb}. {options[lb]}" for lb in labels)
    return f"Question: {question}\n{option_lines}\nAnswer:"


# ---------------------------------------------------------------------------
# 知识 Tokenize
# ---------------------------------------------------------------------------


def tokenize_knowledge(
    tokenizer: PreTrainedTokenizerBase,
    passage: str,
    max_len: int = 256,
    device: str = "cuda:0",
) -> Dict[str, torch.Tensor]:
    """将 passage 文本 tokenize 为 knowledge_input_ids + knowledge_attention_mask。

    对 passage 进行右截断，返回可直接传入 TokenMemForCausalLM.forward() 的张量字典。

    参数：
        tokenizer: 对应模型的 tokenizer。
        passage: 知识段落文本。
        max_len: 最大 token 长度，超出时截断（默认 256）。
        device: 目标设备。

    返回：
        包含以下键的字典：
        - "knowledge_input_ids": shape [1, L]，L <= max_len
        - "knowledge_attention_mask": shape [1, L]，L <= max_len
    """
    encoded = tokenizer(
        passage,
        max_length=max_len,
        truncation=True,
        padding=False,
        return_tensors="pt",
        add_special_tokens=False,
    )
    knowledge_input_ids: torch.Tensor = encoded["input_ids"].to(device)
    knowledge_attention_mask: torch.Tensor = encoded["attention_mask"].to(device)
    return {
        "knowledge_input_ids": knowledge_input_ids,
        "knowledge_attention_mask": knowledge_attention_mask,
    }


# ---------------------------------------------------------------------------
# Logprob 评分
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_logprob(
    model: TokenMemForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    labels: List[str],
    knowledge_input_ids: torch.Tensor,
    knowledge_attention_mask: torch.Tensor,
    device: str = "cuda:0",
) -> int:
    """对多选 prompt 进行 loglikelihood 打分，返回最优选项索引。

    与 eval_baseline 相同的 logprob 打分逻辑，但 forward 时传入 knowledge 张量。

    参数：
        model: TokenMemForCausalLM 模型（已加载 gate 权重）。
        tokenizer: 对应的 tokenizer。
        prompt: 已格式化的多选 prompt（以 "Answer:" 结尾）。
        labels: 选项标签列表，如 ["A","B","C","D"] 或 ["A","B","C"]。
        knowledge_input_ids: knowledge token ids，shape [1, K]。
        knowledge_attention_mask: knowledge attention mask，shape [1, K]。
        device: 推理设备。

    返回：
        最优选项索引 ∈ [0, len(labels)-1]。
    """
    context_ids: List[int] = tokenizer.encode(prompt, add_special_tokens=False)

    scores: List[float] = []
    for label in labels:
        letter = " " + label
        cont_ids: List[int] = tokenizer.encode(letter, add_special_tokens=False)
        full_ids_list = context_ids + cont_ids
        full_ids = torch.tensor([full_ids_list], dtype=torch.long, device=device)

        outputs = model(
            input_ids=full_ids,
            knowledge_input_ids=knowledge_input_ids,
            knowledge_attention_mask=knowledge_attention_mask,
        )
        logits = outputs.logits

        cont_start = len(context_ids) - 1
        cont_end = len(full_ids_list) - 1
        cont_logits = logits[0, cont_start:cont_end, :]
        cont_tokens = torch.tensor(cont_ids, dtype=torch.long, device=device)

        log_probs = F.log_softmax(cont_logits.float(), dim=-1)
        token_ll = log_probs.gather(1, cont_tokens.unsqueeze(-1)).squeeze(-1)
        scores.append(token_ll.sum().item())

    return int(torch.tensor(scores).argmax().item())


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
    gate_dir: str,
    dataset: str,
    data_dir: str = "data/ood",
    output_dir: str = "results/tokenmem",
    n_samples: int = -1,
    knowledge_max_len: int = 256,
    device: str = "cuda:0",
) -> Dict:
    """执行 TokenMem 评测的完整流程。

    加载训练好的 TokenMemForCausalLM（base model + gate 权重），对每条样本
    将 passage 通过 cross-attention 注入，对问题选项进行 loglikelihood 打分。

    参数：
        model_path: HuggingFace 模型路径（本地）。
        gate_dir: 训练好的 gate 权重目录（包含 gate_*.pt 文件）。
        dataset: 数据集名称（对应 data_dir 下的 {dataset}.jsonl）。
        data_dir: 数据目录。
        output_dir: 结果输出目录。
        n_samples: 评测样本数，-1 表示全部。
        knowledge_max_len: passage tokenize 的最大长度。
        device: 推理设备。

    返回：
        包含评测结果的字典（accuracy, correct, latency 等）。
    """
    # Phase 1: 加载模型
    logger.info("加载 TokenMemForCausalLM: %s", model_path)
    model = TokenMemForCausalLM(
        model_name_or_path=model_path,
        knowledge_max_seq_len=64,
        torch_dtype=torch.bfloat16,
    )
    logger.info("加载 gate 权重: %s", gate_dir)
    model.load_gates(gate_dir)
    model.to(device).eval()

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
    latencies: List[float] = []

    for sample in tqdm(samples, desc=f"tokenmem/{dataset}"):
        options, correct_letter = normalize_options(
            sample["options"], sample["correct_letter"]
        )
        labels = sorted(options.keys())
        label_to_idx = {lb: i for i, lb in enumerate(labels)}

        prompt = build_mc_prompt(
            question=sample["question"],
            options=options,
        )

        # Phase 3.1: tokenize passage 为 knowledge 张量
        passage: str = sample.get("passage", "")
        knowledge_tensors = tokenize_knowledge(
            tokenizer=tokenizer,
            passage=passage,
            max_len=knowledge_max_len,
            device=device,
        )

        t0 = time.perf_counter()
        pred_idx = evaluate_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            labels=labels,
            knowledge_input_ids=knowledge_tensors["knowledge_input_ids"],
            knowledge_attention_mask=knowledge_tensors["knowledge_attention_mask"],
            device=device,
        )
        latencies.append((time.perf_counter() - t0) * 1000)

        gold_idx = label_to_idx[correct_letter]
        if pred_idx == gold_idx:
            correct += 1

    # Phase 4: 汇总结果
    n_total = len(samples)
    accuracy = correct / n_total if n_total > 0 else 0.0
    latency_ms_mean = sum(latencies) / len(latencies) if latencies else 0.0

    model_name = Path(model_path).name
    result = {
        "model": model_name,
        "model_path": model_path,
        "method": "tokenmem",
        "gate_dir": gate_dir,
        "dataset": dataset,
        "n_samples": n_total,
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "knowledge_max_len": knowledge_max_len,
        "latency_ms_mean": round(latency_ms_mean, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
    }

    # Phase 5: 保存结果
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_tokenmem_{dataset}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("结果已保存: %s (accuracy=%.4f)", out_path, accuracy)

    return result


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------


def main() -> None:
    """命令行入口，解析参数并启动 TokenMem 评测。"""
    parser = argparse.ArgumentParser(description="TokenMem 模型评测")
    parser.add_argument(
        "--model-path", type=str, required=True, help="HuggingFace 模型路径（本地）"
    )
    parser.add_argument(
        "--gate-dir", type=str, required=True, help="训练好的 gate 权重目录"
    )
    parser.add_argument("--dataset", type=str, required=True, help="数据集名称")
    parser.add_argument("--data-dir", type=str, default="data/ood", help="数据目录")
    parser.add_argument(
        "--output-dir", type=str, default="results/tokenmem", help="结果输出目录"
    )
    parser.add_argument(
        "--n-samples", type=int, default=-1, help="评测样本数，-1 表示全部"
    )
    parser.add_argument(
        "--knowledge-max-len", type=int, default=256, help="passage tokenize 最大长度"
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="推理设备")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run_evaluation(
        model_path=args.model_path,
        gate_dir=args.gate_dir,
        dataset=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        knowledge_max_len=args.knowledge_max_len,
        device=args.device,
    )


if __name__ == "__main__":
    main()
