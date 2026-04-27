"""eval_baseline — No-Memory / VanillaRAG baseline 评测。

统一入口，通过 --method 切换评测模式。使用 loglikelihood 打分
（对 " A"/" B"/" C"/" D" 的 continuation log-prob 累加取 argmax）。

使用：
    python -m evaluation.eval_baseline \
        --model-path hugglingface_model/qwen3-0.6B \
        --method no_memory \
        --dataset medqa \
        --data-dir data/ood \
        --output-dir results/baseline

    python -m evaluation.eval_baseline \
        --model-path hugglingface_model/qwen3-0.6B \
        --method vanilla_rag \
        --dataset medqa \
        --data-dir data/ood \
        --output-dir results/baseline \
        --n-samples 10
"""

import argparse
import json
import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

_CHOICE_LABELS = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# Prompt 构造
# ---------------------------------------------------------------------------


def build_mc_prompt(
    question: str,
    options: Dict[str, str],
    passage: Optional[str] = None,
) -> str:
    """构造多选题 prompt。

    格式对齐 training/data.py 第 69-76 行的 SFT 数据格式。

    参数：
        question: 问题文本。
        options: 选项字典，键为 "A","B","C","D"，值为选项文本。
        passage: 参考段落。为 None 时构造 no_memory prompt；
                 否则在开头添加 "Reference: ..." 构造 vanilla_rag prompt。

    返回：
        格式化后的 prompt 字符串，以 "Answer:" 结尾。
    """
    body = (
        f"Question: {question}\n"
        f"A. {options['A']}\n"
        f"B. {options['B']}\n"
        f"C. {options['C']}\n"
        f"D. {options['D']}\n"
        f"Answer:"
    )
    if passage is not None:
        return f"Reference: {passage}\n\n{body}"
    return body


# ---------------------------------------------------------------------------
# Logprob 评分
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_logprob(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    device: str = "cuda:0",
) -> int:
    """对多选 prompt 进行 loglikelihood 打分，返回最优选项索引。

    对齐 Reference/Memory-LoRA-old/memory_lora/pipeline.py:263-356 的评分协议：
        1. context_ids = tokenizer.encode(prompt, add_special_tokens=False)
        2. 对每个 letter ∈ [" A"," B"," C"," D"]：
           cont_ids = tokenizer.encode(letter, add_special_tokens=False)
           full_ids = context_ids + cont_ids
           forward → 取 continuation 位置 logits → log_softmax → gather → sum
        3. return argmax(scores)

    参数：
        model: HuggingFace CausalLM 模型。
        tokenizer: 对应的 tokenizer。
        prompt: 已格式化的多选 prompt（以 "Answer:" 结尾）。
        device: 推理设备，如 "cuda:0" 或 "cpu"。

    返回：
        最优选项索引 ∈ [0, 3]（0=A, 1=B, 2=C, 3=D）。
    """
    context_ids: List[int] = tokenizer.encode(prompt, add_special_tokens=False)

    scores: List[float] = []
    for label in _CHOICE_LABELS:
        letter = " " + label
        cont_ids: List[int] = tokenizer.encode(letter, add_special_tokens=False)
        full_ids_list = context_ids + cont_ids
        full_ids = torch.tensor([full_ids_list], dtype=torch.long, device=device)

        outputs = model(input_ids=full_ids)
        logits = outputs.logits  # [1, L_full, V]

        # 取 continuation 位置的 logits（预测下一个 token，故 -1 shift）
        cont_start = len(context_ids) - 1
        cont_end = len(full_ids_list) - 1
        cont_logits = logits[0, cont_start:cont_end, :]  # [len(cont_ids), V]
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
    method: str,
    dataset: str,
    data_dir: str = "data/ood",
    output_dir: str = "results/baseline",
    n_samples: int = -1,
    device: str = "cuda:0",
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

    返回：
        包含评测结果的字典（accuracy, correct, latency 等）。
    """
    # Phase 1: 加载模型
    logger.info("加载模型: %s", model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device).eval()
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
    label_to_idx = {label: i for i, label in enumerate(_CHOICE_LABELS)}

    for sample in tqdm(samples, desc=f"{method}/{dataset}"):
        passage = sample.get("passage") if method == "vanilla_rag" else None
        prompt = build_mc_prompt(
            question=sample["question"],
            options=sample["options"],
            passage=passage,
        )

        t0 = time.perf_counter()
        pred_idx = evaluate_logprob(model, tokenizer, prompt, device=device)
        latencies.append((time.perf_counter() - t0) * 1000)

        gold_idx = label_to_idx[sample["correct_letter"]]
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
        "method": method,
        "dataset": dataset,
        "n_samples": n_total,
        "accuracy": round(accuracy, 4),
        "correct": correct,
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
    parser = argparse.ArgumentParser(
        description="No-Memory / VanillaRAG baseline 评测"
    )
    parser.add_argument("--model-path", type=str, required=True, help="HuggingFace 模型路径")
    parser.add_argument("--method", type=str, required=True, choices=["no_memory", "vanilla_rag"], help="评测方法")
    parser.add_argument("--dataset", type=str, required=True, help="数据集名称")
    parser.add_argument("--data-dir", type=str, default="data/ood", help="数据目录")
    parser.add_argument("--output-dir", type=str, default="results/baseline", help="结果输出目录")
    parser.add_argument("--n-samples", type=int, default=-1, help="评测样本数，-1 表示全部")
    parser.add_argument("--device", type=str, default="cuda:0", help="推理设备")
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
    )


if __name__ == "__main__":
    main()
