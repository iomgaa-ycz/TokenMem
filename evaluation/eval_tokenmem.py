"""eval_tokenmem — TokenMem 模型 CoT 评测脚本。

使用训练好的 TokenMemForCausalLM（frozen base LLM + trainable gates）
对多选题数据集进行 CoT 生成评测。

与 eval_baseline.py 的核心区别：
1. 模型类：TokenMemForCausalLM + load_gates() 加载训练好的 gate 权重
2. 知识注入：passage 通过 knowledge_outputs cross-attention 注入（非 prompt 拼接）
3. prompt 格式：build_cot_prompt(passage=None)，无 passage 文本
4. generate 接口：model.model.generate(knowledge_outputs=...) 透传到 forward
5. 结果目录：results/tokenmem/

使用示例：
    python -m evaluation.eval_tokenmem \\
        --model-path hugglingface_model/qwen3-4B \\
        --gate-dir checkpoints/qwen3-4b_sft_p2/best \\
        --dataset medqa \\
        --data-dir data/ood \\
        --output-dir results/tokenmem \\
        --cot-max-new-tokens 2048 \\
        --batch-size 8
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
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from evaluation.eval_baseline import (
    build_cot_prompt,
    extract_answer_letter,
    _supports_thinking,
    normalize_options,
)
from memory_lora.knowledge_encoder import compute_knowledge_hidden_states
from memory_lora.tokenmem_model import TokenMemForCausalLM

logger = logging.getLogger(__name__)


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
# Prompt Tokenize
# ---------------------------------------------------------------------------


def _tokenize_prompt(
    user_content: str,
    tokenizer: PreTrainedTokenizerBase,
    device: str = "cuda:0",
) -> torch.Tensor:
    """将用户内容编码为 input_ids，自动处理 chat template 和 thinking 开关。

    Qwen3 系列: apply_chat_template(enable_thinking=False) 注入空 think 块。
    其他模型: 有 chat_template 则使用，否则直接 tokenize。

    参数：
        user_content: 用户问题文本。
        tokenizer: 对应模型的 tokenizer。
        device: 目标设备。

    返回：
        input_ids tensor [1, L]。
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
# CoT 批量生成评测
# ---------------------------------------------------------------------------


@torch.no_grad()
def batch_evaluate_cot(
    model: TokenMemForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    valid_labels_list: List[set],
    knowledge_outputs: List[torch.Tensor],
    device: str = "cuda:0",
    max_new_tokens: int = 2048,
) -> List[tuple]:
    """批量 CoT 生成评测（TokenMem 版）。

    与 eval_baseline._batch_evaluate_cot 的核心区别：
    1. 调用 model.model.generate()（内核模型，非外壳）
    2. 传入 knowledge_outputs 通过 kwargs 透传到 forward()

    参数：
        model: TokenMemForCausalLM 模型（已加载 gate 权重）。
        tokenizer: 对应的 tokenizer。
        prompts: 用户内容列表（不含 passage，知识通过 cross-attention 注入）。
        valid_labels_list: 每条样本的合法标签集列表。
        knowledge_outputs: 预计算的逐层知识 hidden states，List[Tensor] × num_layers，
            每个 Tensor shape [B, knowledge_max_seq_len, hidden_dim]。
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
            tk: Dict = dict(tokenize=False, add_generation_prompt=True)
            if use_thinking_off:
                tk["enable_thinking"] = False
            messages = [{"role": "user", "content": content}]
            texts.append(tokenizer.apply_chat_template(messages, **tk))

    orig_pad_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    batch = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    tokenizer.padding_side = orig_pad_side

    outputs = model.model.generate(
        **batch,
        knowledge_outputs=knowledge_outputs,
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
    gate_dir: str,
    dataset: str,
    data_dir: str = "data/ood",
    output_dir: str = "results/tokenmem",
    n_samples: int = -1,
    knowledge_max_len: int = 256,
    cot_max_new_tokens: int = 2048,
    batch_size: int = 8,
    device: str = "cuda:0",
) -> Dict:
    """执行 TokenMem CoT 评测的完整流程。

    加载训练好的 TokenMemForCausalLM（base model + gate 权重），对每批样本
    将 passage 通过 cross-attention 注入，使用 CoT 生成并提取答案。

    参数：
        model_path: HuggingFace 模型路径（本地）。
        gate_dir: 训练好的 gate 权重目录（包含 gate_*.pt 文件）。
        dataset: 数据集名称（对应 data_dir 下的 {dataset}.jsonl）。
        data_dir: 数据目录。
        output_dir: 结果输出目录。
        n_samples: 评测样本数，-1 表示全部。
        knowledge_max_len: passage tokenize 的最大长度。
        cot_max_new_tokens: CoT 生成最大 token 数。
        batch_size: 批量推理 batch size。
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
    logger.info("评测样本数: %d, batch_size: %d", len(samples), batch_size)

    # Phase 3: 批量推理 + 逐条 JSONL 输出
    correct = 0
    extract_success = 0
    latencies: List[float] = []
    gen_lengths: List[int] = []

    model_name = Path(model_path).name
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{model_name}_tokenmem_{dataset}.jsonl"

    n_batches = (len(samples) + batch_size - 1) // batch_size
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for batch_start in tqdm(
            range(0, len(samples), batch_size),
            desc=f"tokenmem/{dataset}",
            total=n_batches,
        ):
            batch_end = min(batch_start + batch_size, len(samples))
            batch_samples = samples[batch_start:batch_end]
            cur_batch_size = len(batch_samples)

            # 3.1: 预处理 prompt + 选项
            prompts: List[str] = []
            labels_list: List[List[str]] = []
            correct_letters: List[str] = []
            sample_ids: List[str] = []

            for sample in batch_samples:
                options, cl = normalize_options(
                    sample["options"], sample["correct_letter"]
                )
                sorted_labels = sorted(options.keys())
                prompt = build_cot_prompt(
                    question=sample["question"],
                    options=options,
                    passage=None,
                )
                prompts.append(prompt)
                labels_list.append(sorted_labels)
                correct_letters.append(cl)
                sample_ids.append(sample.get("id", ""))

            # 3.2: 批量编码 knowledge
            passages = [s.get("passage", "") for s in batch_samples]
            batch_k_ids_list = []
            batch_k_mask_list = []
            for passage in passages:
                kt = tokenize_knowledge(
                    tokenizer=tokenizer,
                    passage=passage,
                    max_len=knowledge_max_len,
                    device=device,
                )
                batch_k_ids_list.append(kt["knowledge_input_ids"])
                batch_k_mask_list.append(kt["knowledge_attention_mask"])

            max_k_len = max(t.shape[1] for t in batch_k_ids_list)
            padded_ids = []
            padded_masks = []
            for kid, kmask in zip(batch_k_ids_list, batch_k_mask_list):
                pad_len = max_k_len - kid.shape[1]
                if pad_len > 0:
                    kid = torch.nn.functional.pad(
                        kid, (0, pad_len), value=tokenizer.pad_token_id
                    )
                    kmask = torch.nn.functional.pad(kmask, (0, pad_len), value=0)
                padded_ids.append(kid)
                padded_masks.append(kmask)

            batch_k_ids = torch.cat(padded_ids, dim=0)
            batch_k_mask = torch.cat(padded_masks, dim=0)

            knowledge_outputs = compute_knowledge_hidden_states(
                model.model,
                knowledge_input_ids=batch_k_ids,
                knowledge_attention_mask=batch_k_mask,
                knowledge_max_seq_len=model.knowledge_max_seq_len,
            )

            # 3.3: 批量 CoT 生成
            batch_valid = [set(ll) for ll in labels_list]

            t0 = time.perf_counter()
            batch_results = batch_evaluate_cot(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                valid_labels_list=batch_valid,
                knowledge_outputs=knowledge_outputs,
                device=device,
                max_new_tokens=cot_max_new_tokens,
            )
            elapsed = (time.perf_counter() - t0) * 1000

            # 3.4: 记录结果
            for j, (pred, gen_len, raw_output) in enumerate(batch_results):
                per_sample_ms = elapsed / cur_batch_size
                latencies.append(per_sample_ms)
                gen_lengths.append(gen_len)

                if pred != "?":
                    extract_success += 1
                if pred == correct_letters[j]:
                    correct += 1

                record = {
                    "id": sample_ids[j],
                    "pred": pred,
                    "correct": correct_letters[j],
                    "gen_length": gen_len,
                    "raw_output": raw_output,
                }
                jf.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Phase 4: 汇总结果
    n_total = len(samples)
    accuracy = correct / n_total if n_total > 0 else 0.0
    latency_ms_mean = sum(latencies) / len(latencies) if latencies else 0.0
    avg_gen_len = sum(gen_lengths) / len(gen_lengths) if gen_lengths else 0.0

    result = {
        "model": model_name,
        "model_path": model_path,
        "method": "tokenmem",
        "gate_dir": gate_dir,
        "dataset": dataset,
        "scoring": "cot_nothink",
        "knowledge_max_len": knowledge_max_len,
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

    out_path = out_dir / f"{model_name}_tokenmem_{dataset}.json"
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
    """命令行入口，解析参数并启动 TokenMem CoT 评测。"""
    parser = argparse.ArgumentParser(description="TokenMem 模型 CoT 评测")
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
    parser.add_argument(
        "--cot-max-new-tokens",
        type=int,
        default=2048,
        help="CoT 生成最大 token 数 (默认: 2048)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="批量推理 batch size (默认: 8)",
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
        cot_max_new_tokens=args.cot_max_new_tokens,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
