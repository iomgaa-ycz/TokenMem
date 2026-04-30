"""诊断脚本：统计 4B/8B 在 vanilla_rag 和 no_memory 下的 CoT 生成长度分布。"""
import sys
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.eval_baseline import (
    build_cot_prompt,
    compress_passage,
    evaluate_cot,
    load_samples_jsonl,
    normalize_options,
)

N = 30
DATASETS = {
    "medqa": "data/ood/medqa.jsonl",
    "cf_medqa_val": "data/counterfactual/cf_medqa_val.jsonl",
    "arc_easy": "data/ood/arc_easy.jsonl",
}


def run_diag(model_path: str, device: str = "cuda:0"):
    print(f"\n{'='*60}")
    print(f"模型: {model_path}  |  device: {device}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    model.eval()

    for ds_name, ds_path in DATASETS.items():
        fpath = ROOT / ds_path
        if not fpath.exists():
            print(f"  [skip] {ds_name}: 文件不存在")
            continue
        samples = load_samples_jsonl(fpath, n_samples=N)

        for method in ["vanilla_rag", "no_memory"]:
            lengths = []
            for s in samples:
                opts, correct = normalize_options(s["options"], s["correct_letter"])
                valid = set(opts.keys())
                raw_passage = s.get("passage") or s.get("knowledge", "")

                if method == "vanilla_rag":
                    if not raw_passage:
                        continue
                    passage = compress_passage(raw_passage, target_token=64)
                    prompt = build_cot_prompt(s["question"], opts, passage=passage)
                else:
                    prompt = build_cot_prompt(s["question"], opts, passage=None)

                _, gen_len = evaluate_cot(model, tokenizer, prompt, valid, device=device)
                lengths.append(gen_len)

            if not lengths:
                continue
            arr = np.array(lengths)
            pct_max = np.mean(arr >= 1020) * 100
            print(
                f"  {ds_name:20s} | {method:12s} | "
                f"mean={arr.mean():.0f}  median={np.median(arr):.0f}  "
                f"min={arr.min()}  max={arr.max()}  "
                f">=1020tok: {pct_max:.0f}%  (n={len(arr)})"
            )

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    run_diag(args.model_path, args.device)
