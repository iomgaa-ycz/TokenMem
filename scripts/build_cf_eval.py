"""从反事实 JSONL 中提取 test split，重映射字段为 eval_baseline.py 兼容格式。

用法：
    conda run -n ExplicitLLM python scripts/build_cf_eval.py
"""

import json
from pathlib import Path


def build_cf_eval(src: Path, dst: Path) -> int:
    """从 src 提取 split=='test' 的行，重映射字段，写入 dst。

    字段映射：
        counterfactual_passage → passage
        target_letter → correct_letter
    保留：question, options, dataset, cf_id

    参数：
        src: 源反事实 JSONL 路径。
        dst: 输出 eval-ready JSONL 路径。

    返回：
        写入的行数。
    """
    count = 0
    with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            if row.get("split") != "test":
                continue
            out = {
                "id": row["cf_id"],
                "dataset": row["dataset"],
                "question": row["question"],
                "options": row["options"],
                "correct_letter": row["target_letter"],
                "passage": row["counterfactual_passage"],
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    """预处理两个反事实数据集。"""
    root = Path(__file__).resolve().parent.parent
    tasks = [
        (root / "data/counterfactual/arc_easy.jsonl", root / "data/counterfactual/cf_arc_easy_val.jsonl"),
        (root / "data/counterfactual/medqa.jsonl", root / "data/counterfactual/cf_medqa_val.jsonl"),
    ]
    for src, dst in tasks:
        n = build_cf_eval(src, dst)
        print(f"[done] {src.name} → {dst.name}: {n} 条 (test split)")


if __name__ == "__main__":
    main()
