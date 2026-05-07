"""绘制 Difficulty-conditioned KC Comparison 柱状图。

从 data_difficulty_kc.json 读取数据，绘制 Vanilla RAG vs TokenMem
在不同 parametric-prior 强度下的 KC 对比。

用法：
    python -m tools.plot_difficulty_kc \
        --input TokenMem-paper/figures/data_difficulty_kc.json \
        --output TokenMem-paper/figures/difficulty_kc_comparison.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# 统一色板 & 出版样式
# ---------------------------------------------------------------------------

C_BLUE = "#2D4A5E"
C_RED = "#C47A52"
C_GREEN = "#2E7D32"

matplotlib.rcParams.update(
    {
        "font.size": 10,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.usetex": False,
        "mathtext.fontset": "stix",
    }
)


def _ci95(p: float, n: int) -> float:
    """二项比例 95% 置信区间半宽。"""
    if n <= 0:
        return 0.0
    p_frac = p / 100.0
    se = np.sqrt(p_frac * (1 - p_frac) / n)
    return 1.96 * se * 100.0


def plot_difficulty_kc(data: dict, output: Path) -> None:
    """绘制双数据集 difficulty-conditioned KC 柱状图。"""
    datasets = list(data.keys())
    groups = ["High", "Medium", "Low"]
    n_groups = len(groups)
    bar_width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=True)

    for ax_idx, (ax, ds_name) in enumerate(zip(axes, datasets)):
        ds = data[ds_name]
        x = np.arange(n_groups)

        rag_vals = [ds[g]["VanillaRAG"] for g in groups]
        tm_vals = [ds[g]["TokenMem"] for g in groups]
        ns = [ds[g]["N"] for g in groups]

        min_n_for_ci = 30
        rag_ci = [_ci95(v, n) if n >= min_n_for_ci else 0.0 for v, n in zip(rag_vals, ns)]
        tm_ci = [_ci95(v, n) if n >= min_n_for_ci else 0.0 for v, n in zip(tm_vals, ns)]

        bars_rag = ax.bar(
            x - bar_width / 2, rag_vals, bar_width,
            color=C_BLUE, edgecolor="none", label="Vanilla RAG",
        )
        bars_tm = ax.bar(
            x + bar_width / 2, tm_vals, bar_width,
            color=C_RED, edgecolor="none", label="TokenMem",
        )

        for i in range(n_groups):
            delta = tm_vals[i] - rag_vals[i]
            bar_rag = bars_rag[i]
            bar_tm = bars_tm[i]
            rag_x = bar_rag.get_x() + bar_rag.get_width() / 2
            tm_x = bar_tm.get_x() + bar_tm.get_width() / 2

            if abs(delta) < 0.05:
                ax.text(
                    rag_x, rag_vals[i] - 4,
                    f"{rag_vals[i]:.1f}", ha="center", va="top", fontsize=7,
                    color="white", fontweight="bold",
                )
                ax.text(
                    tm_x, tm_vals[i] - 4,
                    f"{tm_vals[i]:.1f}", ha="center", va="top", fontsize=7,
                    color="white", fontweight="bold",
                )
                continue

            ax.text(
                rag_x, rag_vals[i] - 4,
                f"{rag_vals[i]:.1f}", ha="center", va="top", fontsize=7,
                color="white", fontweight="bold",
            )
            ax.text(
                tm_x, tm_vals[i] - 4,
                f"{tm_vals[i]:.1f}", ha="center", va="top", fontsize=7,
                color="white", fontweight="bold",
            )

            cap_half = 0.04
            y_bot = rag_vals[i]
            y_top = tm_vals[i]
            ax.plot(
                [rag_x - cap_half, rag_x + cap_half], [y_bot, y_bot],
                color="#555555", linewidth=0.8, zorder=4,
            )
            ax.plot(
                [rag_x, rag_x], [y_bot, y_top],
                color="#555555", linewidth=0.8, zorder=4,
            )
            ax.plot(
                [rag_x - cap_half, rag_x + cap_half], [y_top, y_top],
                color="#555555", linewidth=0.8, zorder=4,
            )
            y_mid = (y_bot + y_top) / 2
            ax.text(
                rag_x, y_mid,
                f"{delta:.1f}", ha="center", va="center", fontsize=7,
                color=C_GREEN, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.8),
                zorder=5,
            )

        ax.set_title(ds_name, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_xlabel("Conflict Difficulty")
        if ax_idx == 0:
            ax.set_ylabel("Knowledge Compliance (%)")
            ax.legend(frameon=False, loc="upper left", fontsize=8)

    ax.set_ylim(0, 95)

    plt.subplots_adjust(wspace=0.08)

    fmt = output.suffix.lstrip(".")
    fig.savefig(output, format=fmt)
    print(f"Saved: {output}")

    png_path = output.with_suffix(".png")
    fig.savefig(png_path, format="png")
    print(f"Saved: {png_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Difficulty KC Comparison 图")
    parser.add_argument(
        "--input", type=str,
        default="TokenMem-paper/figures/data_difficulty_kc.json",
    )
    parser.add_argument(
        "--output", type=str,
        default="TokenMem-paper/figures/difficulty_kc_comparison.pdf",
    )
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_difficulty_kc(data, output)


if __name__ == "__main__":
    main()
