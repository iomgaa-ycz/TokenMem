"""绘制 Gate Activation Magnitude 分析图。

从 gate_activation_top100.jsonl 读取 top 100 对样本数据，
绘制逐层 conflict vs non-conflict 的 gate relative ratio 对比图。

用法：
    python -m tools.plot_mechanism_analysis \
        --input results/mechanism/gate_activation_top100.jsonl \
        --output TokenMem-paper/figures/gate_activation.pdf
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------

matplotlib.rcParams.update(
    {
        "font.size": 10,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 10,
        "axes.titlesize": 11,
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

C_NC = "#2D4A5E"
C_CF = "#C47A52"
C_DIFF = "#C47A52"


def load_top100(path: Path) -> dict:
    """从 top100 JSONL 读取逐层统计。

    返回：
        {"layers": list[int], "nc_mean", "nc_std", "cf_mean", "cf_std", "diff_mean"}
    """
    recs = [json.loads(line) for line in open(path) if line.strip()]

    nc_per_layer: dict[int, list[float]] = defaultdict(list)
    cf_per_layer: dict[int, list[float]] = defaultdict(list)

    for rec in recs:
        for entry in rec["layers"]["nonconflict"]:
            nc_per_layer[entry["layer"]].append(entry["ratio"])
        for entry in rec["layers"]["conflict"]:
            cf_per_layer[entry["layer"]].append(entry["ratio"])

    layers = sorted(nc_per_layer.keys())
    nc_mean = [np.mean(nc_per_layer[l]) for l in layers]
    nc_std = [np.std(nc_per_layer[l]) for l in layers]
    cf_mean = [np.mean(cf_per_layer[l]) for l in layers]
    cf_std = [np.std(cf_per_layer[l]) for l in layers]
    diff_mean = [cf_mean[i] - nc_mean[i] for i in range(len(layers))]

    return {
        "layers": layers,
        "nc_mean": np.array(nc_mean),
        "nc_std": np.array(nc_std),
        "cf_mean": np.array(cf_mean),
        "cf_std": np.array(cf_std),
        "diff_mean": np.array(diff_mean),
    }


def plot_gate_activation(data: dict, output: Path) -> None:
    """绘制宽幅 gate activation 分析图（适配 NeurIPS 单栏）。

    上面板: 双线 + 两线间 gap 填充（绿=conflict更高，灰=反之）
    下面板: 紧凑的 per-layer Δ bar chart
    """
    layers = np.array(data["layers"])
    n_layers = len(layers)

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(5.5, 2.8),
        height_ratios=[1.6, 1],
        sharex=True,
        gridspec_kw={"hspace": 0.20},
    )

    # --- 上面板: 双线 + gap 填充 ---
    ax1.plot(
        layers, data["nc_mean"], color=C_NC, linewidth=1.2,
        label="Non-conflict", zorder=3,
    )
    ax1.plot(
        layers, data["cf_mean"], color=C_CF, linewidth=1.2,
        label="Conflict", zorder=3,
    )

    # 关键：填充两线之间的间隙——绿色表示 conflict > nonconflict
    ax1.fill_between(
        layers, data["nc_mean"], data["cf_mean"],
        where=data["cf_mean"] >= data["nc_mean"],
        color=C_DIFF, alpha=0.30, interpolate=True,
        label="Conflict > Non-conflict",
        zorder=2,
    )
    ax1.fill_between(
        layers, data["nc_mean"], data["cf_mean"],
        where=data["cf_mean"] < data["nc_mean"],
        color="#BBBBBB", alpha=0.30, interpolate=True,
        zorder=2,
    )

    ax1.set_ylabel(r"$\|\mathbf{G}^\ell\| / \|\tilde{\mathbf{h}}^\ell\|$")
    # 三个图例项合并到右上角，竖排排列
    ax1.legend(frameon=False, loc="upper right", ncol=1, fontsize=8)
    ax1.set_xlim(-0.5, n_layers - 0.5)

    # 裁剪 y 轴：排除 layer 0 异常高值，底部紧贴数据
    y_vals_all = np.concatenate([data["nc_mean"][1:], data["cf_mean"][1:]])
    y_min = y_vals_all.min() - 0.005
    y_max = y_vals_all.max() * 1.12
    ax1.set_ylim(max(0, y_min), y_max)

    # --- 下面板: Δ bar chart ---
    bar_colors = [C_DIFF if d >= 0 else "#BBBBBB" for d in data["diff_mean"]]
    ax2.bar(layers, data["diff_mean"], width=0.7, color=bar_colors, edgecolor="none")
    ax2.axhline(0, color="black", linewidth=0.4, zorder=1)
    ax2.set_xlabel("Layer Index")
    ax2.set_ylabel(r"$\Delta$")

    tick_positions = list(range(0, n_layers, 5))
    if (n_layers - 1) not in tick_positions:
        tick_positions.append(n_layers - 1)
    ax2.set_xticks(tick_positions)

    # 标注 top-3 peak 层
    top3_idx = np.argsort(data["diff_mean"])[-3:][::-1]
    for idx in top3_idx:
        ax2.annotate(
            f"L{layers[idx]}",
            xy=(layers[idx], data["diff_mean"][idx]),
            xytext=(0, 2),
            textcoords="offset points",
            fontsize=7,
            color="#2E7D32",
            ha="center",
            fontweight="bold",
        )

    # 保存
    fmt = output.suffix.lstrip(".")
    fig.savefig(output, format=fmt)
    print(f"Saved: {output}")

    # 同时输出 PNG 预览
    png_path = output.with_suffix(".png")
    fig.savefig(png_path, format="png")
    print(f"Saved: {png_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate Activation 分析图")
    parser.add_argument(
        "--input",
        type=str,
        default="results/mechanism/gate_activation_top100.jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="TokenMem-paper/figures/gate_activation.pdf",
    )
    args = parser.parse_args()

    data = load_top100(Path(args.input))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_gate_activation(data, output)


if __name__ == "__main__":
    main()
