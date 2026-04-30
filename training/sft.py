"""SFT 训练脚本 —— 仅训练 LinearFusion gate_crossattention 模块。

基于 DecoupledRAG 验证过的训练参数:
- 优化器: Lamb (lr=1e-3, wd=0.0)
- 调度器: LinearLR 10-step warmup
- 精度: bf16 (via Accelerator)
- 梯度累积: 支持
- 多卡: 通过 Accelerator 自动适配

用法:
    accelerate launch training/sft.py \\
        --model-name-or-path Qwen/Qwen3-0.6B \\
        --train-jsonl data/train.jsonl \\
        --val-jsonl data/val.jsonl \\
        --ckpt-dir checkpoints/qwen3-0.6b
"""

from __future__ import annotations

# S0: 强制行缓冲，确保 tqdm 和日志实时输出
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
from datetime import datetime  # noqa: E402
from typing import Any, Dict, Optional  # noqa: E402

import torch  # noqa: E402
from accelerate import Accelerator  # noqa: E402
from accelerate.utils import set_seed  # noqa: E402
from torch.optim.lr_scheduler import LinearLR  # noqa: E402
from torch.utils.data import ConcatDataset, DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from memory_lora.tokenmem_model import TokenMemForCausalLM  # noqa: E402
from training.data import (  # noqa: E402
    CounterfactualDataset,
    NewsQAOracleDataset,
    OversampledDataset,
    make_collate_fn,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# =========================================================================
# S1: CLI 参数解析
# =========================================================================


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    所有训练参数通过 CLI 传入，不依赖 YAML 配置文件。

    返回:
        argparse.Namespace: 解析后的参数命名空间
    """
    p = argparse.ArgumentParser(description="TokenMem SFT 训练脚本")

    # --- 必需参数 ---
    p.add_argument(
        "--model-name-or-path",
        type=str,
        required=True,
        help="HuggingFace 模型路径或名称",
    )
    p.add_argument(
        "--train-jsonl", type=str, required=True, help="训练集 JSONL 文件路径"
    )
    p.add_argument("--val-jsonl", type=str, required=True, help="验证集 JSONL 文件路径")
    p.add_argument("--ckpt-dir", type=str, required=True, help="检查点保存目录")

    # --- 训练超参数 ---
    p.add_argument("--epochs", type=int, default=5, help="训练轮数 (默认: 5)")
    p.add_argument(
        "--batch-size", type=int, default=16, help="每设备 batch size (默认: 16)"
    )
    p.add_argument("--lr", type=float, default=1e-3, help="学习率 (默认: 1e-3)")
    p.add_argument(
        "--weight-decay", type=float, default=0.0, help="权重衰减 (默认: 0.0)"
    )
    p.add_argument(
        "--grad-clip",
        type=float,
        default=0.0,
        help="梯度裁剪阈值，0=不裁剪 (默认: 0.0)",
    )
    p.add_argument(
        "--grad-accum-steps", type=int, default=1, help="梯度累积步数 (默认: 1)"
    )

    # --- 序列长度 ---
    p.add_argument(
        "--max-seq-len", type=int, default=64, help="query 最大 token 长度 (默认: 64)"
    )
    p.add_argument(
        "--knowledge-max-len",
        type=int,
        default=256,
        help="knowledge 最大 token 长度 (默认: 256)",
    )
    p.add_argument(
        "--knowledge-strided-len",
        type=int,
        default=64,
        help="knowledge strided sampling 目标长度 (默认: 64)",
    )

    # --- 检查点与评估频率 ---
    p.add_argument(
        "--save-steps", type=int, default=500, help="每 N 步保存检查点 (默认: 500)"
    )
    p.add_argument(
        "--eval-steps", type=int, default=500, help="每 N 步执行验证 (默认: 500)"
    )

    # --- 其他选项 ---
    p.add_argument(
        "--gradient-checkpointing", action="store_true", help="启用梯度检查点以节省显存"
    )
    p.add_argument("--no-swanlab", action="store_true", help="禁用 SwanLab 日志记录")
    p.add_argument(
        "--swanlab-project",
        type=str,
        default="tokenmem",
        help="SwanLab 项目名称 (默认: tokenmem)",
    )
    p.add_argument(
        "--num-workers", type=int, default=4, help="DataLoader worker 数量 (默认: 4)"
    )
    p.add_argument(
        "--knowledge-field",
        type=str,
        default="passage",
        help="JSONL 中用作 knowledge 的字段名 (默认: passage)",
    )
    p.add_argument(
        "--cf-train-jsonl",
        type=str,
        nargs="+",
        default=None,
        help="反事实训练集 JSONL 路径（可传多个）",
    )
    p.add_argument(
        "--cf-oversample",
        type=int,
        default=1,
        help="反事实数据过采样倍数 (默认: 1)",
    )
    p.add_argument(
        "--load-gates",
        type=str,
        default=None,
        help="加载已有 gate 权重的目录路径（用于 Phase 2 curriculum 训练）",
    )
    p.add_argument(
        "--cf-val-jsonl",
        type=str,
        nargs="+",
        default=None,
        help="反事实验证集 JSONL 路径（可传多个，用于分开追踪 CF val_loss）",
    )
    p.add_argument(
        "--prompt-mode",
        type=str,
        default="direct",
        choices=["direct", "cot"],
        help="训练 prompt 模式: direct='Answer:' 单字母, cot=CoT 完整推理 (默认: direct)",
    )
    p.add_argument("--seed", type=int, default=42, help="随机种子 (默认: 42)")

    return p.parse_args()


# =========================================================================
# S2: 检查点保存
# =========================================================================


def save_checkpoint(
    accelerator: Accelerator,
    model: TokenMemForCausalLM,
    save_dir: str,
    step: int,
    epoch: int,
    train_loss: float,
    val_loss: Optional[float] = None,
    is_best: bool = False,
) -> None:
    """保存 gate 权重检查点和元信息。

    仅在主进程执行写入操作。使用 unwrap_model 获取原始模型后调用 save_gates。

    参数:
        accelerator: Accelerator 实例
        model: 被 accelerator.prepare 包装后的模型
        save_dir: 检查点保存目录
        step: 当前全局步数
        epoch: 当前 epoch 编号
        train_loss: 当前训练 loss
        val_loss: 当前验证 loss（可选）
        is_best: 是否为最佳验证 loss 对应的检查点
    """
    if not accelerator.is_main_process:
        return

    os.makedirs(save_dir, exist_ok=True)
    unwrapped: TokenMemForCausalLM = accelerator.unwrap_model(model)
    unwrapped.save_gates(save_dir)

    meta = {
        "step": step,
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "is_best": is_best,
        "timestamp": datetime.now().isoformat(),
    }
    meta_path = os.path.join(save_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info("检查点已保存: %s (step=%d, val_loss=%s)", save_dir, step, val_loss)


# =========================================================================
# S3: SwanLab 辅助函数
# =========================================================================


def init_swanlab(args: argparse.Namespace, accelerator: Accelerator) -> None:
    """初始化 SwanLab 日志记录。

    仅主进程初始化。若 --no-swanlab 则跳过。
    SwanLab 未安装时捕获 ImportError 并降级为纯日志模式。

    参数:
        args: 命令行参数
        accelerator: Accelerator 实例
    """
    if args.no_swanlab or not accelerator.is_main_process:
        return

    try:
        import swanlab  # noqa: F811

        config_dict = vars(args)
        swanlab.init(
            project=args.swanlab_project,
            config=config_dict,
        )
        logger.info("SwanLab 已初始化 (project=%s)", args.swanlab_project)
    except ImportError:
        logger.warning("swanlab 未安装，跳过日志记录。pip install swanlab 以启用。")
        args.no_swanlab = True
    except Exception as e:
        logger.warning("SwanLab 初始化失败: %s，降级为纯日志模式。", e)
        args.no_swanlab = True


def log_swanlab(
    metrics: Dict[str, Any],
    step: int,
    args: argparse.Namespace,
    accelerator: Accelerator,
) -> None:
    """向 SwanLab 记录指标。

    仅主进程记录。若 --no-swanlab 则跳过。

    参数:
        metrics: 指标字典 (如 {"train/loss": 0.5, "lr": 1e-3})
        step: 当前全局步数
        args: 命令行参数
        accelerator: Accelerator 实例
    """
    if args.no_swanlab or not accelerator.is_main_process:
        return

    try:
        import swanlab  # noqa: F811

        swanlab.log(metrics, step=step)
    except Exception as e:
        logger.warning("SwanLab 记录失败: %s", e)


# =========================================================================
# S4: 验证函数
# =========================================================================


@torch.no_grad()
def evaluate(
    model: TokenMemForCausalLM,
    val_loader: DataLoader,
    accelerator: Accelerator,
) -> float:
    """在验证集上计算平均 loss。

    使用 @torch.no_grad 禁用梯度计算。
    通过 accelerator.reduce 实现多卡 loss 聚合。
    遇到 NaN loss 时跳过该 batch。

    参数:
        model: 被 accelerator.prepare 包装后的模型
        val_loader: 被 accelerator.prepare 包装后的验证 DataLoader
        accelerator: Accelerator 实例

    返回:
        float: 验证集平均 loss
    """
    model.eval()
    total_loss = torch.tensor(0.0, device=accelerator.device)
    total_count = torch.tensor(0, device=accelerator.device)

    for batch in val_loader:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            knowledge_input_ids=batch["knowledge_input_ids"],
            knowledge_attention_mask=batch["knowledge_attention_mask"],
        )

        loss = outputs.loss
        if loss is None or torch.isnan(loss):
            logger.warning("验证时遇到 NaN loss，跳过该 batch")
            continue

        total_loss += loss.detach()
        total_count += 1

    # 多卡聚合
    total_loss = accelerator.reduce(total_loss, reduction="sum")
    total_count = accelerator.reduce(total_count, reduction="sum")

    if total_count.item() == 0:
        logger.warning("验证集所有 batch 均为 NaN，返回 inf")
        return float("inf")

    avg_loss = (total_loss / total_count).item()
    model.train()
    return avg_loss


# =========================================================================
# S5: 主训练循环
# =========================================================================


def _create_optimizer(
    model: TokenMemForCausalLM, args: argparse.Namespace
) -> torch.optim.Optimizer:
    """创建优化器。

    优先使用 Lamb (torch_optimizer)，不可用时降级为 AdamW。

    参数:
        model: TokenMemForCausalLM 模型
        args: 命令行参数

    返回:
        torch.optim.Optimizer: 优化器实例
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    try:
        from torch_optimizer import Lamb

        optimizer = Lamb(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        logger.info("使用 Lamb 优化器 (lr=%.1e, wd=%.1e)", args.lr, args.weight_decay)
    except ImportError:
        logger.warning("torch_optimizer 未安装，降级为 AdamW")
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        logger.info("使用 AdamW 优化器 (lr=%.1e, wd=%.1e)", args.lr, args.weight_decay)

    return optimizer


def _log_param_stats(model: TokenMemForCausalLM, accelerator: Accelerator) -> None:
    """统计并打印可训练 / 总参数量。

    参数:
        model: TokenMemForCausalLM 模型
        accelerator: Accelerator 实例
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if accelerator.is_main_process:
        logger.info(
            "参数统计: 可训练 %s / 总计 %s (%.4f%%)",
            f"{trainable_params:,}",
            f"{total_params:,}",
            trainable_params / total_params * 100 if total_params > 0 else 0,
        )


def train(args: argparse.Namespace) -> None:
    """主训练函数。

    完整训练流程:
    1. 初始化 Accelerator (bf16)
    2. 加载 TokenMemForCausalLM (frozen base + trainable gates)
    3. 构建 Dataset + DataLoader + collate_fn
    4. 创建 Lamb 优化器 + LinearLR 调度器
    5. accelerator.prepare 包装所有组件
    6. epoch 循环: forward → backward → clip → step → 周期性 eval + save

    参数:
        args: parse_args() 返回的命令行参数
    """
    # --- 初始化 Accelerator ---
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.grad_accum_steps,
    )
    set_seed(args.seed)

    if accelerator.is_main_process:
        logger.info("=" * 60)
        logger.info("TokenMem SFT 训练开始")
        logger.info("=" * 60)
        logger.info(
            "训练参数: %s", json.dumps(vars(args), indent=2, ensure_ascii=False)
        )

    # --- SwanLab ---
    init_swanlab(args, accelerator)

    # --- 加载模型 ---
    if accelerator.is_main_process:
        logger.info("加载模型: %s", args.model_name_or_path)

    model = TokenMemForCausalLM(
        model_name_or_path=args.model_name_or_path,
        knowledge_max_seq_len=args.knowledge_strided_len,
        torch_dtype=torch.bfloat16,
    )

    if args.gradient_checkpointing:
        model.model.gradient_checkpointing_enable()
        if accelerator.is_main_process:
            logger.info("梯度检查点已启用")

    _log_param_stats(model, accelerator)

    # --- 加载已有 gate 权重（Curriculum Phase 2）---
    if args.load_gates:
        model.load_gates(args.load_gates)
        if accelerator.is_main_process:
            logger.info("已加载 gate 权重: %s", args.load_gates)

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if accelerator.is_main_process:
            logger.info("pad_token 设为 eos_token: %s", tokenizer.eos_token)

    # --- 数据集 ---
    train_dataset = NewsQAOracleDataset(
        jsonl_path=args.train_jsonl,
        knowledge_field=args.knowledge_field,
        prompt_mode=args.prompt_mode,
    )

    if args.cf_train_jsonl:
        cf_datasets = []
        for cf_path in args.cf_train_jsonl:
            cf_ds = CounterfactualDataset(cf_path, split="train", prompt_mode=args.prompt_mode)
            if args.cf_oversample > 1:
                cf_ds = OversampledDataset(cf_ds, factor=args.cf_oversample)
            cf_datasets.append(cf_ds)

        news_len = len(train_dataset)
        train_dataset = ConcatDataset([train_dataset] + cf_datasets)

        if accelerator.is_main_process:
            cf_total = sum(len(d) for d in cf_datasets)
            logger.info(
                "合并数据集: News %d + CF %d (oversample=%d) = %d 样本",
                news_len,
                cf_total,
                args.cf_oversample,
                len(train_dataset),
            )

    val_dataset = NewsQAOracleDataset(
        jsonl_path=args.val_jsonl,
        knowledge_field=args.knowledge_field,
        prompt_mode=args.prompt_mode,
    )

    collate_fn = make_collate_fn(
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        knowledge_max_len=args.knowledge_max_len,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # --- CF 验证集（可选）---
    cf_val_loader = None
    if args.cf_val_jsonl:
        cf_val_datasets = []
        for cf_val_path in args.cf_val_jsonl:
            cf_val_ds = CounterfactualDataset(cf_val_path, split="test", prompt_mode=args.prompt_mode)
            cf_val_datasets.append(cf_val_ds)

        cf_val_dataset = ConcatDataset(cf_val_datasets)
        cf_val_loader = DataLoader(
            cf_val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        if accelerator.is_main_process:
            logger.info("CF 验证集: %d 样本", len(cf_val_dataset))

    # --- 优化器 + 调度器 ---
    optimizer = _create_optimizer(model, args)
    scheduler = LinearLR(
        optimizer,
        start_factor=1.0 / 3.0,
        end_factor=1.0,
        total_iters=10,
    )

    # --- Accelerator 包装 ---
    prepare_args = [model, optimizer, train_loader, val_loader, scheduler]
    if cf_val_loader is not None:
        prepare_args.append(cf_val_loader)
    prepared = accelerator.prepare(*prepare_args)
    model, optimizer, train_loader, val_loader, scheduler = prepared[:5]
    if cf_val_loader is not None:
        cf_val_loader = prepared[5]

    # --- 训练状态 ---
    global_step = 0
    best_val_loss = float("inf")
    total_steps_per_epoch = len(train_loader)
    total_steps = total_steps_per_epoch * args.epochs

    if accelerator.is_main_process:
        logger.info(
            "训练规模: %d 样本, %d steps/epoch, %d 总 steps",
            len(train_dataset),
            total_steps_per_epoch,
            total_steps,
        )

    # --- Epoch 循环 ---
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        progress_bar = tqdm(
            total=total_steps_per_epoch,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            disable=not accelerator.is_main_process,
        )

        for batch_idx, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    knowledge_input_ids=batch["knowledge_input_ids"],
                    knowledge_attention_mask=batch["knowledge_attention_mask"],
                )

                loss = outputs.loss
                if loss is None or torch.isnan(loss):
                    logger.warning(
                        "Epoch %d, batch %d: NaN loss，跳过",
                        epoch + 1,
                        batch_idx,
                    )
                    progress_bar.update(1)
                    continue

                accelerator.backward(loss)

                # 梯度裁剪
                if args.grad_clip > 0 and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # 梯度累积完成时才递增 global_step
            if accelerator.sync_gradients:
                global_step += 1
                current_loss = loss.detach().item()
                epoch_loss += current_loss
                epoch_steps += 1

                # 更新进度条
                current_lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix(
                    loss=f"{current_loss:.4f}",
                    lr=f"{current_lr:.2e}",
                    step=global_step,
                )

                # 日志记录
                log_swanlab(
                    {"train/loss": current_loss, "train/lr": current_lr},
                    step=global_step,
                    args=args,
                    accelerator=accelerator,
                )

                # 周期性保存
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    ckpt_path = os.path.join(args.ckpt_dir, f"step_{global_step}")
                    save_checkpoint(
                        accelerator,
                        model,
                        ckpt_path,
                        step=global_step,
                        epoch=epoch + 1,
                        train_loss=current_loss,
                    )

                # 周期性验证
                if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    val_loss = evaluate(model, val_loader, accelerator)
                    if accelerator.is_main_process:
                        logger.info(
                            "Step %d 验证: val_loss=%.4f (best=%.4f)",
                            global_step,
                            val_loss,
                            best_val_loss,
                        )
                    log_swanlab(
                        {"val/loss": val_loss},
                        step=global_step,
                        args=args,
                        accelerator=accelerator,
                    )

                    # CF 验证（可选）
                    if cf_val_loader is not None:
                        cf_val_loss = evaluate(model, cf_val_loader, accelerator)
                        if accelerator.is_main_process:
                            logger.info(
                                "Step %d CF 验证: cf_val_loss=%.4f",
                                global_step,
                                cf_val_loss,
                            )
                        log_swanlab(
                            {"val/cf_loss": cf_val_loss},
                            step=global_step,
                            args=args,
                            accelerator=accelerator,
                        )

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_path = os.path.join(args.ckpt_dir, "best")
                        save_checkpoint(
                            accelerator,
                            model,
                            best_path,
                            step=global_step,
                            epoch=epoch + 1,
                            train_loss=current_loss,
                            val_loss=val_loss,
                            is_best=True,
                        )

                    model.train()

            progress_bar.update(1)

        progress_bar.close()

        # --- Epoch 结束: 验证 + 保存 ---
        avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else float("inf")
        val_loss = evaluate(model, val_loader, accelerator)

        if accelerator.is_main_process:
            logger.info(
                "Epoch %d/%d 完成: avg_train_loss=%.4f, val_loss=%.4f (best=%.4f)",
                epoch + 1,
                args.epochs,
                avg_epoch_loss,
                val_loss,
                best_val_loss,
            )

        log_swanlab(
            {
                "epoch/train_loss": avg_epoch_loss,
                "epoch/val_loss": val_loss,
                "epoch": epoch + 1,
            },
            step=global_step,
            args=args,
            accelerator=accelerator,
        )

        # Epoch 结束 CF 验证（可选）
        if cf_val_loader is not None:
            cf_val_loss = evaluate(model, cf_val_loader, accelerator)
            if accelerator.is_main_process:
                logger.info(
                    "Epoch %d CF 验证: cf_val_loss=%.4f",
                    epoch + 1,
                    cf_val_loss,
                )
            log_swanlab(
                {"epoch/cf_val_loss": cf_val_loss},
                step=global_step,
                args=args,
                accelerator=accelerator,
            )

        # Epoch 检查点
        epoch_path = os.path.join(args.ckpt_dir, f"epoch_{epoch + 1}")
        save_checkpoint(
            accelerator,
            model,
            epoch_path,
            step=global_step,
            epoch=epoch + 1,
            train_loss=avg_epoch_loss,
            val_loss=val_loss,
        )

        # 更新 best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.ckpt_dir, "best")
            save_checkpoint(
                accelerator,
                model,
                best_path,
                step=global_step,
                epoch=epoch + 1,
                train_loss=avg_epoch_loss,
                val_loss=val_loss,
                is_best=True,
            )

    # --- 训练结束 ---
    if accelerator.is_main_process:
        logger.info("=" * 60)
        logger.info("训练完成! best_val_loss=%.4f", best_val_loss)
        logger.info("=" * 60)

    # SwanLab 关闭
    if not args.no_swanlab and accelerator.is_main_process:
        try:
            import swanlab

            swanlab.finish()
        except Exception:
            pass


# =========================================================================
# S6: 入口点
# =========================================================================

if __name__ == "__main__":
    train(parse_args())
