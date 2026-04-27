"""TokenMemoryBank —— 基于 FAISS 的 token 级知识存储与检索。

核心设计：
- 每个 entry 存储 (token_ids, embedding)
- 内部持有 tokenizer，负责 tokenize/decode
- FAISS IndexIDMap(IndexFlatIP) 做余弦相似度检索（embedding 入索引前 L2-normalize）
- 软删除 + 自动 compact 机制回收空间

不变量：
- token_ids.dtype == long, shape [fusion_length]
- embedding.dtype == float32, shape [emb_dim]
- FAISS 内存储的是 L2-normalized embedding；_embs 存原始值
"""

from __future__ import annotations

import warnings
from typing import Any, List, Tuple

import faiss
import numpy as np
import torch
from torch import LongTensor, Tensor


class TokenMemoryBank:
    """Token 级知识存储库：存储 (token_ids, embedding) 对，支持 FAISS 检索。

    功能：
    - add: 批量写入 (text, embedding)，内部 tokenize
    - retrieve: FAISS 余弦 top-k 检索
    - edit/delete: 单条更新/软删除
    - audit/migrate_to: 文本审计/批量导出
    - save/load: 持久化（不含 FAISS 索引，load 时重建）
    """

    def __init__(
        self,
        tokenizer: Any,
        capacity: int = 1_000_000,
        fusion_length: int = 256,
        emb_dim: int = 1024,
        device: torch.device = torch.device("cpu"),
        compact_threshold: float = 0.3,
    ):
        """初始化 TokenMemoryBank。

        参数：
            tokenizer: HuggingFace tokenizer 实例，用于 tokenize/decode
            capacity: 最大 entry 数量
            fusion_length: 每条 entry 的 token 序列长度
            emb_dim: embedding 维度
            device: 张量存储设备
            compact_threshold: 已删除比例超过此阈值时自动触发 compact
        """
        self.tokenizer = tokenizer
        self.capacity = capacity
        self.fusion_length = fusion_length
        self.emb_dim = emb_dim
        self.device = device
        self.compact_threshold = compact_threshold

        # 预分配存储 buffer
        self._tokens = torch.zeros(
            capacity, fusion_length, dtype=torch.long, device=device
        )
        self._embs = torch.zeros(capacity, emb_dim, dtype=torch.float32, device=device)
        self._deleted = torch.zeros(capacity, dtype=torch.bool, device=device)
        self._n = 0
        self._n_deleted = 0

        # 创建 FAISS 索引：IndexFlatIP + IDMap（L2-normalize 后 IP 等效余弦）
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(emb_dim))

    # ─────────────────────────────────────────────
    # 基础查询
    # ─────────────────────────────────────────────

    def __len__(self) -> int:
        """返回活跃（未删除）entry 数量。"""
        return self._n - self._n_deleted

    # ─────────────────────────────────────────────
    # 校验
    # ─────────────────────────────────────────────

    def _validate(self, token_ids: LongTensor, embedding: Tensor) -> None:
        """校验 entry 张量的 dtype 和形状。

        参数：
            token_ids: 待校验的 token 序列，期望 dtype=long, shape=[fusion_length]
            embedding: 待校验的嵌入向量，期望 shape=[emb_dim]

        异常：
            TypeError: token_ids dtype 不是 torch.long
            ValueError: 形状不符合预期
        """
        if token_ids.dtype != torch.long:
            raise TypeError(f"token_ids must be long, got {token_ids.dtype}")
        if token_ids.shape != (self.fusion_length,):
            raise ValueError(
                f"token_ids shape must be [{self.fusion_length}], "
                f"got {list(token_ids.shape)}"
            )
        if embedding.shape != (self.emb_dim,):
            raise ValueError(
                f"embedding shape must be [{self.emb_dim}], got {list(embedding.shape)}"
            )

    # ─────────────────────────────────────────────
    # 文本 → token_ids
    # ─────────────────────────────────────────────

    def _tokenize_text(self, text: str) -> LongTensor:
        """将文本 tokenize 并 pad/truncate 到 fusion_length。

        参数：
            text: 输入文本

        返回：
            形状 [fusion_length] 的 LongTensor
        """
        ids: List[int] = self.tokenizer.encode(text, add_special_tokens=False)

        # 截断
        if len(ids) > self.fusion_length:
            ids = ids[: self.fusion_length]

        # 填充
        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else 0
        )
        if len(ids) < self.fusion_length:
            ids = ids + [pad_id] * (self.fusion_length - len(ids))

        return torch.tensor(ids, dtype=torch.long, device=self.device)

    # ─────────────────────────────────────────────
    # 写入
    # ─────────────────────────────────────────────

    def add(self, entries: List[Tuple[str, Tensor]]) -> List[int]:
        """批量写入 (text, embedding) 对。

        内部流程：tokenize text → _validate → 存储到 buffer → 加入 FAISS。
        容量不足时先尝试 _compact()；仍不够则抛 RuntimeError。

        参数：
            entries: [(text, embedding), ...] 列表

        返回：
            分配的 entry_id 列表

        异常：
            RuntimeError: compact 后仍无足够空间
            ValueError: embedding 形状不匹配
        """
        # 检查容量（考虑 compact 可能回收空间）
        if self._n + len(entries) > self.capacity:
            self._compact()
            if self._n + len(entries) > self.capacity:
                raise RuntimeError(
                    f"TokenMemoryBank full: capacity={self.capacity}, "
                    f"active={len(self)}, requested={len(entries)}"
                )

        assigned_ids: List[int] = []
        faiss_ids = []
        faiss_embs = []

        for text, embedding in entries:
            token_ids = self._tokenize_text(text)
            self._validate(token_ids, embedding)

            idx = self._n
            self._tokens[idx] = token_ids
            self._embs[idx] = embedding.to(device=self.device, dtype=torch.float32)
            self._deleted[idx] = False
            self._n += 1
            assigned_ids.append(idx)

            # 准备 FAISS 批量添加数据
            emb_np = embedding.detach().float().cpu().numpy().reshape(1, -1)
            emb_np = emb_np / (np.linalg.norm(emb_np, axis=1, keepdims=True) + 1e-8)
            faiss_ids.append(idx)
            faiss_embs.append(emb_np)

        # 批量添加到 FAISS
        if faiss_embs:
            all_embs = np.concatenate(faiss_embs, axis=0).astype(np.float32)
            all_ids = np.array(faiss_ids, dtype=np.int64)
            self._index.add_with_ids(all_embs, all_ids)

        return assigned_ids

    # ─────────────────────────────────────────────
    # 单条访问
    # ─────────────────────────────────────────────

    def __getitem__(self, entry_id: int) -> Tuple[LongTensor, Tensor]:
        """返回指定 entry 的 (token_ids, embedding) 克隆。

        参数：
            entry_id: entry 索引

        返回：
            (token_ids clone, embedding clone) 元组

        异常：
            IndexError: entry_id 不在 [0, _n) 范围
            ValueError: entry 已被删除
        """
        if not (0 <= entry_id < self._n):
            raise IndexError(f"entry_id {entry_id} out of range [0, {self._n})")
        if self._deleted[entry_id]:
            raise ValueError(f"entry {entry_id} has been deleted")
        return self._tokens[entry_id].clone(), self._embs[entry_id].clone()

    # ─────────────────────────────────────────────
    # 批量获取 token_ids
    # ─────────────────────────────────────────────

    def get_token_ids(self, entry_ids: LongTensor) -> LongTensor:
        """批量获取 token_ids。

        参数：
            entry_ids: 1D [B] 或 2D [B, k] 的 entry id 张量

        返回：
            1D 输入 → [B, fusion_length]
            2D 输入 → [B, k, fusion_length]
        """
        original_shape = entry_ids.shape
        flat_ids = entry_ids.reshape(-1)
        result = self._tokens[flat_ids]  # [N, fusion_length]
        return result.reshape(*original_shape, self.fusion_length)

    # ─────────────────────────────────────────────
    # FAISS 检索
    # ─────────────────────────────────────────────

    def retrieve(self, query_emb: Tensor, k: int = 1) -> Tuple[LongTensor, Tensor]:
        """FAISS 余弦 top-k 检索。

        查询 embedding 先 L2-normalize，再用 IndexFlatIP 做内积搜索。

        参数：
            query_emb: [B, emb_dim] 查询嵌入
            k: 返回的候选数量

        返回：
            (ids [B, k], scores [B, k]) 元组

        异常：
            RuntimeError: bank 为空
        """
        if len(self) == 0:
            raise RuntimeError("TokenMemoryBank empty — cannot retrieve")

        # L2-normalize 查询
        q_np = query_emb.detach().float().cpu().numpy()
        norms = np.linalg.norm(q_np, axis=1, keepdims=True) + 1e-8
        q_np = (q_np / norms).astype(np.float32)

        scores_np, ids_np = self._index.search(q_np, k)

        ids_tensor = torch.from_numpy(ids_np).long()
        scores_tensor = torch.from_numpy(scores_np).float()
        return ids_tensor, scores_tensor

    # ─────────────────────────────────────────────
    # 编辑
    # ─────────────────────────────────────────────

    def edit(self, entry_id: int, text: str, embedding: Tensor) -> None:
        """更新指定 entry 的 token_ids 和 embedding。

        同时更新 FAISS 索引（移除旧 → 添加新）。

        参数：
            entry_id: 目标 entry 索引
            text: 新文本
            embedding: 新嵌入向量

        异常：
            IndexError: entry_id 不在有效范围
            ValueError: entry 已被删除
        """
        if not (0 <= entry_id < self._n):
            raise IndexError(f"entry_id {entry_id} out of range [0, {self._n})")
        if self._deleted[entry_id]:
            raise ValueError(f"entry {entry_id} has been deleted")

        token_ids = self._tokenize_text(text)
        self._validate(token_ids, embedding)

        # 更新存储
        self._tokens[entry_id] = token_ids
        self._embs[entry_id] = embedding.to(device=self.device, dtype=torch.float32)

        # 更新 FAISS：先移除旧的，再添加新的
        self._index.remove_ids(np.array([entry_id], dtype=np.int64))
        emb_np = embedding.detach().float().cpu().numpy().reshape(1, -1)
        emb_np = emb_np / (np.linalg.norm(emb_np, axis=1, keepdims=True) + 1e-8)
        self._index.add_with_ids(
            emb_np.astype(np.float32),
            np.array([entry_id], dtype=np.int64),
        )

    # ─────────────────────────────────────────────
    # 审计 / 迁移
    # ─────────────────────────────────────────────

    def audit(self, entry_id: int) -> str:
        """将指定 entry 的 token_ids 解码为可读文本。

        参数：
            entry_id: entry 索引

        返回：
            解码后的文本字符串

        异常：
            IndexError: entry_id 不在有效范围
            ValueError: entry 已被删除
        """
        if not (0 <= entry_id < self._n):
            raise IndexError(f"entry_id {entry_id} out of range [0, {self._n})")
        if self._deleted[entry_id]:
            raise ValueError(f"entry {entry_id} has been deleted")

        token_ids = self._tokens[entry_id]
        return self.tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)

    def migrate_to(self) -> List[str]:
        """导出所有未删除 entry 的文本。

        返回：
            文本列表（按存储顺序）
        """
        texts: List[str] = []
        for i in range(self._n):
            if not self._deleted[i]:
                decoded = self.tokenizer.decode(
                    self._tokens[i].tolist(), skip_special_tokens=True
                )
                texts.append(decoded)
        return texts

    # ─────────────────────────────────────────────
    # 删除
    # ─────────────────────────────────────────────

    def delete(self, entry_id: int) -> None:
        """软删除指定 entry。

        标记 _deleted、增加 _n_deleted、从 FAISS 移除、检查是否需要 compact。

        参数：
            entry_id: entry 索引

        异常：
            IndexError: entry_id 不在有效范围
            ValueError: entry 已被删除
        """
        if not (0 <= entry_id < self._n):
            raise IndexError(f"entry_id {entry_id} out of range [0, {self._n})")
        if self._deleted[entry_id]:
            raise ValueError(f"entry {entry_id} already deleted")

        self._deleted[entry_id] = True
        self._n_deleted += 1

        # 从 FAISS 移除
        self._index.remove_ids(np.array([entry_id], dtype=np.int64))

        self._maybe_compact()

    def _maybe_compact(self) -> None:
        """检查删除比例是否超过阈值，超过则触发 compact。"""
        if self._n > 0 and self._n_deleted / self._n > self.compact_threshold:
            self._compact()

    def _compact(self) -> None:
        """压缩存储：将活跃 entry 移到前端，重置删除标记，重建 FAISS。

        步骤：
        1. 收集所有未删除 entry 的 tokens/embs
        2. 移到 buffer 前端
        3. 清除后续位置和删除标记
        4. 重建 FAISS 索引
        """
        alive_mask = ~self._deleted[: self._n]
        n_alive = int(alive_mask.sum().item())

        if n_alive == 0:
            self._n = 0
            self._n_deleted = 0
            self._deleted.zero_()
            self._build_faiss_index()
            return

        # 提取活跃数据
        alive_tokens = self._tokens[: self._n][alive_mask].clone()
        alive_embs = self._embs[: self._n][alive_mask].clone()

        # 写回 buffer 前端
        self._tokens[:n_alive] = alive_tokens
        self._embs[:n_alive] = alive_embs

        # 清理后续位置
        if n_alive < self.capacity:
            self._tokens[n_alive:].zero_()
            self._embs[n_alive:].zero_()

        # 重置删除标记
        self._deleted.zero_()
        self._n = n_alive
        self._n_deleted = 0

        # 重建 FAISS
        self._build_faiss_index()

    def _build_faiss_index(self) -> None:
        """从当前活跃 _embs 重建 FAISS 索引（L2-normalize 后添加）。"""
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(self.emb_dim))

        if self._n == 0:
            return

        alive_mask = ~self._deleted[: self._n]
        alive_indices = torch.where(alive_mask)[0]

        if len(alive_indices) == 0:
            return

        embs = self._embs[alive_indices].detach().float().cpu().numpy()
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
        embs_normalized = (embs / norms).astype(np.float32)

        ids = alive_indices.numpy().astype(np.int64)
        self._index.add_with_ids(embs_normalized, ids)

    # ─────────────────────────────────────────────
    # 持久化
    # ─────────────────────────────────────────────

    def save(self, path: str) -> None:
        """将 bank 状态序列化到文件。

        保存内容：capacity, fusion_length, emb_dim, n, n_deleted,
        tokens[:n], embs[:n], deleted[:n], tokenizer_name, compact_threshold。
        FAISS 索引不持久化（load 时重建）。

        参数：
            path: 保存路径
        """
        tokenizer_name = getattr(self.tokenizer, "name_or_path", "unknown")
        torch.save(
            {
                "capacity": self.capacity,
                "fusion_length": self.fusion_length,
                "emb_dim": self.emb_dim,
                "n": self._n,
                "n_deleted": self._n_deleted,
                "tokens": self._tokens[: self._n].cpu(),
                "embs": self._embs[: self._n].cpu(),
                "deleted": self._deleted[: self._n].cpu(),
                "tokenizer_name": tokenizer_name,
                "compact_threshold": self.compact_threshold,
            },
            path,
        )

    def load(self, path: str) -> None:
        """从文件恢复 bank 状态。

        校验 fusion_length/emb_dim 是否匹配。
        tokenizer_name 不同时发出警告。
        capacity 不同时重建内部 buffer。
        加载完成后重建 FAISS 索引。

        参数：
            path: 状态文件路径

        异常：
            ValueError: fusion_length 或 emb_dim 不匹配
        """
        state = torch.load(path, map_location=self.device, weights_only=True)

        # Phase 1: 校验不可变参数
        if state["fusion_length"] != self.fusion_length:
            raise ValueError(
                f"fusion_length mismatch: saved={state['fusion_length']} "
                f"vs bank={self.fusion_length}"
            )
        if state["emb_dim"] != self.emb_dim:
            raise ValueError(
                f"emb_dim mismatch: saved={state['emb_dim']} vs bank={self.emb_dim}"
            )

        # Phase 2: tokenizer 名称检查
        saved_name = state.get("tokenizer_name", "unknown")
        current_name = getattr(self.tokenizer, "name_or_path", "unknown")
        if saved_name != current_name:
            warnings.warn(
                f"Tokenizer name mismatch: saved='{saved_name}' "
                f"vs current='{current_name}'",
                UserWarning,
                stacklevel=2,
            )

        # Phase 3: capacity 处理
        saved_capacity = int(state["capacity"])
        if saved_capacity != self.capacity:
            self.capacity = saved_capacity
            self._tokens = torch.zeros(
                saved_capacity, self.fusion_length, dtype=torch.long, device=self.device
            )
            self._embs = torch.zeros(
                saved_capacity, self.emb_dim, dtype=torch.float32, device=self.device
            )
            self._deleted = torch.zeros(
                saved_capacity, dtype=torch.bool, device=self.device
            )

        # Phase 4: 恢复数据
        n = int(state["n"])
        n_deleted = int(state["n_deleted"])
        self._n = n
        self._n_deleted = n_deleted
        self._tokens[:n] = state["tokens"].to(self.device)
        self._embs[:n] = state["embs"].to(device=self.device, dtype=torch.float32)
        self._deleted[:n] = state["deleted"].to(self.device)

        # Phase 5: compact_threshold
        self.compact_threshold = state.get("compact_threshold", self.compact_threshold)

        # Phase 6: 重建 FAISS
        self._build_faiss_index()
