# TokenMemoryBank 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 TokenMemoryBank 类——TokenMem pipeline 的核心存储组件，管理 per-model 的 token_ids + embedding 条目及 FAISS 检索索引。

**Architecture:** 单类设计，内置 tokenizer + FAISS IndexIDMap(IndexFlatIP)。预分配 tensor buffer 存储 token_ids 和 embedding（借鉴老项目 FusionBank），新增软删除 + 自动 compact + FAISS 同步。所有 embedding 计算在外部完成，bank 只做存储/检索/tokenize/decode。

**Tech Stack:** Python 3.11, PyTorch 2.9, faiss-cpu, transformers 4.57, pytest

**Spec:** `docs/superpowers/specs/2026-04-26-token-memory-bank-design.md`

**参考实现:** `Reference/Memory-LoRA-old/memory_lora/fusion_bank.py`

---

## 文件结构

| 文件 | 职责 | 动作 |
|------|------|------|
| `memory_lora/__init__.py` | 包入口 | 创建 |
| `memory_lora/token_bank.py` | TokenMemoryBank 核心类 | 创建 |
| `tests/__init__.py` | 测试包入口 | 创建 |
| `tests/unit/__init__.py` | 单元测试包入口 | 创建 |
| `tests/unit/test_token_bank.py` | TokenMemoryBank 单元测试 | 创建 |

---

## Task 0: 环境准备

**Files:**
- Create: `memory_lora/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/unit/__init__.py`

- [ ] **Step 1: 安装 faiss-cpu**

```bash
conda run -n ExplicitLLM pip install faiss-cpu
```

验证：

```bash
conda run -n ExplicitLLM python -c "import faiss; print(faiss.__version__)"
```

Expected: 打印版本号，无报错。

- [ ] **Step 2: 创建包目录结构**

创建 `memory_lora/__init__.py`：

```python
"""TokenMem pipeline 核心模块。"""
```

创建 `tests/__init__.py`：

```python
```

创建 `tests/unit/__init__.py`：

```python
```

- [ ] **Step 3: 验证 pytest 可运行**

```bash
conda run -n ExplicitLLM python -m pytest tests/ -v --co
```

Expected: 输出 "no tests ran" 或空收集，无 import 错误。

- [ ] **Step 4: Commit**

```bash
git add memory_lora/__init__.py tests/__init__.py tests/unit/__init__.py
git commit -m "chore: 初始化 memory_lora 和 tests 包结构"
```

---

## Task 1: 构造函数 + `__len__`

**Files:**
- Create: `memory_lora/token_bank.py`
- Create: `tests/unit/test_token_bank.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/unit/test_token_bank.py`：

```python
"""TokenMemoryBank 单元测试。"""

import pytest
import torch
from transformers import AutoTokenizer

from memory_lora.token_bank import TokenMemoryBank


@pytest.fixture
def tokenizer():
    """加载轻量 tokenizer 用于测试（不需要模型权重）。"""
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture
def bank(tokenizer):
    """创建默认测试 bank（小容量，加速测试）。"""
    return TokenMemoryBank(
        tokenizer=tokenizer,
        capacity=100,
        fusion_length=32,
        emb_dim=64,
    )


class TestConstruction:
    """构造函数和基本属性测试。"""

    def test_empty_bank_len_is_zero(self, bank):
        assert len(bank) == 0

    def test_capacity_stored(self, bank):
        assert bank.capacity == 100

    def test_fusion_length_stored(self, bank):
        assert bank.fusion_length == 32

    def test_emb_dim_stored(self, bank):
        assert bank.emb_dim == 64

    def test_internal_tensors_shape(self, bank):
        assert bank._tokens.shape == (100, 32)
        assert bank._tokens.dtype == torch.long
        assert bank._embs.shape == (100, 64)
        assert bank._embs.dtype == torch.float32
        assert bank._deleted.shape == (100,)
        assert bank._deleted.dtype == torch.bool

    def test_faiss_index_created(self, bank):
        import faiss
        assert bank._index is not None
        assert bank._index.ntotal == 0
```

- [ ] **Step 2: 运行测试验证失败**

```bash
conda run -n ExplicitLLM python -m pytest tests/unit/test_token_bank.py::TestConstruction -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'memory_lora.token_bank'`

- [ ] **Step 3: 实现最小构造函数**

创建 `memory_lora/token_bank.py`：

```python
"""TokenMemoryBank —— TokenMem pipeline 核心存储组件。

职责：
    - 管理 per-model 的 token_ids + embedding 条目（预分配 tensor buffer）
    - 内置 FAISS 索引，add/edit/delete 时自动同步
    - 内置 tokenizer，支持 tokenize/decode（audit/migrate_to）
    - 软删除 + 自动 compact（对使用者透明）

参考：Reference/Memory-LoRA-old/memory_lora/fusion_bank.py
"""

from __future__ import annotations

from typing import List, Tuple

import faiss
import numpy as np
import torch
from torch import LongTensor, Tensor
from transformers import PreTrainedTokenizer


class TokenMemoryBank:
    """Token 级知识存储 + FAISS 检索。

    每条 entry 包含：
        - token_ids: [fusion_length] long，由内部 tokenizer 编码
        - embedding: [emb_dim] float32，由外部预计算

    内部使用 faiss.IndexIDMap(IndexFlatIP) 管理检索索引，
    所有 embedding 存入 FAISS 前 L2-normalize（等效余弦相似度）。
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        capacity: int = 1_000_000,
        fusion_length: int = 256,
        emb_dim: int = 1024,
        device: torch.device = torch.device("cpu"),
        compact_threshold: float = 0.3,
    ) -> None:
        """初始化 TokenMemoryBank。

        参数：
            tokenizer: 内置 tokenizer，用于 tokenize/decode
            capacity: 预分配最大条目数，默认 1M
            fusion_length: 每条 entry 的 token 数
            emb_dim: embedding 维度（= model hidden_dim）
            device: tensor 存储设备
            compact_threshold: 删除比例达此值时自动 compact
        """
        self.tokenizer = tokenizer
        self.capacity = capacity
        self.fusion_length = fusion_length
        self.emb_dim = emb_dim
        self.device = device
        self.compact_threshold = compact_threshold

        self._tokens = torch.zeros(
            capacity, fusion_length, dtype=torch.long, device=device
        )
        self._embs = torch.zeros(capacity, emb_dim, dtype=torch.float32, device=device)
        self._deleted = torch.zeros(capacity, dtype=torch.bool, device=device)
        self._n: int = 0
        self._n_deleted: int = 0

        base_index = faiss.IndexFlatIP(emb_dim)
        self._index = faiss.IndexIDMap(base_index)

    def __len__(self) -> int:
        """返回活跃（未删除）条目数。"""
        return self._n - self._n_deleted
```

- [ ] **Step 4: 运行测试验证通过**

```bash
conda run -n ExplicitLLM python -m pytest tests/unit/test_token_bank.py::TestConstruction -v
```

Expected: 全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add memory_lora/token_bank.py tests/unit/test_token_bank.py
git commit -m "feat: TokenMemoryBank 构造函数 + __len__"
```

---

## Task 2: `_validate` + `_tokenize_text` 内部方法

**Files:**
- Modify: `memory_lora/token_bank.py`
- Modify: `tests/unit/test_token_bank.py`

- [ ] **Step 1: 写失败测试**

在 `tests/unit/test_token_bank.py` 末尾追加：

```python
class TestValidate:
    """输入校验测试。"""

    def test_valid_input_passes(self, bank):
        emb = torch.randn(64)
        bank._validate(torch.zeros(32, dtype=torch.long), emb)

    def test_wrong_emb_dim_raises(self, bank):
        emb = torch.randn(128)
        with pytest.raises(ValueError, match="emb_dim"):
            bank._validate(torch.zeros(32, dtype=torch.long), emb)

    def test_wrong_token_length_raises(self, bank):
        emb = torch.randn(64)
        with pytest.raises(ValueError, match="fusion_length"):
            bank._validate(torch.zeros(16, dtype=torch.long), emb)

    def test_wrong_token_dtype_raises(self, bank):
        emb = torch.randn(64)
        with pytest.raises(TypeError, match="long"):
            bank._validate(torch.zeros(32, dtype=torch.float32), emb)


class TestTokenizeText:
    """内部 tokenize 方法测试。"""

    def test_output_shape_is_fusion_length(self, bank):
        ids = bank._tokenize_text("Hello world")
        assert ids.shape == (32,)
        assert ids.dtype == torch.long

    def test_long_text_truncated(self, bank):
        long_text = "word " * 1000
        ids = bank._tokenize_text(long_text)
        assert ids.shape == (32,)

    def test_short_text_padded(self, bank):
        ids = bank._tokenize_text("Hi")
        assert ids.shape == (32,)
        pad_id = bank.tokenizer.pad_token_id
        assert ids[-1].item() == pad_id
```

- [ ] **Step 2: 运行测试验证失败**

```bash
conda run -n ExplicitLLM python -m pytest tests/unit/test_token_bank.py::TestValidate tests/unit/test_token_bank.py::TestTokenizeText -v
```

Expected: FAIL — `AttributeError: 'TokenMemoryBank' object has no attribute '_validate'`

- [ ] **Step 3: 实现 `_validate` 和 `_tokenize_text`**

在 `memory_lora/token_bank.py` 的 `TokenMemoryBank` 类中，`__len__` 之后追加：

```python
    def _validate(self, token_ids: LongTensor, embedding: Tensor) -> None:
        """校验 entry 张量的 dtype 和形状。

        参数：
            token_ids: 待校验的 token 序列
            embedding: 待校验的 embedding 向量

        异常：
            TypeError: token_ids dtype 不是 torch.long
            ValueError: 形状不符合 [fusion_length] 或 [emb_dim]
        """
        if token_ids.dtype != torch.long:
            raise TypeError(f"token_ids must be long, got {token_ids.dtype}")
        if token_ids.shape != (self.fusion_length,):
            raise ValueError(
                f"token_ids shape must be [{self.fusion_length}] (fusion_length), "
                f"got {tuple(token_ids.shape)}"
            )
        if embedding.shape != (self.emb_dim,):
            raise ValueError(
                f"embedding shape must be [{self.emb_dim}] (emb_dim), "
                f"got {tuple(embedding.shape)}"
            )

    def _tokenize_text(self, text: str) -> LongTensor:
        """将文本 tokenize 并 pad/truncate 到 fusion_length。

        参数：
            text: 原始文本

        返回：
            [fusion_length] long tensor
        """
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > self.fusion_length:
            ids = ids[: self.fusion_length]
        pad_id = self.tokenizer.pad_token_id or 0
        if len(ids) < self.fusion_length:
            ids = ids + [pad_id] * (self.fusion_length - len(ids))
        return torch.tensor(ids, dtype=torch.long, device=self.device)
```

- [ ] **Step 4: 运行测试验证通过**

```bash
conda run -n ExplicitLLM python -m pytest tests/unit/test_token_bank.py::TestValidate tests/unit/test_token_bank.py::TestTokenizeText -v
```

Expected: 全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add memory_lora/token_bank.py tests/unit/test_token_bank.py
git commit -m "feat: TokenMemoryBank._validate + _tokenize_text"
```

---

## Task 3: `add` 方法

**Files:**
- Modify: `memory_lora/token_bank.py`
- Modify: `tests/unit/test_token_bank.py`

- [ ] **Step 1: 写失败测试**

在 `tests/unit/test_token_bank.py` 末尾追加：

```python
class TestAdd:
    """add 方法测试。"""

    def test_add_single_entry(self, bank):
        emb = torch.randn(64)
        ids = bank.add([("Hello world", emb)])
        assert ids == [0]
        assert len(bank) == 1

    def test_add_multiple_entries(self, bank):
        entries = [("Text one", torch.randn(64)), ("Text two", torch.randn(64))]
        ids = bank.add(entries)
        assert ids == [0, 1]
        assert len(bank) == 2

    def test_add_stores_correct_tokens(self, bank, tokenizer):
        text = "Hello world"
        emb = torch.randn(64)
        bank.add([(text, emb)])
        stored_ids = bank._tokens[0]
        expected_ids = bank._tokenize_text(text)
        assert torch.equal(stored_ids, expected_ids)

    def test_add_stores_correct_embedding(self, bank):
        emb = torch.randn(64)
        bank.add([("Hello", emb)])
        assert torch.allclose(bank._embs[0], emb)

    def test_add_updates_faiss_index(self, bank):
        emb = torch.randn(64)
        bank.add([("Hello", emb)])
        assert bank._index.ntotal == 1

    def test_add_multiple_updates_faiss_count(self, bank):
        entries = [(f"Text {i}", torch.randn(64)) for i in range(5)]
        bank.add(entries)
        assert bank._index.ntotal == 5

    def test_add_raises_when_full(self, tokenizer):
        small_bank = TokenMemoryBank(
            tokenizer=tokenizer, capacity=2, fusion_length=32, emb_dim=64
        )
        small_bank.add([("A", torch.randn(64)), ("B", torch.randn(64))])
        with pytest.raises(RuntimeError, match="capacity"):
            small_bank.add([("C", torch.randn(64))])

    def test_add_wrong_emb_dim_raises(self, bank):
        with pytest.raises(ValueError, match="emb_dim"):
            bank.add([("Hello", torch.randn(128))])
```

- [ ] **Step 2: 运行测试验证失败**

```bash
conda run -n ExplicitLLM python -m pytest tests/unit/test_token_bank.py::TestAdd -v
```

Expected: FAIL — `AttributeError: 'TokenMemoryBank' object has no attribute 'add'`

- [ ] **Step 3: 实现 `add` 方法**

在 `memory_lora/token_bank.py` 的 `_tokenize_text` 之后追加：

```python
    def add(self, entries: List[Tuple[str, Tensor]]) -> List[int]:
        """批量写入条目。

        参数：
            entries: [(text, embedding), ...]
                     text 由内部 tokenizer 编码为 token_ids
                     embedding 为外部预计算的 [emb_dim] tensor

        返回：
            分配的 entry_id 列表

        异常：
            RuntimeError: 容量不足（compact 后仍不够）
            ValueError: embedding 形状不匹配
        """
        n_new = len(entries)
        if self._n + n_new > self.capacity:
            if self._n_deleted > 0:
                self._compact()
            if self._n + n_new > self.capacity:
                raise RuntimeError(
                    f"TokenMemoryBank 容量不足: 需要 {self._n + n_new}, "
                    f"capacity={self.capacity}"
                )

        assigned_ids: List[int] = []
        embs_to_index = []
        ids_to_index = []

        for text, embedding in entries:
            token_ids = self._tokenize_text(text)
            self._validate(token_ids, embedding)

            idx = self._n
            self._tokens[idx] = token_ids
            self._embs[idx] = embedding.to(device=self.device, dtype=torch.float32)
            self._n += 1
            assigned_ids.append(idx)

            embs_to_index.append(embedding)
            ids_to_index.append(idx)

        emb_np = torch.stack(embs_to_index).detach().cpu().numpy().astype(np.float32)
        faiss.normalize_L2(emb_np)
        ids_np = np.array(ids_to_index, dtype=np.int64)
        self._index.add_with_ids(emb_np, ids_np)

        return assigned_ids
```

- [ ] **Step 4: 运行测试验证通过**

```bash
conda run -n ExplicitLLM python -m pytest tests/unit/test_token_bank.py::TestAdd -v
```

Expected: 全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add memory_lora/token_bank.py tests/unit/test_token_bank.py
git commit -m "feat: TokenMemoryBank.add 批量写入"
```

---

## Task 4: `__getitem__` + `get_token_ids`

**Files:**
- Modify: `memory_lora/token_bank.py`
- Modify: `tests/unit/test_token_bank.py`

- [ ] **Step 1: 写失败测试**

在 `tests/unit/test_token_bank.py` 末尾追加：

```python
class TestGetItem:
    """__getitem__ 测试。"""

    def test_getitem_returns_clone(self, bank):
        emb = torch.randn(64)
        bank.add([("Hello", emb)])
        tok, e = bank[0]
        assert tok.shape == (32,)
        assert e.shape == (64,)
        tok[0] = -1
        assert bank._tokens[0, 0].item() != -1

    def test_getitem_out_of_range_raises(self, bank):
        with pytest.raises(IndexError):
            _ = bank[0]

    def test_getitem_deleted_raises(self, bank):
        bank.add([("Hello", torch.randn(64))])
        bank._deleted[0] = True
        bank._n_deleted = 1
        with pytest.raises(ValueError, match="已删除"):
            _ = bank[0]


class TestGetTokenIds:
    """get_token_ids 批量获取测试。"""

    def test_1d_input(self, bank):
        bank.add([("A", torch.randn(64)), ("B", torch.randn(64))])
        entry_ids = torch.tensor([0, 1])
        result = bank.get_token_ids(entry_ids)
        assert result.shape == (2, 32)

    def test_2d_input(self, bank):
        bank.add([("A", torch.randn(64)), ("B", torch.randn(64)), ("C", torch.randn(64))])
        entry_ids = torch.tensor([[0, 1], [1, 2]])
        result = bank.get_token_ids(entry_ids)
        assert result.shape == (2, 2, 32)
```

- [ ] **Step 2: 运行测试验证失败**

```bash
conda run -n ExplicitLLM python -m pytest tests/unit/test_token_bank.py::TestGetItem tests/unit/test_token_bank.py::TestGetTokenIds -v
```

Expected: FAIL — `TypeError: 'TokenMemoryBank' object is not subscriptable`

- [ ] **Step 3: 实现 `__getitem__` 和 `get_token_ids`**

在 `memory_lora/token_bank.py` 的 `add` 之后追加：

```python
    def __getitem__(self, entry_id: int) -> Tuple[LongTensor, Tensor]:
        """返回 (token_ids, embedding) 的 clone。

        参数：
            entry_id: 条目索引

        返回：
            (token_ids clone, embedding clone) 元组

        异常：
            IndexError: entry_id 越界
            ValueError: entry_id 已删除
        """
        if not (0 <= entry_id < self._n):
            raise IndexError(f"entry_id {entry_id} out of range [0, {self._n})")
        if self._deleted[entry_id]:
            raise ValueError(f"entry_id {entry_id} 已删除")
        return self._tokens[entry_id].clone(), self._embs[entry_id].clone()

    def get_token_ids(self, entry_ids: LongTensor) -> LongTensor:
        """批量获取 token_ids，供推理时 frozen LLM 前向计算 KV 用。

        参数：
            entry_ids: [B] 或 [B, k] 检索返回的 entry id

        返回：
            [B, fusion_length] 或 [B, k, fusion_length]
        """
        original_shape = entry_ids.shape
        flat_ids = entry_ids.reshape(-1)
        result = self._tokens[flat_ids]
        return result.view(*original_shape, self.fusion_length)
```

- [ ] **Step 4: 运行测试验证通过**

```bash
conda run -n ExplicitLLM python -m pytest tests/unit/test_token_bank.py::TestGetItem tests/unit/test_token_bank.py::TestGetTokenIds -v
```

Expected: 全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add memory_lora/token_bank.py tests/unit/test_token_bank.py
git commit -m "feat: TokenMemoryBank.__getitem__ + get_token_ids"
```

---

## Task 5: `retrieve` 方法

**Files:**
- Modify: `memory_lora/token_bank.py`
- Modify: `tests/unit/test_token_bank.py`

- [ ] **Step 1: 写失败测试**

在 `tests/unit/test_token_bank.py` 末尾追加：

```python
class TestRetrieve:
    """FAISS 检索测试。"""

    def test_retrieve_top1(self, bank):
        target_emb = torch.tensor([1.0] + [0.0] * 63)
        noise_emb = torch.randn(64)
        bank.add([("target", target_emb), ("noise", noise_emb)])

        query = torch.tensor([1.0] + [0.0] * 63).unsqueeze(0)
        ids, scores = bank.retrieve(query, k=1)
        assert ids.shape == (1, 1)
        assert ids[0, 0].item() == 0

    def test_retrieve_topk(self, bank):
        entries = [(f"doc{i}", torch.randn(64)) for i in range(10)]
        bank.add(entries)

        query = bank._embs[3].unsqueeze(0)
        ids, scores = bank.retrieve(query, k=3)
        assert ids.shape == (1, 3)
        assert 3 in ids[0].tolist()

    def test_retrieve_batch(self, bank):
        entries = [(f"doc{i}", torch.randn(64)) for i in range(5)]
        bank.add(entries)

        queries = torch.stack([bank._embs[0], bank._embs[2]])
        ids, scores = bank.retrieve(queries, k=2)
        assert ids.shape == (2, 2)

    def test_retrieve_empty_raises(self, bank):
        query = torch.randn(1, 64)
        with pytest.raises(RuntimeError, match="空"):
            bank.retrieve(query, k=1)
```

- [ ] **Step 2: 运行测试验证失败**

```bash
conda run -n ExplicitLLM python -m pytest tests/unit/test_token_bank.py::TestRetrieve -v
```

Expected: FAIL — `AttributeError: 'TokenMemoryBank' object has no attribute 'retrieve'`

- [ ] **Step 3: 实现 `retrieve`**

在 `memory_lora/token_bank.py` 的 `get_token_ids` 之后追加：

```python
    def retrieve(self, query_emb: Tensor, k: int = 1) -> Tuple[LongTensor, Tensor]:
        """FAISS top-k 检索。

        参数：
            query_emb: [B, emb_dim] 查询向量
            k: top-k 数量

        返回：
            (entry_ids [B, k], scores [B, k])

        异常：
            RuntimeError: bank 为空
            ValueError: k 超出活跃条目数
        """
        n_alive = self._n - self._n_deleted
        if n_alive == 0:
            raise RuntimeError("TokenMemoryBank 为空，无法检索")
        if k > n_alive:
            raise ValueError(f"k={k} 超出活跃条目数 {n_alive}")

        q_np = query_emb.detach().cpu().numpy().astype(np.float32)
        if q_np.ndim == 1:
            q_np = q_np.reshape(1, -1)
        faiss.normalize_L2(q_np)

        scores_np, ids_np = self._index.search(q_np, k)
        return (
            torch.from_numpy(ids_np).long(),
            torch.from_numpy(scores_np).float(),
        )
```

- [ ] **Step 4: 运行测试验证通过**

```bash
conda run -n ExplicitLLM python -m pytest tests/unit/test_token_bank.py::TestRetrieve -v
```

Expected: 全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add memory_lora/token_bank.py tests/unit/test_token_bank.py
git commit -m "feat: TokenMemoryBank.retrieve FAISS top-k 检索"
```

---

## Task 6: `delete` + `_maybe_compact` + `_compact`

**Files:**
- Modify: `memory_lora/token_bank.py`
- Modify: `tests/unit/test_token_bank.py`

- [ ] **Step 1: 写失败测试**

在 `tests/unit/test_token_bank.py` 末尾追加：

```python
class TestDelete:
    """软删除测试。"""

    def test_delete_reduces_len(self, bank):
        bank.add([("A", torch.randn(64)), ("B", torch.randn(64))])
        bank.delete(0)
        assert len(bank) == 1

    def test_delete_marks_deleted(self, bank):
        bank.add([("A", torch.randn(64))])
        bank.delete(0)
        assert bank._deleted[0] is True or bank._deleted[0].item() is True

    def test_delete_removes_from_faiss(self, bank):
        bank.add([("A", torch.randn(64)), ("B", torch.randn(64))])
        bank.delete(0)
        assert bank._index.ntotal == 1

    def test_delete_invalid_id_raises(self, bank):
        with pytest.raises(IndexError):
            bank.delete(99)

    def test_delete_already_deleted_raises(self, bank):
        bank.add([("A", torch.randn(64))])
        bank.delete(0)
        with pytest.raises(ValueError, match="已删除"):
            bank.delete(0)

    def test_getitem_after_delete_raises(self, bank):
        bank.add([("A", torch.randn(64))])
        bank.delete(0)
        with pytest.raises(ValueError, match="已删除"):
            _ = bank[0]


class TestCompact:
    """自动 compact 测试。"""

    def test_compact_reclaims_space(self, tokenizer):
        small_bank = TokenMemoryBank(
            tokenizer=tokenizer, capacity=10, fusion_length=32, emb_dim=64,
            compact_threshold=0.3,
        )
        entries = [(f"doc{i}", torch.randn(64)) for i in range(6)]
        small_bank.add(entries)
        assert small_bank._n == 6

        small_bank.delete(0)
        small_bank.delete(1)
        assert len(small_bank) == 4
        assert small_bank._n == 6

        small_bank._compact()
        assert small_bank._n == 4
        assert small_bank._n_deleted == 0
        assert len(small_bank) == 4
        assert small_bank._index.ntotal == 4

    def test_auto_compact_triggered(self, tokenizer):
        small_bank = TokenMemoryBank(
            tokenizer=tokenizer, capacity=10, fusion_length=32, emb_dim=64,
            compact_threshold=0.3,
        )
        entries = [(f"doc{i}", torch.randn(64)) for i in range(10)]
        small_bank.add(entries)

        small_bank.delete(0)
        small_bank.delete(1)
        assert small_bank._n == 10

        small_bank.delete(2)
        assert small_bank._n_deleted == 0
        assert small_bank._n == 7

    def test_add_triggers_compact_when_needed(self, tokenizer):
        small_bank = TokenMemoryBank(
            tokenizer=tokenizer, capacity=5, fusion_length=32, emb_dim=64,
            compact_threshold=0.5,
        )
        entries = [(f"doc{i}", torch.randn(64)) for i in range(5)]
        small_bank.add(entries)

        small_bank.delete(0)
        small_bank.delete(1)
        small_bank.delete(2)

        new_entries = [("new", torch.randn(64))]
        ids = small_bank.add(new_entries)
        assert len(ids) == 1
        assert len(small_bank) == 3
```

- [ ] **Step 2: 运行测试验证失败**

```bash
conda run -n ExplicitLLM python -m pytest tests/unit/test_token_bank.py::TestDelete tests/unit/test_token_bank.py::TestCompact -v
```

Expected: FAIL — `AttributeError: 'TokenMemoryBank' object has no attribute 'delete'`

- [ ] **Step 3: 实现 `delete` + `_maybe_compact` + `_compact` + `_build_faiss_index`**

在 `memory_lora/token_bank.py` 的 `retrieve` 之后追加：

```python
    def delete(self, entry_id: int) -> None:
        """软删除一条 entry。

        参数：
            entry_id: 要删除的条目索引

        异常：
            IndexError: entry_id 越界
            ValueError: entry_id 已删除
        """
        if not (0 <= entry_id < self._n):
            raise IndexError(f"entry_id {entry_id} out of range [0, {self._n})")
        if self._deleted[entry_id]:
            raise ValueError(f"entry_id {entry_id} 已删除")

        self._deleted[entry_id] = True
        self._n_deleted += 1
        self._index.remove_ids(np.array([entry_id], dtype=np.int64))
        self._maybe_compact()

    def _maybe_compact(self) -> None:
        """检查是否需要自动 compact。"""
        if self._n > 0 and self._n_deleted / self._n >= self.compact_threshold:
            self._compact()

    def _compact(self) -> None:
        """将所有未删除条目前移压缩，重建 FAISS 索引。"""
        alive_mask = ~self._deleted[: self._n]
        n_alive = int(alive_mask.sum().item())

        if n_alive == 0:
            self._n = 0
            self._n_deleted = 0
            self._deleted.zero_()
            self._build_faiss_index()
            return

        alive_tokens = self._tokens[: self._n][alive_mask].clone()
        alive_embs = self._embs[: self._n][alive_mask].clone()

        self._tokens[: n_alive] = alive_tokens
        self._embs[: n_alive] = alive_embs
        self._tokens[n_alive : self._n].zero_()
        self._embs[n_alive : self._n].zero_()

        self._deleted.zero_()
        self._n = n_alive
        self._n_deleted = 0

        self._build_faiss_index()

    def _build_faiss_index(self) -> None:
        """从当前未删除的 _embs 重建 FAISS 索引。"""
        base_index = faiss.IndexFlatIP(self.emb_dim)
        self._index = faiss.IndexIDMap(base_index)

        if self._n == 0:
            return

        alive_mask = ~self._deleted[: self._n]
        alive_indices = torch.where(alive_mask)[0].numpy().astype(np.int64)

        if len(alive_indices) == 0:
            return

        embs_np = self._embs[alive_indices].detach().cpu().numpy().astype(np.float32)
        faiss.normalize_L2(embs_np)
        self._index.add_with_ids(embs_np, alive_indices)
```

- [ ] **Step 4: 运行测试验证通过**

```bash
conda run -n ExplicitLLM python -m pytest tests/unit/test_token_bank.py::TestDelete tests/unit/test_token_bank.py::TestCompact -v
```

Expected: 全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add memory_lora/token_bank.py tests/unit/test_token_bank.py
git commit -m "feat: TokenMemoryBank.delete + 软删除 + 自动 compact"
```

---

## Task 7: `edit` 方法

**Files:**
- Modify: `memory_lora/token_bank.py`
- Modify: `tests/unit/test_token_bank.py`

- [ ] **Step 1: 写失败测试**

在 `tests/unit/test_token_bank.py` 末尾追加：

```python
class TestEdit:
    """edit 方法测试。"""

    def test_edit_updates_tokens(self, bank, tokenizer):
        bank.add([("old text", torch.randn(64))])
        new_emb = torch.randn(64)
        bank.edit(0, "new text", new_emb)

        expected_ids = bank._tokenize_text("new text")
        assert torch.equal(bank._tokens[0], expected_ids)

    def test_edit_updates_embedding(self, bank):
        bank.add([("old", torch.randn(64))])
        new_emb = torch.randn(64)
        bank.edit(0, "new", new_emb)
        assert torch.allclose(bank._embs[0], new_emb)

    def test_edit_updates_faiss(self, bank):
        target_emb = torch.tensor([1.0] + [0.0] * 63)
        bank.add([("old", torch.randn(64))])
        bank.edit(0, "new", target_emb)

        query = torch.tensor([1.0] + [0.0] * 63).unsqueeze(0)
        ids, scores = bank.retrieve(query, k=1)
        assert ids[0, 0].item() == 0
        assert scores[0, 0].item() > 0.9

    def test_edit_invalid_id_raises(self, bank):
        with pytest.raises(IndexError):
            bank.edit(99, "text", torch.randn(64))

    def test_edit_deleted_raises(self, bank):
        bank.add([("A", torch.randn(64))])
        bank.delete(0)
        with pytest.raises(ValueError, match="已删除"):
            bank.edit(0, "new", torch.randn(64))

    def test_edit_preserves_len(self, bank):
        bank.add([("A", torch.randn(64)), ("B", torch.randn(64))])
        bank.edit(0, "C", torch.randn(64))
        assert len(bank) == 2
```

- [ ] **Step 2: 运行测试验证失败**

```bash
conda run -n ExplicitLLM python -m pytest tests/unit/test_token_bank.py::TestEdit -v
```

Expected: FAIL — `AttributeError: 'TokenMemoryBank' object has no attribute 'edit'`

- [ ] **Step 3: 实现 `edit`**

在 `memory_lora/token_bank.py` 的 `delete` 之前追加：

```python
    def edit(self, entry_id: int, text: str, embedding: Tensor) -> None:
        """原地编辑一条 entry。

        参数：
            entry_id: 要编辑的条目索引
            text: 新文本
            embedding: 新的外部预计算 embedding [emb_dim]

        异常：
            IndexError: entry_id 越界
            ValueError: entry_id 已删除，或 embedding 形状不匹配
        """
        if not (0 <= entry_id < self._n):
            raise IndexError(f"entry_id {entry_id} out of range [0, {self._n})")
        if self._deleted[entry_id]:
            raise ValueError(f"entry_id {entry_id} 已删除")

        token_ids = self._tokenize_text(text)
        self._validate(token_ids, embedding)

        self._tokens[entry_id] = token_ids
        self._embs[entry_id] = embedding.to(device=self.device, dtype=torch.float32)

        self._index.remove_ids(np.array([entry_id], dtype=np.int64))
        emb_np = embedding.detach().cpu().numpy().astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(emb_np)
        self._index.add_with_ids(emb_np, np.array([entry_id], dtype=np.int64))
```

- [ ] **Step 4: 运行测试验证通过**

```bash
conda run -n ExplicitLLM python -m pytest tests/unit/test_token_bank.py::TestEdit -v
```

Expected: 全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add memory_lora/token_bank.py tests/unit/test_token_bank.py
git commit -m "feat: TokenMemoryBank.edit 原地编辑"
```

---

## Task 8: `audit` + `migrate_to`

**Files:**
- Modify: `memory_lora/token_bank.py`
- Modify: `tests/unit/test_token_bank.py`

- [ ] **Step 1: 写失败测试**

在 `tests/unit/test_token_bank.py` 末尾追加：

```python
class TestAudit:
    """audit 方法测试。"""

    def test_audit_returns_readable_text(self, bank):
        bank.add([("The quick brown fox", torch.randn(64))])
        text = bank.audit(0)
        assert isinstance(text, str)
        assert "quick" in text or "brown" in text

    def test_audit_invalid_id_raises(self, bank):
        with pytest.raises(IndexError):
            bank.audit(99)

    def test_audit_deleted_raises(self, bank):
        bank.add([("A", torch.randn(64))])
        bank.delete(0)
        with pytest.raises(ValueError, match="已删除"):
            bank.audit(0)


class TestMigrateTo:
    """migrate_to 方法测试。"""

    def test_migrate_returns_all_active_texts(self, bank):
        bank.add([
            ("First document", torch.randn(64)),
            ("Second document", torch.randn(64)),
            ("Third document", torch.randn(64)),
        ])
        bank.delete(1)

        texts = bank.migrate_to()
        assert len(texts) == 2
        assert any("First" in t for t in texts)
        assert any("Third" in t for t in texts)

    def test_migrate_empty_bank(self, bank):
        texts = bank.migrate_to()
        assert texts == []
```

- [ ] **Step 2: 运行测试验证失败**

```bash
conda run -n ExplicitLLM python -m pytest tests/unit/test_token_bank.py::TestAudit tests/unit/test_token_bank.py::TestMigrateTo -v
```

Expected: FAIL — `AttributeError: 'TokenMemoryBank' object has no attribute 'audit'`

- [ ] **Step 3: 实现 `audit` 和 `migrate_to`**

在 `memory_lora/token_bank.py` 的 `edit` 之后、`delete` 之前追加：

```python
    def audit(self, entry_id: int) -> str:
        """解码 entry 为人类可读文本。

        参数：
            entry_id: 条目索引

        返回：
            解码后的文本

        异常：
            IndexError: entry_id 越界
            ValueError: entry_id 已删除
        """
        if not (0 <= entry_id < self._n):
            raise IndexError(f"entry_id {entry_id} out of range [0, {self._n})")
        if self._deleted[entry_id]:
            raise ValueError(f"entry_id {entry_id} 已删除")
        return self.tokenizer.decode(
            self._tokens[entry_id].tolist(), skip_special_tokens=True
        )

    def migrate_to(self) -> List[str]:
        """导出所有未删除条目的原始文本。

        返回：
            解码后的文本列表（调用方用新 tokenizer + embedding 构建新 bank）
        """
        texts: List[str] = []
        for i in range(self._n):
            if not self._deleted[i]:
                text = self.tokenizer.decode(
                    self._tokens[i].tolist(), skip_special_tokens=True
                )
                texts.append(text)
        return texts
```

- [ ] **Step 4: 运行测试验证通过**

```bash
conda run -n ExplicitLLM python -m pytest tests/unit/test_token_bank.py::TestAudit tests/unit/test_token_bank.py::TestMigrateTo -v
```

Expected: 全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add memory_lora/token_bank.py tests/unit/test_token_bank.py
git commit -m "feat: TokenMemoryBank.audit + migrate_to"
```

---

## Task 9: `save` + `load` 持久化

**Files:**
- Modify: `memory_lora/token_bank.py`
- Modify: `tests/unit/test_token_bank.py`

- [ ] **Step 1: 写失败测试**

在 `tests/unit/test_token_bank.py` 末尾追加：

```python
class TestPersistence:
    """save/load 持久化测试。"""

    def test_save_load_roundtrip(self, bank, tokenizer, tmp_path):
        entries = [("Hello world", torch.randn(64)), ("Foo bar", torch.randn(64))]
        bank.add(entries)

        save_path = str(tmp_path / "bank.pt")
        bank.save(save_path)

        bank2 = TokenMemoryBank(
            tokenizer=tokenizer, capacity=100, fusion_length=32, emb_dim=64
        )
        bank2.load(save_path)

        assert len(bank2) == 2
        assert bank2._n == 2
        assert torch.equal(bank2._tokens[:2], bank._tokens[:2])
        assert torch.allclose(bank2._embs[:2], bank._embs[:2])

    def test_save_load_with_deletions(self, bank, tokenizer, tmp_path):
        entries = [(f"doc{i}", torch.randn(64)) for i in range(5)]
        bank.add(entries)
        bank.delete(1)
        bank.delete(3)

        save_path = str(tmp_path / "bank_del.pt")
        bank.save(save_path)

        bank2 = TokenMemoryBank(
            tokenizer=tokenizer, capacity=100, fusion_length=32, emb_dim=64
        )
        bank2.load(save_path)

        assert len(bank2) == 3
        assert bank2._n == 5
        assert bank2._n_deleted == 2
        assert bank2._deleted[1].item() is True
        assert bank2._deleted[3].item() is True

    def test_load_rebuilds_faiss(self, bank, tokenizer, tmp_path):
        entries = [("A", torch.randn(64)), ("B", torch.randn(64))]
        bank.add(entries)

        save_path = str(tmp_path / "bank_faiss.pt")
        bank.save(save_path)

        bank2 = TokenMemoryBank(
            tokenizer=tokenizer, capacity=100, fusion_length=32, emb_dim=64
        )
        bank2.load(save_path)
        assert bank2._index.ntotal == 2

    def test_load_mismatched_fusion_length_raises(self, bank, tokenizer, tmp_path):
        bank.add([("A", torch.randn(64))])
        save_path = str(tmp_path / "bank_mismatch.pt")
        bank.save(save_path)

        bank_wrong = TokenMemoryBank(
            tokenizer=tokenizer, capacity=100, fusion_length=64, emb_dim=64
        )
        with pytest.raises(ValueError, match="fusion_length"):
            bank_wrong.load(save_path)

    def test_load_mismatched_emb_dim_raises(self, bank, tokenizer, tmp_path):
        bank.add([("A", torch.randn(64))])
        save_path = str(tmp_path / "bank_mismatch2.pt")
        bank.save(save_path)

        bank_wrong = TokenMemoryBank(
            tokenizer=tokenizer, capacity=100, fusion_length=32, emb_dim=128
        )
        with pytest.raises(ValueError, match="emb_dim"):
            bank_wrong.load(save_path)

    def test_load_different_capacity_rebuilds_buffer(self, bank, tokenizer, tmp_path):
        bank.add([("A", torch.randn(64))])
        save_path = str(tmp_path / "bank_cap.pt")
        bank.save(save_path)

        bank2 = TokenMemoryBank(
            tokenizer=tokenizer, capacity=50, fusion_length=32, emb_dim=64
        )
        bank2.load(save_path)
        assert bank2.capacity == 100
        assert len(bank2) == 1
```

- [ ] **Step 2: 运行测试验证失败**

```bash
conda run -n ExplicitLLM python -m pytest tests/unit/test_token_bank.py::TestPersistence -v
```

Expected: FAIL — `AttributeError: 'TokenMemoryBank' object has no attribute 'save'`

- [ ] **Step 3: 实现 `save` 和 `load`**

在 `memory_lora/token_bank.py` 的 `_build_faiss_index` 之后追加：

```python
    def save(self, path: str) -> None:
        """序列化 bank 到单文件。

        FAISS 索引不持久化，load 时从 _embs 重建。

        参数：
            path: 保存路径
        """
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
                "tokenizer_name": self.tokenizer.name_or_path,
                "compact_threshold": self.compact_threshold,
            },
            path,
        )

    def load(self, path: str) -> None:
        """从 save 文件恢复状态。

        参数：
            path: 状态文件路径

        异常：
            ValueError: fusion_length / emb_dim 不一致
        """
        import warnings

        state = torch.load(path, map_location=self.device, weights_only=True)

        if state["fusion_length"] != self.fusion_length:
            raise ValueError(
                f"fusion_length 不匹配: 文件={state['fusion_length']} vs "
                f"bank={self.fusion_length}"
            )
        if state["emb_dim"] != self.emb_dim:
            raise ValueError(
                f"emb_dim 不匹配: 文件={state['emb_dim']} vs bank={self.emb_dim}"
            )

        saved_tokenizer = state.get("tokenizer_name", "")
        if saved_tokenizer and saved_tokenizer != self.tokenizer.name_or_path:
            warnings.warn(
                f"tokenizer 不匹配: 文件={saved_tokenizer} vs "
                f"bank={self.tokenizer.name_or_path}",
                stacklevel=2,
            )

        saved_capacity = int(state["capacity"])
        if saved_capacity != self.capacity:
            self.capacity = saved_capacity
            self._tokens = torch.zeros(
                saved_capacity, self.fusion_length,
                dtype=torch.long, device=self.device,
            )
            self._embs = torch.zeros(
                saved_capacity, self.emb_dim,
                dtype=torch.float32, device=self.device,
            )
            self._deleted = torch.zeros(
                saved_capacity, dtype=torch.bool, device=self.device,
            )

        n = int(state["n"])
        self._n = n
        self._n_deleted = int(state["n_deleted"])
        self.compact_threshold = float(state.get("compact_threshold", 0.3))
        self._tokens[:n] = state["tokens"].to(self.device)
        self._embs[:n] = state["embs"].to(device=self.device, dtype=torch.float32)
        self._deleted[:n] = state["deleted"].to(self.device)

        self._build_faiss_index()
```

- [ ] **Step 4: 运行测试验证通过**

```bash
conda run -n ExplicitLLM python -m pytest tests/unit/test_token_bank.py::TestPersistence -v
```

Expected: 全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add memory_lora/token_bank.py tests/unit/test_token_bank.py
git commit -m "feat: TokenMemoryBank.save/load 持久化"
```

---

## Task 10: 全量测试 + 代码质量检查

**Files:**
- Modify: `memory_lora/token_bank.py` (如需修复)
- Modify: `tests/unit/test_token_bank.py` (如需修复)

- [ ] **Step 1: 运行全量测试**

```bash
conda run -n ExplicitLLM python -m pytest tests/unit/test_token_bank.py -v --tb=short
```

Expected: 全部 PASS（9 个测试类，~30 个测试用例）。

- [ ] **Step 2: 运行覆盖率检查**

```bash
conda run -n ExplicitLLM python -m pytest tests/unit/test_token_bank.py --cov=memory_lora --cov-report=term-missing
```

Expected: `token_bank.py` 覆盖率 ≥ 80%。

- [ ] **Step 3: 代码格式化 + lint**

```bash
conda run -n ExplicitLLM ruff format memory_lora/token_bank.py tests/unit/test_token_bank.py
conda run -n ExplicitLLM ruff check memory_lora/token_bank.py tests/unit/test_token_bank.py --fix
```

Expected: 无错误或已自动修复。

- [ ] **Step 4: Commit**

```bash
git add memory_lora/token_bank.py tests/unit/test_token_bank.py
git commit -m "test: TokenMemoryBank 全量测试通过 + lint 清理"
```
