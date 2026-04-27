# TokenMemoryBank 设计文档

**日期**: 2026-04-26
**目标文件**: `memory_lora/token_bank.py`
**参考实现**: `Reference/Memory-LoRA-old/memory_lora/fusion_bank.py`

---

## 1. 概述

TokenMemoryBank 是 TokenMem pipeline 的核心存储组件，负责管理 per-model 的 token 级知识条目（token_ids + embedding）以及 FAISS 检索索引。相比老项目的 FusionBank，主要变化：

- **合并设计**: 内置 tokenizer + FAISS 索引，一个类完成存储/检索/审计/迁移
- **软删除 + 自动 compact**: 对使用者透明的删除机制
- **去掉 LLMLingua 依赖**: 直接 tokenize，不压缩
- **不计算 embedding**: embedding 由外部预计算传入

### 设计决策记录

| 决策点 | 选项 | 决定 | 理由 |
|--------|------|------|------|
| 存储 vs 文本处理 | 分层 / 合并 | 合并 | 一个对象完成一切，使用简单 |
| FAISS 位置 | 内置 / 外置 | 内置 | add/edit/delete 自动同步索引，避免不一致 |
| 删除策略 | 软 / 硬 / 软+compact | 软+自动compact | 日常 O(1) 删除，阈值时批量回收 |
| embedding 计算 | 内部 / 外部 | 外部 | embedding 需 GPU 模型，bank 不持有 |
| tokenizer | 内置 / 外部传入 | 内置 | audit/migrate_to 需要 decode 能力 |
| migrate_to 返回 | 新 bank / List[str] | List[str] | 职责清晰，调用方自行构建新 bank |
| 预分配容量 | 60K / 1M | 1M | 预留充足空间 |

---

## 2. 数据结构

### 2.1 构造参数

```python
def __init__(
    self,
    tokenizer: PreTrainedTokenizer,
    capacity: int = 1_000_000,
    fusion_length: int = 256,
    emb_dim: int,
    device: torch.device = torch.device("cpu"),
    compact_threshold: float = 0.3,
):
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `tokenizer` | PreTrainedTokenizer | 内置，用于 tokenize/decode |
| `capacity` | int | 预分配容量，默认 1M |
| `fusion_length` | int | 每条 entry 的 token 数，默认 256 |
| `emb_dim` | int | embedding 维度（= model hidden_dim） |
| `device` | torch.device | tensor 存储设备 |
| `compact_threshold` | float | 删除比例达此值时自动 compact，默认 0.3 |

### 2.2 内部存储

```python
_tokens: Tensor      # [capacity, fusion_length], dtype=long
_embs: Tensor        # [capacity, emb_dim], dtype=float32
_deleted: Tensor     # [capacity], dtype=bool
_n: int              # 已写入条目数（含已删除）
_n_deleted: int      # 已删除条目计数
_index: faiss.IndexIDMap    # IndexIDMap(IndexFlatIP(emb_dim))，支持 id-based 操作
```

### 2.3 与 FusionBank 对比

| 要素 | FusionBank（老） | TokenMemoryBank（新） |
|------|-----------------|---------------------|
| `_tokens` / `_embs` | ✅ [capacity, fusion_length] / [capacity, emb_dim] | ✅ 相同模式保留 |
| `_deleted` | ❌ 无 | ✅ 新增 bool 位图 |
| `tokenizer` | ❌ 在 BankWriter 中 | ✅ 内置 |
| 检索 | 手写 cosine（L2-normalize + matmul） | faiss.IndexFlatIP |
| `dtype` 参数 | 可配 | float32（MVP 简化） |

---

## 3. 接口设计

### 3.1 CRUD

#### `add(entries: List[Tuple[str, Tensor]]) -> List[int]`

批量写入条目。

- **输入**: `[(text, embedding), ...]`，text 由内部 tokenizer 编码，embedding 为外部预计算
- **输出**: 分配的 entry_id 列表
- **流程**:
  1. 容量检查：若 `_n + len(entries) > capacity`，先尝试 `_compact()` 回收空间；compact 后仍不够则 raise `RuntimeError`
  2. 逐条: `tokenizer.encode(text)` → pad/truncate 到 `fusion_length` → 存入 `_tokens[_n]`
  3. 逐条: embedding 存入 `_embs[_n]`
  4. 批量: L2-normalize embeddings → `_index.add()`
  5. 返回 entry_id 列表
- **异常**: `RuntimeError` 容量满（compact 后仍不够）

#### `edit(entry_id: int, text: str, embedding: Tensor) -> None`

原地编辑一条 entry。

- **流程**:
  1. 校验 entry_id 有效（`0 <= entry_id < _n`）且未删除
  2. 重新 tokenize text → 覆盖 `_tokens[entry_id]`
  3. 覆盖 `_embs[entry_id]`
  4. FAISS: `remove_ids([entry_id])` + `add(new_emb)`（使用 `IndexIDMap` 包装以支持 id-based 操作）
- **异常**: `IndexError` 越界, `ValueError` 已删除

#### `delete(entry_id: int) -> None`

软删除，用户无感。

- **流程**:
  1. 校验 entry_id 有效且未删除
  2. `_deleted[entry_id] = True`
  3. `_n_deleted += 1`
  4. FAISS: `remove_ids([entry_id])`
  5. `_maybe_compact()` 检查是否触发自动 compact
- **异常**: `IndexError` 越界, `ValueError` 已删除

#### `audit(entry_id: int) -> str`

解码 entry 为人类可读文本。

- **流程**:
  1. 校验 entry_id 有效且未删除
  2. `tokenizer.decode(_tokens[entry_id], skip_special_tokens=True)`
  3. 返回文本

#### `migrate_to() -> List[str]`

导出所有未删除条目的原始文本。

- **流程**:
  1. 遍历 `[0, _n)` 中 `_deleted[i] == False` 的条目
  2. `tokenizer.decode` 每条 token_ids
  3. 返回文本列表

### 3.2 检索

#### `retrieve(query_emb: Tensor, k: int = 1) -> Tuple[LongTensor, Tensor]`

FAISS top-k 检索。

- **输入**: `query_emb [B, emb_dim]`，`k` top-k 数量
- **输出**: `(entry_ids [B, k], scores [B, k])`
- **流程**:
  1. L2-normalize `query_emb`
  2. `_index.search(query_emb, k)` → scores, ids
  3. 返回 `(ids, scores)`
- **异常**: `RuntimeError` bank 为空, `ValueError` k 超出活跃条目数

### 3.3 访问

#### `__len__() -> int`

返回活跃（未删除）条目数：`_n - _n_deleted`。

#### `__getitem__(entry_id: int) -> Tuple[LongTensor, Tensor]`

返回 `(token_ids, embedding)` 的 clone。已删除则 raise `ValueError`。
借鉴 `FusionBank.__getitem__`。

#### `get_token_ids(entry_ids: LongTensor) -> LongTensor`

批量获取 token_ids，供推理时 frozen LLM 前向计算 KV 用。

- **输入**: `entry_ids [B]` 或 `[B, k]`
- **输出**: `[B, fusion_length]` 或 `[B, k, fusion_length]`
- **用途**: TokenMem 推理流程的关键接口：`retrieve → get_token_ids → frozen LLM forward → KV → cross-attention`

### 3.4 持久化

#### `save(path: str) -> None`

```python
torch.save({
    "capacity": self.capacity,
    "fusion_length": self.fusion_length,
    "emb_dim": self.emb_dim,
    "n": self._n,
    "n_deleted": self._n_deleted,
    "tokens": self._tokens[:self._n].cpu(),
    "embs": self._embs[:self._n].cpu(),
    "deleted": self._deleted[:self._n].cpu(),
    "tokenizer_name": self.tokenizer.name_or_path,
    "compact_threshold": self.compact_threshold,
}, path)
```

FAISS 索引不持久化，load 时从 `_embs` 重建。

#### `load(path: str) -> None`

- 校验 `fusion_length` / `emb_dim` 一致性（不一致 raise `ValueError`）
- `tokenizer_name` 不一致时 warning
- 恢复所有内部状态
- 从未删除的 `_embs` 重建 FAISS 索引
- 若保存的 capacity 与当前不同 → 重建 buffer（借鉴 `FusionBank.load_state`）

---

## 4. 软删除 + 自动 Compact

### 4.1 触发条件

```python
def _maybe_compact(self) -> None:
    if self._n > 0 and self._n_deleted / self._n >= self.compact_threshold:
        self._compact()
```

### 4.2 Compact 流程

1. 收集未删除条目 mask: `alive_mask = ~_deleted[:_n]`
2. 前移: `_tokens[:N_alive] = _tokens[alive_mask]`
3. 前移: `_embs[:N_alive] = _embs[alive_mask]`
4. 清零尾部区域
5. 重置: `_deleted[:] = False`, `_n = N_alive`, `_n_deleted = 0`
6. 重建 FAISS: `_index.reset()` → `_index.add(L2-normalized alive_embs)`

### 4.3 entry_id 语义

- entry_id 是物理索引，compact 后重编号
- 设计假设：调用方不跨 compact 边界长期缓存 entry_id
- 实际场景中 retrieve 返回的 id 即用即弃

---

## 5. FAISS 索引管理

### 5.1 索引类型

使用 `faiss.IndexIDMap(faiss.IndexFlatIP(emb_dim))` 包装：
- `IndexFlatIP`: 内积搜索（L2-normalize 后等效余弦），brute-force 精确
- `IndexIDMap`: 支持自定义 id 映射 + `remove_ids()` 操作

### 5.2 同步策略

| 操作 | FAISS 动作 | 说明 |
|------|-----------|------|
| `add` | `index.add_with_ids(embs, ids)` | 使用 entry_id 作为 FAISS id |
| `edit` | `remove_ids` + `add_with_ids` | 先删后加 |
| `delete` | `remove_ids` | 仅从索引移除 |
| `compact` | `index.reset()` + 全量 `add_with_ids` | 重建，新 id |

### 5.3 归一化约定

所有存入 FAISS 的 embedding 先 L2-normalize，查询时同样 normalize，使 IndexFlatIP 等效余弦相似度。`_embs` 中存储的是**原始 embedding**（不归一化），归一化只在 FAISS 交互时执行。

---

## 6. 完整类接口总览

```
TokenMemoryBank
├── __init__(tokenizer, capacity=1_000_000, fusion_length=256, emb_dim, device, compact_threshold=0.3)
├── CRUD
│   ├── add(entries: List[Tuple[str, Tensor]]) -> List[int]
│   ├── edit(entry_id: int, text: str, embedding: Tensor) -> None
│   ├── delete(entry_id: int) -> None
│   ├── audit(entry_id: int) -> str
│   └── migrate_to() -> List[str]
├── 检索
│   └── retrieve(query_emb: Tensor, k: int) -> Tuple[LongTensor, Tensor]
├── 访问
│   ├── __len__() -> int
│   ├── __getitem__(entry_id: int) -> Tuple[LongTensor, Tensor]
│   └── get_token_ids(entry_ids: LongTensor) -> LongTensor
├── 持久化
│   ├── save(path: str) -> None
│   └── load(path: str) -> None
└── 内部
    ├── _maybe_compact() -> None
    ├── _compact() -> None
    ├── _build_faiss_index() -> None
    └── _validate(token_ids: LongTensor, embedding: Tensor) -> None
```

---

## 7. 从老项目保留的设计

| 来源 | 保留内容 | 文件 |
|------|---------|------|
| `FusionBank.__init__` | 预分配 tensor buffer 模式 | `fusion_bank.py:42-69` |
| `FusionBank._validate` | dtype/shape 校验逻辑 | `fusion_bank.py:263-284` |
| `FusionBank.save_state/load_state` | 序列化格式 + capacity 自适应 | `fusion_bank.py:193-261` |
| `FusionBank.__getitem__` | clone 返回语义 | `fusion_bank.py:118-132` |
| `BankWriter._prepare_tokens` | pad/truncate 到 fusion_length | `fusion_bank.py:398-417` |

## 8. 从老项目去掉的设计

| 去掉内容 | 原因 |
|---------|------|
| BankWriter 分层 | 合并进 TokenMemoryBank |
| LLMLingua 压缩 | TokenMem 不压缩，直接 tokenize |
| BruteForceRouter | FAISS 内置替代 |
| `dtype` 可配参数 | MVP 统一 float32 |
| `retrieve_top1` / `retrieve_topk` 分离 | 统一为 `retrieve(query_emb, k)` |
