# FINAL PROPOSAL: TokenMem

## 论文标题（候选）
- *TokenMem: A Readable, Editable, and Portable Internalized Memory Pipeline for Frozen LLMs*
- *Retrofitting Token-Level Memory into Frozen LLMs via Cross-Attention Adapters*
- *TokenMem: Plug-and-Play Knowledge Memory for Large Language Models*

---

## Problem Anchor（防止scope drift）

**问题**: 现有LLM缺乏一个完整的内部化记忆系统——能够高效检索、在hidden-state层融合、支持运行时编辑、且跨模型通用的知识记忆方案。

- RAG消耗context window，受限于"lost in the middle"
- 参数编辑（ROME/MEMIT）不可逆，难以扩展
- ExplicitLM（本组prior work）证明token记忆有效，但需预训练
- 现有记忆系统最多测3个模型（KBLaM），无人验证6+模型通用性

**非目标**:
- 不追求超越ExplicitLM的绝对性能（不同范式）
- 不追求解决知识冲突检测（future work）
- 不追求PKM高效检索（当前用FAISS，PKM留future work）

---

## 核心方法

### 1. Pipeline总览

```
┌──────────────────────────────────────────────────────┐
│                   TokenMem Pipeline                   │
│                                                       │
│  ┌────────────┐   ┌─────────────┐   ┌─────────────┐  │
│  │  RETRIEVE   │──▶│    FUSE     │──▶│   UPDATE    │  │
│  │             │   │             │   │             │  │
│  │ FAISS       │   │ Cross-Attn  │   │ Token级     │  │
│  │ top-k检索   │   │ 注入(借鉴    │   │ 增/删/编辑  │  │
│  │             │   │ DecoupledRAG)│   │             │  │
│  └────────────┘   └─────────────┘   └─────────────┘  │
│                                                       │
│  TokenMemoryBank (per-model): tokenized知识序列        │
│  跨模型迁移: detokenize → retokenize                   │
└──────────────────────────────────────────────────────┘
```

### 2. TokenMemoryBank (per-model)

知识以**该模型tokenizer编码后的token序列**存储：

```python
class TokenMemoryBank:
    """合并设计：内置tokenizer + FAISS索引，embedding由外部预计算。"""
    _tokens: Tensor   # [capacity, fusion_length], dtype=long
    _embs: Tensor     # [capacity, emb_dim], dtype=float32
    _deleted: Tensor  # [capacity], dtype=bool, 软删除标记
    _index: faiss.IndexIDMap  # 内置FAISS索引

    def __init__(self, tokenizer, emb_dim, capacity=1M, fusion_length=256, ...): ...

    def add(self, entries: List[Tuple[str, Tensor]]) -> List[int]:
        """批量写入(text, embedding)对 → 内部tokenize → 存储 → 更新FAISS"""

    def edit(self, entry_id: int, text: str, embedding: Tensor) -> None:
        """重新tokenize → 更新embedding → 更新FAISS"""

    def delete(self, entry_id: int) -> None:
        """软删除 → 从FAISS移除 → 自动compact(>=30%阈值)"""

    def audit(self, entry_id: int) -> str:
        """tokenizer.decode(token_ids) → 人类可读文本"""

    def migrate_to(self) -> List[str]:
        """decode所有未删除条目 → 返回文本列表（调用方自行构建新bank）"""

    def retrieve(self, query_emb, k) -> Tuple[LongTensor, Tensor]:
        """FAISS top-k检索 → 返回(entry_ids, scores)"""

    def get_token_ids(self, entry_ids) -> LongTensor:
        """批量获取token_ids → 供推理时frozen LLM前向计算KV"""
```

> **实现细节见**: `memory_lora/token_bank.py` (已实现, 56/56测试通过)

### 3. 知识检索: FAISS

- 用LLM embedding层或sentence-transformers计算知识条目的embedding
- FAISS索引: bank < 100K用flat brute-force, > 100K用IVF-PQ
- 检索top-k条目的预计算KV缓存用于cross-attention注入
- PKM留作future work（需要解决key聚类平衡性问题）

### 4. 知识融合: Cross-Attention（借鉴DecoupledRAG）

```python
class GateCrossAttention(nn.Module):
    """每个注入层的融合模块（唯一可训练组件）"""

    def __init__(self, hidden_dim, rank=16, alpha=32):
        # 零初始化低秩融合权重（DecoupledRAG的W_β设计）
        self.A = nn.Linear(hidden_dim, rank, bias=False)   # Gaussian init σ=0.01
        self.B = nn.Linear(rank, hidden_dim, bias=False)   # Zero init
        self.alpha = alpha / rank  # scaling factor
        self.dropout = nn.Dropout(0.2)

    def forward(self, hidden_int, hidden_ext):
        """
        hidden_int: 内部self-attention输出 [B, seq, D]
        hidden_ext: 外部cross-attention输出 [B, seq, D]
        """
        fusion = self.alpha * self.B(self.dropout(self.A(hidden_ext)))
        return hidden_int + fusion
```

**设计要点**（与DecoupledRAG一致，代码验证）:
- 基座LLM**完全冻结**，不使用LoRA微调基座
- 只训练GateCrossAttention（融合权重）
- B矩阵零初始化 → t=0时融合输出为零 → LLM原始能力完整保留
- 注入位置：模型层均匀分布的4层（如28层模型的[6,12,18,24]）

### 5. 训练策略: 一次SFT，跨域泛化

```
训练数据: News dataset 50K（时间分割，较早文章）
  格式: (question, knowledge_passage, answer)
  每条: question通过FAISS检索到knowledge → cross-attention(检索到的知识token_ids → 实时编码为KV) → 预测answer

训练配置:
  可训练: 仅GateCrossAttention（各注入层的融合权重）
  冻结: 基座LLM全部参数
  epochs: 5
  lr: 1e-3
  batch_size: 16

泛化验证:
  ├── News test 10K（较新文章） → 时间泛化
  ├── MedQA test              → 领域泛化（医学）
  ├── ARC test                → 领域泛化（科学）
  └── MMLU test               → 领域泛化（通用知识）
```

**核心论点**: adapter学到的是"如何通过cross-attention利用外部知识"的通用能力，而非特定领域知识。训练一次后，知识bank可自由切换，无需重训练。

### 6. 多模型适配

每个模型需要：
1. **独立的TokenMemoryBank**: token_ids + cached_emb + 内置FAISS索引（用该模型tokenizer+embedding层）
2. **独立的GateCrossAttention**: SFT训练融合权重

跨模型迁移：`bank.migrate_to()` → 文本列表 → 新tokenizer+embedding重建新bank
推理时KV计算：`bank.retrieve()` → `bank.get_token_ids()` → frozen LLM前向 → 实时得到KV供cross-attention

---

## 实验策略

### 核心思路

用**模型覆盖广度**作为核心evidence，基线精简到业内标准。

### 模型矩阵

| 模型 | 家族 | 参数量 | 目的 |
|------|------|--------|------|
| Qwen3-0.6B | Qwen | 0.6B | 最小规模验证 |
| Qwen3-1.7B | Qwen | 1.7B | 同家族scaling |
| Qwen3-4B | Qwen | 4B | 同家族scaling + **E2-E6默认模型** |
| Qwen3-8B | Qwen | 8B | 大模型验证 |
| Gemma3-1B | Google | 1B | 跨家族验证 |
| Ministral-3B | Mistral | 3B | 跨家族验证 |

### 基线优先级

```
P0 (必做):
├── No-Memory — 冻结LLM直推（下界）
└── VanillaRAG — 检索+放入prompt

P1 (推荐):
└── DecoupledRAG — cross-attention注入但无持久记忆

P2 (如果时间允许):
└── LoRA+RAG
```

### 数据集

| 数据集 | 规模 | 角色 | 说明 |
|--------|------|------|------|
| News train | 50K | **SFT训练** | 时间分割较早文章 |
| News test | 10K | 测试(in-domain) | 时间分割较新文章 |
| MedQA test | ~1.3K | 测试(out-of-domain) | 医学知识 |
| ARC test | ~1.2K | 测试(out-of-domain) | 科学推理 |
| MMLU test | ~14K | 测试(out-of-domain) | 综合知识 |

---

## 论文结构

```
1. Introduction
   - LLM知识增强的局限
   - 完整记忆pipeline需求
   - TokenMem: 即插即用 + 6模型验证 + 跨域泛化

2. Related Work
   - RAG及变体（VanillaRAG, DecoupledRAG）
   - 显式记忆（ExplicitLM, MemoryLLM, KBLaM）

3. Method: TokenMem Pipeline
   - 3.1 TokenMemoryBank
   - 3.2 FAISS检索
   - 3.3 Cross-Attention融合
   - 3.4 知识生命周期管理
   - 3.5 多模型适配与迁移

4. Experiments
   - 4.1 设置
   - 4.2 多模型注入性能（E1, 核心表格）
   - 4.3 跨领域泛化
   - 4.4 知识编辑（E2）
   - 4.5 消融（E3-E4）

5. Analysis & Discussion
6. Conclusion
```

---

## 关键风险

| 风险 | 概率 | 应对 |
|------|------|------|
| 某模型adapter训不动 | 20% | 从6模型中去掉；5模型仍足够 |
| Out-of-domain泛化差 | 20% | 增加SFT数据多样性；分析讨论 |
| 8B训练时间超预期 | 20% | 优先4B；8B放supplementary |
| Novelty质疑 | 40% | 6模型通用性+跨域泛化实验说话 |
