# GateCrossAttention 模块设计文档

> **项目**: TokenMem — NeurIPS 2026
> **日期**: 2026-04-27
> **状态**: 已批准
> **核心决策**: 融合架构完全复刻 DecoupledRAG (WWW 2025)

---

## 1. 目标

为 TokenMem 实现知识融合模块，将 TokenMemoryBank 中检索到的知识条目注入 frozen LLM 的隐状态。支持 6 个模型（Qwen3-0.6B/1.7B/4B/8B、Gemma3-1B、Ministral-3B），仅训练门控参数（~1-5M），base LLM 完全冻结。

## 2. 架构总览

```
推理流程:
1. RETRIEVE: query_emb → FAISS(TokenMemoryBank) → top-k token_ids [k, fusion_length]
2. ENCODE:   token_ids → frozen LLM full forward → per-layer KV cache → strided sampling
3. FUSE:     每层 decoder layer: self-attn → FFN → cross-attn(Q=hidden, KV=知识KV) → LinearFusion gate
```

训练时仅 LinearFusion 的 W_A / W_B 可训练，其余参数全部 frozen。

## 3. 核心组件

### 3.1 LinearFusion（门控模块）

照搬 DecoupledRAG `LinearFusion`，零修改。

```python
class LinearFusion(nn.Module):
    """低秩门控融合: output = A + alpha * dropout(B) @ W_A @ W_B"""

    def __init__(self, hidden_dim: int, rank: int = 16, alpha: int = 32, dropout_prob: float = 0.2):
        self.W_A = Parameter(randn(hidden_dim, rank) * 0.01)   # Gaussian σ=0.01
        self.W_B = Parameter(zeros(rank, hidden_dim))            # Zero init
        self.rank = rank
        self.alpha = alpha
        self.dropout_prob = dropout_prob

    def forward(self, A: Tensor, B: Tensor) -> Tensor:
        """A: residual (self-attn+FFN output), B: cross-attention output."""
        dtype = A.dtype
        A, B = A.to(self.W_A.dtype), B.to(self.W_A.dtype)
        B = F.dropout(B, p=self.dropout_prob, training=self.training)
        C = A + self.alpha * torch.matmul(torch.matmul(B, self.W_A), self.W_B)
        return C.to(dtype)
```

**设计要点**:
- Zero init W_B → t=0 时 fusion 输出为零，不干扰 base LLM
- alpha=32 放大低秩信号
- Dropout 0.2 防止对知识过拟合

### 3.2 Modified Decoder Layer

每个模型的 `DecoderLayer.__init__` 中，根据 config 决定是否添加 cross-attention:

```python
self.add_cross_attention = (config.add_cross_attention
                            and layer_idx <= config.add_cross_attention_layer_number)
if self.add_cross_attention:
    self.gate_crossattention = LinearFusion(config.hidden_size)
```

`DecoderLayer.forward` 在 self-attn + FFN 之后追加:

```python
if self.add_cross_attention and encoder_hidden_states is not None:
    residual = hidden_states
    is_kv_cache = isinstance(encoder_hidden_states, tuple)
    if not is_kv_cache:
        encoder_hidden_states = self.input_layernorm(encoder_hidden_states)
    hidden_states, _, _ = self.self_attn(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        is_cross_attention=True,
        is_kv_cache=is_kv_cache,
        ...
    )
    hidden_states = self.gate_crossattention(residual, hidden_states)
```

### 3.3 Modified Attention

Attention.forward 增加参数 `encoder_hidden_states`, `is_cross_attention`, `is_kv_cache`:

**QKV 分叉**:
- `is_cross_attention=False`: 标准自注意力，Q/K/V 均来自 hidden_states
- `is_cross_attention=True, is_kv_cache=True`: Q 来自 hidden_states，K/V 直接使用预计算的 KV cache（已 projected）
- `is_cross_attention=True, is_kv_cache=False`: Q 来自 hidden_states，K/V 由 self.k_proj/v_proj(encoder_hidden_states) 计算

**RoPE**: 仅自注意力时对 Q/K 施加 RoPE，cross-attention 时不加（知识无位置语义）

**Causal mask**: 仅自注意力时应用，cross-attention 时知识全可见（无 mask）

**repeat_kv**: 非 KV cache 模式时需要对 GQA 模型做 KV head repeat；KV cache 模式下已在知识编码阶段完成 repeat

### 3.4 知识编码 (compute_knowledge_hidden_states)

```python
@torch.no_grad()
def compute_knowledge_hidden_states(
    model,
    knowledge_input_ids: LongTensor,       # [batch*num_docs, seq_len]
    knowledge_attention_mask: LongTensor,   # [batch*num_docs, seq_len]
    knowledge_max_seq_len: int = 64,
    num_docs: int = 1,
    encode_batch_size: int = 2,
) -> List[Tuple[Tensor, Tensor]]:
    """
    知识 token_ids → frozen LLM 全层 forward → 逐层 KV cache → strided sampling

    返回: List[Tuple[K, V]] × num_layers
          K, V shape: [batch, num_heads, num_docs * knowledge_max_seq_len, head_dim]
    """
```

**流程**:
1. 分 mini-batch 将 knowledge_input_ids 送入 frozen LLM (`use_cache=True`)
2. 提取 `outputs.past_key_values`：每层一组 (K, V)
3. 对每层 KV 做 strided sampling：按 attention_mask 有效长度均匀采样到 `knowledge_max_seq_len`
4. Multi-doc reshape：将 `[batch*num_docs, ...]` → `[batch, ..., num_docs*kv_len, ...]`
5. 对 GQA 模型做 `repeat_kv`

### 3.5 Strided Sampling

照搬 DecoupledRAG `strided_sampling_hidden_states`。

对每个样本：
- 计算有效长度（attention_mask 中 1 的数量）
- 若有效长度 ≤ max_length：取有效部分 + 左侧 zero-pad
- 若有效长度 > max_length：按步长 `stride = valid_length / max_length` 均匀采样

支持两种输入格式：
- 普通 hidden states `[B, L, D]`
- KV cache tuple `([B, H, L, head_dim], [B, H, L, head_dim])`

## 4. 包装器: TokenMemForCausalLM

```python
class TokenMemForCausalLM(nn.Module):
    """
    封装 modified base model，管理:
    - 参数冻结 (仅 gate_crossattention 可训练)
    - 知识编码 (compute_knowledge_hidden_states)
    - 前向传播 (带 cross-attention 的 LM forward)
    - 门控权重存取 (save_gates / load_gates)
    """

    def __init__(self, model_name_or_path: str, knowledge_max_seq_len: int = 64):
        # 1. 加载对应的 modified modeling 文件
        # 2. 设置 config: add_cross_attention=True, add_cross_attention_layer_number=全层
        # 3. freeze_all_except_gate_crossattention()

    def forward(self, input_ids, attention_mask, labels,
                knowledge_input_ids, knowledge_attention_mask, num_docs=1):
        # 1. compute_knowledge_hidden_states → per_layer_kv
        # 2. model.forward(input_ids, ..., knowledge_outputs=per_layer_kv)
        # 3. return CausalLMOutput(loss, logits)

    def freeze_all_except_gates(self):
        for name, param in self.named_parameters():
            param.requires_grad = "gate_crossattention" in name

    def save_gates(self, directory: str):
        for layer_idx, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'gate_crossattention'):
                torch.save(layer.gate_crossattention.state_dict(),
                           f"{directory}/gate_{layer_idx}.pt")

    def load_gates(self, directory: str):
        for layer_idx, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'gate_crossattention'):
                path = f"{directory}/gate_{layer_idx}.pt"
                if os.path.exists(path):
                    layer.gate_crossattention.load_state_dict(torch.load(path))
```

## 5. 文件结构

```
memory_lora/
  __init__.py
  token_bank.py                          # 已实现 (TokenMemoryBank)
  linear_fusion.py                       # NEW: LinearFusion 模块 (~40行)
  knowledge_encoder.py                   # NEW: compute_knowledge_hidden_states + strided_sampling (~200行)
  tokenmem_model.py                      # NEW: TokenMemForCausalLM 包装器 (~250行)
  modified_models/                       # NEW: Fork 的 transformers modeling 文件
    __init__.py
    modeling_qwen3.py                    # Qwen3 (0.6B/1.7B/4B/8B)
    modeling_gemma3.py                   # Gemma3 (1B)
    modeling_mistral.py                  # Ministral (3B)
```

## 6. Modified Models: 修改范围

对每个模型家族的 `modeling_*.py`，需要修改 3 个位置：

| 修改位置 | 修改内容 | 行数估算 |
|---------|---------|---------|
| `*Attention.forward` | 增加 is_cross_attention / is_kv_cache 分叉，cross-attn 时跳过 RoPE 和 causal mask | ~30行 |
| `*DecoderLayer.__init__` | 添加 `gate_crossattention = LinearFusion(hidden_size)` | ~5行 |
| `*DecoderLayer.forward` | self-attn+FFN 后追加 cross-attention + gate 融合 | ~15行 |
| `*Model.forward` | 接收 `knowledge_outputs` 参数，按层传递给 decoder layer | ~10行 |
| `*ForCausalLM.forward` | 透传 `knowledge_outputs` / `knowledge_input_ids` | ~10行 |

每个模型文件修改量 ~70 行增量代码，模式完全一致。

## 7. Config 扩展

各模型 config 需要新增的字段：

```python
config.add_cross_attention = True
config.add_cross_attention_layer_number = num_layers - 1   # 全层
config.knowledge_max_seq_len = 64                          # strided sampling 目标
config.cross_attention_activation_function = "silu"
config.kg_model_name_or_path = ""                          # 预训练 gate 权重路径
```

## 8. 各模型参数估算

| 模型 | hidden_dim | 层数 | LinearFusion 参数 | 总参数占比 |
|------|-----------|------|------------------|-----------|
| Qwen3-0.6B | 1024 | 28 | 917K | 0.15% |
| Qwen3-1.7B | 2048 | 28 | 1.84M | 0.11% |
| Qwen3-4B | 2560 | 36 | 2.95M | 0.07% |
| Qwen3-8B | 4096 | 36 | 4.72M | 0.06% |
| Gemma3-1B | 1536 | 26 | 1.28M | 0.13% |
| Ministral-3B | 2560 | 36 | 2.95M | 0.10% |

## 9. 训练配置

```yaml
epochs: 5
optimizer: Adam
lr: 1e-3
batch_size: 16
knowledge_max_seq_len: 64
cross_attention_activation: silu
gradient_checkpointing: true    # 4B+ 模型
trainable: gate_crossattention (LinearFusion) only
frozen: all base LLM parameters
```

## 10. 与 TokenMemoryBank 的集成

**训练时 (oracle)**:
1. 数据集提供 `(question, knowledge_passage, answer)` 三元组
2. `knowledge_passage` → tokenize → `knowledge_input_ids`
3. `compute_knowledge_hidden_states` → per-layer KV cache
4. Query forward with cross-attention → LM loss on answer

**推理时**:
1. Query → embedding → `bank.retrieve(query_emb, k)` → entry_ids
2. `bank.get_token_ids(entry_ids)` → `knowledge_input_ids [k, fusion_length]`
3. `compute_knowledge_hidden_states` → per-layer KV cache
4. Query forward with cross-attention → generate answer

## 11. 与参考项目的关键差异

| 维度 | DecoupledRAG | Memory-LoRA-old | TokenMem (本设计) |
|------|-------------|-----------------|------------------|
| 知识编码 | 全层 LLM → 逐层 KV | 前6层 encoder → 单一表示 | **全层 LLM → 逐层 KV** (同 DecoupledRAG) |
| 融合机制 | LLM QKV + LinearFusion | 独立 QKV + zero-init out_proj | **LLM QKV + LinearFusion** (同 DecoupledRAG) |
| 注入层数 | 全部 32 层 | 4 层 hook | **全部层** (同 DecoupledRAG) |
| 实现方式 | Fork transformers | Forward hook | **Fork transformers** (同 DecoupledRAG) |
| 知识来源 | 在线检索文档 | LLMLingua 压缩 64 tokens | **TokenMemoryBank 持久化 token_ids** |
| 知识管理 | 无 (每次检索) | FusionBank (不可编辑) | **TokenMemoryBank (可增删改审迁移)** |
| 模型支持 | LLaMA / Qwen2 | Qwen3 only | **Qwen3 / Gemma3 / Mistral** (3 families) |
| Retrieval head | 有 (contrastive loss) | BruteForceRouter | **无** (FAISS on cached_emb) |

## 12. 不做的事 (Non-Goals)

- 不实现 retrieval head / contrastive loss（用 FAISS 替代）
- 不实现 DualEncoder / Reranker（TokenMemoryBank 已有检索能力）
- 不实现 per-layer 独立 QKV 投影（复用 LLM 自身权重）
- 不实现 Null KV fallback（strided sampling 处理 padding）
- 不支持 LLMLingua 压缩（直接用原始 token_ids + strided sampling）

## 13. 测试策略

| 测试层级 | 内容 | 验证标准 |
|---------|------|---------|
| Unit: LinearFusion | 零初始化 → output == input; 非零权重 → output ≠ input | 数值精度 1e-6 |
| Unit: strided_sampling | 各长度输入 → 输出长度 == max_length; padding 处理 | shape + 内容 |
| Integration: Modified Model | 无知识 → 输出与原始模型一致; 有知识 → 输出改变 | logits 对比 |
| Integration: TokenMemForCausalLM | forward → loss 下降; save/load gates → 输出一致 | loss + 数值 |
| Smoke: SFT | Qwen3-0.6B + 100 samples → loss 持续下降 | loss 曲线 |

## 14. 风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| Qwen3/Gemma3/Mistral 内部结构差异导致 modeling 修改困难 | 30% | 优先实现 Qwen3，其余参照调整 |
| 全层 cross-attention 显存不足 (8B) | 20% | gradient_checkpointing + 减小 batch_size + A100 |
| strided sampling 损失关键信息 | 10% | 可调 knowledge_max_seq_len (64→128) |
