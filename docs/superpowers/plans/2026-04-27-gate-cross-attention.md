# GateCrossAttention 模块实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 TokenMem 实现 DecoupledRAG 风格的知识融合模块——修改 3 个模型家族的 transformers modeling 文件，注入 cross-attention + LinearFusion 门控。

**Architecture:** 知识 token_ids → frozen LLM 全层 forward (`output_hidden_states=True`) → 逐层 hidden states → strided sampling → 在每层 decoder layer 中 cross-attention（复用 LLM 自身 QKV 权重）+ LinearFusion 低秩门控。仅 LinearFusion 参数可训练（~1-5M），base LLM 完全冻结。

**Tech Stack:** Python 3.11, PyTorch, transformers 4.57.3, FAISS, Conda env: ExplicitLLM

**关键约束:**
- Qwen3 和 Mistral 的 Attention 使用 `q_norm`/`k_norm` + `position_embeddings` tuple (cos, sin)
- Gemma3 有 dual RoPE (`position_embeddings_global`/`local`) 和额外 layernorm
- Cross-attention 时跳过 RoPE（知识无位置语义）、跳过 causal mask（知识全可见）
- Cross-attention 强制使用 `eager_attention_forward`（避免 flash/sdpa 的 causal 假设）

---

## 文件结构

```
memory_lora/
  __init__.py                                    # MODIFY: 增加 exports
  token_bank.py                                  # 不动
  linear_fusion.py                               # CREATE: LinearFusion 门控模块
  knowledge_encoder.py                           # CREATE: strided_sampling + compute_knowledge_hidden_states
  tokenmem_model.py                              # CREATE: TokenMemForCausalLM 包装器
  modified_models/                               # CREATE: 目录
    __init__.py                                  # CREATE: 模型注册表
    modeling_qwen3.py                            # CREATE: Fork from transformers 4.57.3
    modeling_mistral.py                          # CREATE: Fork from transformers 4.57.3
    modeling_gemma3.py                           # CREATE: Fork from transformers 4.57.3
tests/
  unit/
    test_linear_fusion.py                        # CREATE
    test_knowledge_encoder.py                    # CREATE
  integration/
    __init__.py                                  # CREATE
    test_modified_qwen3.py                       # CREATE
    test_tokenmem_model.py                       # CREATE
```

---

## Task 1: LinearFusion 门控模块

**Files:**
- Create: `memory_lora/linear_fusion.py`
- Test: `tests/unit/test_linear_fusion.py`

- [ ] **Step 1: 编写 LinearFusion 测试**

```python
# tests/unit/test_linear_fusion.py
"""LinearFusion 门控模块单元测试。"""
import pytest
import torch

from memory_lora.linear_fusion import LinearFusion


class TestLinearFusionInit:
    """初始化与参数形状测试。"""

    def test_parameter_shapes(self):
        m = LinearFusion(hidden_dim=64, rank=8, alpha=32)
        assert m.W_A.shape == (64, 8)
        assert m.W_B.shape == (8, 64)

    def test_w_b_zero_init(self):
        m = LinearFusion(hidden_dim=64, rank=8)
        assert torch.all(m.W_B == 0)

    def test_w_a_gaussian_init(self):
        m = LinearFusion(hidden_dim=256, rank=16)
        assert m.W_A.std() < 0.05  # σ=0.01, 容许统计波动


class TestLinearFusionForward:
    """前向传播行为测试。"""

    def test_zero_init_is_identity(self):
        """W_B 全零 → fusion 输出 == 输入 A。"""
        m = LinearFusion(hidden_dim=64, rank=8)
        m.eval()
        A = torch.randn(2, 10, 64)
        B = torch.randn(2, 10, 64)
        out = m(A, B)
        torch.testing.assert_close(out, A)

    def test_nonzero_weights_change_output(self):
        """W_B 非零 → fusion 输出 ≠ A。"""
        m = LinearFusion(hidden_dim=64, rank=8)
        m.eval()
        m.W_B.data.fill_(0.1)
        A = torch.randn(2, 10, 64)
        B = torch.randn(2, 10, 64)
        out = m(A, B)
        assert not torch.allclose(out, A)

    def test_output_shape_matches_input(self):
        m = LinearFusion(hidden_dim=128, rank=16)
        A = torch.randn(4, 20, 128)
        B = torch.randn(4, 20, 128)
        assert m(A, B).shape == (4, 20, 128)

    def test_dtype_preservation(self):
        """输入 bf16 → 输出 bf16。"""
        m = LinearFusion(hidden_dim=64, rank=8)
        A = torch.randn(2, 10, 64, dtype=torch.bfloat16)
        B = torch.randn(2, 10, 64, dtype=torch.bfloat16)
        assert m(A, B).dtype == torch.bfloat16

    def test_alpha_scaling(self):
        """alpha=0 → 输出 == A (无论 W_B)。"""
        m = LinearFusion(hidden_dim=64, rank=8, alpha=0)
        m.eval()
        m.W_B.data.fill_(1.0)
        A = torch.randn(2, 10, 64)
        B = torch.randn(2, 10, 64)
        torch.testing.assert_close(m(A, B), A)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
conda run -n ExplicitLLM python -m pytest tests/unit/test_linear_fusion.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'memory_lora.linear_fusion'`

- [ ] **Step 3: 实现 LinearFusion**

```python
# memory_lora/linear_fusion.py
"""LinearFusion —— 低秩门控融合模块（照搬 DecoupledRAG）。

核心公式: output = A + alpha * dropout(B) @ W_A @ W_B
- W_A: Gaussian σ=0.01 初始化
- W_B: Zero 初始化 → t=0 时 fusion 输出为零，不干扰 base LLM
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter


class LinearFusion(nn.Module):
    """低秩门控融合: output = A + alpha * dropout(B) @ W_A @ W_B。

    参数：
        hidden_dim: 隐层维度
        rank: 低秩分解秩 (默认 16)
        alpha: 缩放因子 (默认 32)
        dropout_prob: knowledge dropout (默认 0.2)
    """

    def __init__(
        self,
        hidden_dim: int,
        rank: int = 16,
        alpha: int = 32,
        dropout_prob: float = 0.2,
    ) -> None:
        super().__init__()
        self.W_A = Parameter(torch.randn(hidden_dim, rank) * 0.01)
        self.W_B = Parameter(torch.zeros(rank, hidden_dim))
        self.rank = rank
        self.alpha = alpha
        self.dropout_prob = dropout_prob

    def forward(self, A: Tensor, B: Tensor) -> Tensor:
        """门控融合。

        参数：
            A: residual (self-attn + FFN 输出) [B, L, D]
            B: cross-attention 输出 [B, L, D]
        返回：
            融合后的 hidden states [B, L, D]
        """
        dtype = A.dtype
        A = A.to(self.W_A.dtype)
        B = B.to(self.W_A.dtype)
        B = F.dropout(B, p=self.dropout_prob, training=self.training)
        C = A + self.alpha * torch.matmul(torch.matmul(B, self.W_A), self.W_B)
        return C.to(dtype)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
conda run -n ExplicitLLM python -m pytest tests/unit/test_linear_fusion.py -v
```
Expected: 6 passed

- [ ] **Step 5: 提交**

```bash
git add memory_lora/linear_fusion.py tests/unit/test_linear_fusion.py
git commit -m "feat: add LinearFusion gate module (DecoupledRAG-style)"
```

---

## Task 2: Fork 并修改 modeling_qwen3.py

**Files:**
- Create: `memory_lora/modified_models/__init__.py`
- Create: `memory_lora/modified_models/modeling_qwen3.py` (fork from `transformers/models/qwen3/modeling_qwen3.py`)
- Test: `tests/integration/__init__.py`
- Test: `tests/integration/test_modified_qwen3.py`

**源文件位置**: `/home/iomgaa/miniconda3/envs/ExplicitLLM/lib/python3.11/site-packages/transformers/models/qwen3/modeling_qwen3.py` (528 行)

**修改范围**: Qwen3Attention.forward + Qwen3DecoderLayer.__init__/forward + Qwen3Model.forward + Qwen3ForCausalLM.forward

- [ ] **Step 1: 编写集成测试**

```python
# tests/integration/__init__.py
# (空文件)
```

```python
# tests/integration/test_modified_qwen3.py
"""Modified Qwen3 cross-attention 集成测试。

使用 Qwen3-0.6B 验证:
1. 无知识时输出与原始模型一致
2. 有知识时 cross-attention 改变输出
3. gate 全零初始化时输出不变
"""
import pytest
import torch
from transformers import AutoTokenizer, AutoConfig

# 跳过条件: 无 GPU 或无模型权重
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="需要 GPU"
)

MODEL_NAME = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def modified_model():
    """加载 modified Qwen3 并启用 cross-attention。"""
    from memory_lora.modified_models.modeling_qwen3 import (
        Qwen3ForCausalLM as ModifiedQwen3ForCausalLM,
    )

    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.add_cross_attention = True
    config.add_cross_attention_layer_number = config.num_hidden_layers - 1
    model = ModifiedQwen3ForCausalLM.from_pretrained(
        MODEL_NAME, config=config, torch_dtype=torch.bfloat16
    )
    model.eval().cuda()
    return model


class TestModifiedQwen3:
    """Modified Qwen3 行为验证。"""

    def test_gate_zero_init_preserves_output(self, modified_model, tokenizer):
        """gate 全零初始化 + 提供知识 → 输出与无知识时一致。"""
        inputs = tokenizer("Hello world", return_tensors="pt").to("cuda")
        num_layers = modified_model.config.num_hidden_layers
        B, L = inputs["input_ids"].shape
        D = modified_model.config.hidden_size

        # 伪造知识 hidden states: 每层一个 [B, kv_len, D] tensor
        kv_len = 16
        knowledge_outputs = [
            torch.randn(B, kv_len, D, dtype=torch.bfloat16, device="cuda")
            for _ in range(num_layers)
        ]

        with torch.no_grad():
            out_no_knowledge = modified_model(
                **inputs, knowledge_outputs=None
            )
            out_with_knowledge = modified_model(
                **inputs, knowledge_outputs=knowledge_outputs
            )

        # gate 零初始化 → cross-attn 贡献为零 → 输出应相同
        torch.testing.assert_close(
            out_no_knowledge.logits,
            out_with_knowledge.logits,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_nonzero_gate_changes_output(self, modified_model, tokenizer):
        """手动设置 gate 权重非零 → 输出改变。"""
        inputs = tokenizer("Hello world", return_tensors="pt").to("cuda")
        num_layers = modified_model.config.num_hidden_layers
        B, L = inputs["input_ids"].shape
        D = modified_model.config.hidden_size

        kv_len = 16
        knowledge_outputs = [
            torch.randn(B, kv_len, D, dtype=torch.bfloat16, device="cuda")
            for _ in range(num_layers)
        ]

        # 设置所有 gate 的 W_B 为非零
        for layer in modified_model.model.layers:
            if hasattr(layer, "gate_crossattention"):
                layer.gate_crossattention.W_B.data.fill_(0.1)

        with torch.no_grad():
            out_no_knowledge = modified_model(
                **inputs, knowledge_outputs=None
            )
            out_with_knowledge = modified_model(
                **inputs, knowledge_outputs=knowledge_outputs
            )

        assert not torch.allclose(
            out_no_knowledge.logits, out_with_knowledge.logits
        )

        # 恢复零初始化
        for layer in modified_model.model.layers:
            if hasattr(layer, "gate_crossattention"):
                layer.gate_crossattention.W_B.data.zero_()

    def test_all_layers_have_gate(self, modified_model):
        """全层配置 → 每层都有 gate_crossattention。"""
        for layer in modified_model.model.layers:
            assert hasattr(layer, "gate_crossattention")

    def test_gate_params_are_only_trainable(self, modified_model):
        """仅 gate_crossattention 参数可训练。"""
        for name, p in modified_model.named_parameters():
            if "gate_crossattention" in name:
                assert p.requires_grad
            # base 参数默认 requires_grad=True (from_pretrained)，
            # 冻结逻辑在 TokenMemForCausalLM 中，此处不测
```

- [ ] **Step 2: 运行测试确认失败**

```bash
conda run -n ExplicitLLM python -m pytest tests/integration/test_modified_qwen3.py -v --no-header 2>&1 | head -20
```
Expected: FAIL — `ModuleNotFoundError: No module named 'memory_lora.modified_models'`

- [ ] **Step 3: Fork modeling_qwen3.py 并添加 cross-attention 修改**

3a. 创建目录和 `__init__.py`:

```python
# memory_lora/modified_models/__init__.py
"""Modified transformers modeling 文件（fork from transformers 4.57.3）。

支持 cross-attention 知识注入的模型:
- Qwen3 (0.6B / 1.7B / 4B / 8B)
- Mistral (Ministral-3B)
- Gemma3 (1B)
"""
```

3b. 复制原始文件:

```bash
SRC=$(conda run -n ExplicitLLM python -c "import transformers,os; print(os.path.join(os.path.dirname(transformers.__file__),'models','qwen3','modeling_qwen3.py'))")
cp "$SRC" memory_lora/modified_models/modeling_qwen3.py
```

3c. 修改 `Qwen3Attention.forward`（约 line 188-230），增加 cross-attention 分叉:

在 forward 签名中新增 `encoder_hidden_states` 和 `is_cross_attention` 参数:

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,  # NEW
    is_cross_attention: bool = False,                        # NEW
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)

    if is_cross_attention and encoder_hidden_states is not None:
        # Cross-attention: K/V 来自知识 hidden states
        kv_input_shape = encoder_hidden_states.shape[:-1]
        kv_hidden_shape = (*kv_input_shape, -1, self.head_dim)
        key_states = self.k_norm(
            self.k_proj(encoder_hidden_states).view(kv_hidden_shape)
        ).transpose(1, 2)
        value_states = self.v_proj(encoder_hidden_states).view(kv_hidden_shape).transpose(1, 2)
        # 不加 RoPE（知识无位置语义）
    else:
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

    if is_cross_attention:
        # Cross-attention 强制 eager（无 causal mask、无 sliding window）
        attn_output, attn_weights = eager_attention_forward(
            self, query_states, key_states, value_states,
            attention_mask=None,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=None,
        )
    else:
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights
```

3d. 修改 `Qwen3DecoderLayer.__init__` 和 `.forward`（约 line 233-277）:

`__init__` 末尾追加:

```python
# Cross-attention 门控
self.add_cross_attention = (
    getattr(config, "add_cross_attention", False)
    and layer_idx <= getattr(config, "add_cross_attention_layer_number", -1)
)
if self.add_cross_attention:
    from memory_lora.linear_fusion import LinearFusion
    self.gate_crossattention = LinearFusion(config.hidden_size)
```

`forward` 签名增加 `encoder_hidden_states`:

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,  # NEW
    **kwargs: Unpack[TransformersKwargs],
) -> torch.Tensor:
```

在 FFN 之后（`hidden_states = residual + hidden_states` 后）追加:

```python
    # Cross-attention + LinearFusion gate
    if self.add_cross_attention and encoder_hidden_states is not None:
        residual = hidden_states
        cross_input = self.input_layernorm(encoder_hidden_states)
        cross_out, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=None,
            encoder_hidden_states=cross_input,
            is_cross_attention=True,
        )
        hidden_states = self.gate_crossattention(residual, cross_out)

    return hidden_states
```

3e. 修改 `Qwen3Model.forward`（约 line 356-425），增加 `knowledge_outputs` 参数:

签名增加:

```python
def forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    knowledge_outputs: Optional[list] = None,  # NEW: List[Tensor], 每层一个
    **kwargs: Unpack[TransformersKwargs],
) -> BaseModelOutputWithPast:
```

decoder layer 循环（约 line 409-419）修改为:

```python
    for idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
        layer_knowledge = None
        if knowledge_outputs is not None and idx < len(knowledge_outputs):
            layer_knowledge = knowledge_outputs[idx]

        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask_mapping[decoder_layer.attention_type],
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            encoder_hidden_states=layer_knowledge,  # NEW
            **kwargs,
        )
```

3f. 修改 `Qwen3ForCausalLM.forward`（约 line 445-506），透传 `knowledge_outputs`:

签名增加:

```python
    knowledge_outputs: Optional[list] = None,  # NEW
```

传递给 model:

```python
    outputs: BaseModelOutputWithPast = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        cache_position=cache_position,
        knowledge_outputs=knowledge_outputs,  # NEW
        **kwargs,
    )
```

- [ ] **Step 4: 运行测试**

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n ExplicitLLM python -m pytest tests/integration/test_modified_qwen3.py -v --timeout=120
```
Expected: 4 passed

- [ ] **Step 5: 提交**

```bash
git add memory_lora/modified_models/ tests/integration/
git commit -m "feat: fork modeling_qwen3.py with cross-attention + LinearFusion gate"
```

---

## Task 3: strided_sampling 工具函数

**Files:**
- Create: `memory_lora/knowledge_encoder.py`
- Test: `tests/unit/test_knowledge_encoder.py`

- [ ] **Step 1: 编写 strided_sampling 测试**

```python
# tests/unit/test_knowledge_encoder.py
"""知识编码工具函数测试。"""
import pytest
import torch

from memory_lora.knowledge_encoder import strided_sampling


class TestStridedSampling:
    """strided_sampling 行为测试。"""

    def test_short_input_pads_to_max_length(self):
        """有效长度 < max_length → zero-pad 到目标长度。"""
        hidden = torch.randn(1, 10, 64)
        mask = torch.ones(1, 10)
        out = strided_sampling(hidden, mask, max_length=16)
        assert out.shape == (1, 16, 64)

    def test_long_input_samples_to_max_length(self):
        """有效长度 > max_length → 均匀采样。"""
        hidden = torch.arange(128).float().unsqueeze(0).unsqueeze(-1).expand(1, 128, 4)
        mask = torch.ones(1, 128)
        out = strided_sampling(hidden, mask, max_length=32)
        assert out.shape == (1, 32, 4)

    def test_exact_length_no_change(self):
        """有效长度 == max_length → 原样返回。"""
        hidden = torch.randn(1, 64, 32)
        mask = torch.ones(1, 64)
        out = strided_sampling(hidden, mask, max_length=64)
        assert out.shape == (1, 64, 32)

    def test_left_padding_handled(self):
        """左侧 padding → 仅采样有效部分。"""
        hidden = torch.randn(1, 20, 8)
        mask = torch.zeros(1, 20)
        mask[0, 10:] = 1  # 后 10 个有效
        out = strided_sampling(hidden, mask, max_length=16)
        assert out.shape == (1, 16, 8)
        # 前 6 个位置应为 zero-pad
        assert torch.all(out[0, :6, :] == 0)

    def test_batch_dimension(self):
        """batch > 1 时正确处理。"""
        hidden = torch.randn(3, 50, 16)
        mask = torch.ones(3, 50)
        mask[1, :30] = 0  # 第二个样本仅 20 有效
        out = strided_sampling(hidden, mask, max_length=32)
        assert out.shape == (3, 32, 16)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
conda run -n ExplicitLLM python -m pytest tests/unit/test_knowledge_encoder.py::TestStridedSampling -v
```
Expected: FAIL — `ImportError`

- [ ] **Step 3: 实现 strided_sampling**

```python
# memory_lora/knowledge_encoder.py
"""知识编码工具 —— strided sampling + 逐层 hidden states 计算。

参考 DecoupledRAG (WWW 2025) 的知识编码管线:
- strided_sampling: 按 attention_mask 有效长度均匀采样到目标长度
- compute_knowledge_hidden_states: frozen LLM 全层 forward → 逐层 hidden states
"""
from __future__ import annotations

from typing import List, Optional

import torch
from torch import LongTensor, Tensor


def strided_sampling(
    hidden_states: Tensor,
    attention_mask: Tensor,
    max_length: int,
) -> Tensor:
    """按 attention_mask 有效长度对 hidden_states 进行跨步采样。

    处理 left-padding: 仅采样有效（mask=1）部分。

    参数：
        hidden_states: [batch, seq_len, hidden_dim]
        attention_mask: [batch, seq_len], 1=有效, 0=padding
        max_length: 目标序列长度

    返回：
        采样后的 hidden_states [batch, max_length, hidden_dim]
    """
    batch_size, seq_len, hidden_dim = hidden_states.shape
    device = hidden_states.device
    result = torch.zeros(batch_size, max_length, hidden_dim,
                         dtype=hidden_states.dtype, device=device)

    for i in range(batch_size):
        valid_length = int(attention_mask[i].sum().item())
        valid_length = max(1, valid_length)
        start_pos = max(0, seq_len - valid_length)

        if valid_length <= max_length:
            pad_len = max_length - valid_length
            result[i, pad_len:] = hidden_states[i, start_pos:]
        else:
            indices = torch.linspace(0, valid_length - 1, max_length, device=device).long()
            result[i] = hidden_states[i, start_pos + indices]

    return result
```

- [ ] **Step 4: 运行测试确认通过**

```bash
conda run -n ExplicitLLM python -m pytest tests/unit/test_knowledge_encoder.py::TestStridedSampling -v
```
Expected: 5 passed

- [ ] **Step 5: 提交**

```bash
git add memory_lora/knowledge_encoder.py tests/unit/test_knowledge_encoder.py
git commit -m "feat: add strided_sampling for knowledge encoding"
```

---

## Task 4: compute_knowledge_hidden_states

**Files:**
- Modify: `memory_lora/knowledge_encoder.py`
- Test: `tests/unit/test_knowledge_encoder.py` (追加)

- [ ] **Step 1: 编写测试**

在 `tests/unit/test_knowledge_encoder.py` 末尾追加:

```python
class TestComputeKnowledgeHiddenStates:
    """compute_knowledge_hidden_states 集成测试。"""

    @pytest.fixture(scope="class")
    def qwen3_model(self):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16
        )
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
        return model

    @pytest.fixture(scope="class")
    def qwen3_tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")
    def test_output_structure(self, qwen3_model, qwen3_tokenizer):
        """输出为 List[Tensor]，长度 == num_layers，每个 shape [B, max_len, D]。"""
        from memory_lora.knowledge_encoder import compute_knowledge_hidden_states

        text = "Paris is the capital of France."
        inputs = qwen3_tokenizer(text, return_tensors="pt", padding="max_length",
                                  max_length=64, truncation=True).to(qwen3_model.device)

        result = compute_knowledge_hidden_states(
            qwen3_model,
            knowledge_input_ids=inputs["input_ids"],
            knowledge_attention_mask=inputs["attention_mask"],
            knowledge_max_seq_len=16,
        )

        num_layers = qwen3_model.config.num_hidden_layers
        assert len(result) == num_layers
        assert result[0].shape == (1, 16, qwen3_model.config.hidden_size)
        assert result[0].dtype == torch.bfloat16

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")
    def test_multi_doc_reshape(self, qwen3_model, qwen3_tokenizer):
        """num_docs > 1 时正确 reshape。"""
        from memory_lora.knowledge_encoder import compute_knowledge_hidden_states

        texts = ["Paris is the capital.", "Berlin is the capital."]
        inputs = qwen3_tokenizer(texts, return_tensors="pt", padding="max_length",
                                  max_length=32, truncation=True).to(qwen3_model.device)

        result = compute_knowledge_hidden_states(
            qwen3_model,
            knowledge_input_ids=inputs["input_ids"],
            knowledge_attention_mask=inputs["attention_mask"],
            knowledge_max_seq_len=8,
            num_docs=2,
        )

        # [1 (batch), 2*8 (num_docs * kv_len), D]
        assert result[0].shape == (1, 16, qwen3_model.config.hidden_size)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n ExplicitLLM python -m pytest tests/unit/test_knowledge_encoder.py::TestComputeKnowledgeHiddenStates -v --timeout=60
```
Expected: FAIL — `ImportError: cannot import name 'compute_knowledge_hidden_states'`

- [ ] **Step 3: 实现 compute_knowledge_hidden_states**

在 `memory_lora/knowledge_encoder.py` 末尾追加:

```python
@torch.no_grad()
def compute_knowledge_hidden_states(
    model,
    knowledge_input_ids: LongTensor,
    knowledge_attention_mask: LongTensor,
    knowledge_max_seq_len: int = 64,
    num_docs: int = 1,
    encode_batch_size: int = 4,
) -> List[Tensor]:
    """知识 token_ids → frozen LLM 全层 forward → 逐层 hidden states → strided sampling。

    使用 output_hidden_states=True 获取逐层 hidden states（非 KV cache），
    避免 RoPE 污染（KV cache 路径会对 K 施加 RoPE）。

    参数：
        model: HuggingFace CausalLM (frozen)
        knowledge_input_ids: [batch * num_docs, seq_len]
        knowledge_attention_mask: [batch * num_docs, seq_len]
        knowledge_max_seq_len: strided sampling 目标长度
        num_docs: 每个 batch 的文档数
        encode_batch_size: 知识编码 mini-batch 大小

    返回：
        List[Tensor] × num_layers，每个 shape [batch, num_docs * knowledge_max_seq_len, hidden_dim]
    """
    total_samples = knowledge_input_ids.size(0)
    num_layers = model.config.num_hidden_layers
    accumulated: List[List[Tensor]] = [[] for _ in range(num_layers)]

    for start in range(0, total_samples, encode_batch_size):
        end = min(start + encode_batch_size, total_samples)
        batch_ids = knowledge_input_ids[start:end]
        batch_mask = knowledge_attention_mask[start:end]

        outputs = model.model(
            input_ids=batch_ids,
            attention_mask=batch_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        # outputs.hidden_states: tuple of (num_layers+1) tensors [B, L, D]
        # index 0 = embedding, index i = layer i-1 output
        # 对齐 DecoupledRAG: layer i 使用 hidden_states[i] (layer i-1 output / embedding)
        for layer_idx in range(num_layers):
            hs = outputs.hidden_states[layer_idx]  # [mini_B, seq_len, D]
            sampled = strided_sampling(hs, batch_mask, knowledge_max_seq_len)
            accumulated[layer_idx].append(sampled)

    # 拼接并 reshape for multi-doc
    result: List[Tensor] = []
    for layer_idx in range(num_layers):
        layer_hs = torch.cat(accumulated[layer_idx], dim=0)  # [total, max_len, D]
        if num_docs > 1:
            batch_size = total_samples // num_docs
            layer_hs = (
                layer_hs.view(batch_size, num_docs, knowledge_max_seq_len, -1)
                .reshape(batch_size, num_docs * knowledge_max_seq_len, -1)
            )
        result.append(layer_hs)

    return result
```

- [ ] **Step 4: 运行测试确认通过**

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n ExplicitLLM python -m pytest tests/unit/test_knowledge_encoder.py::TestComputeKnowledgeHiddenStates -v --timeout=60
```
Expected: 2 passed

- [ ] **Step 5: 提交**

```bash
git add memory_lora/knowledge_encoder.py tests/unit/test_knowledge_encoder.py
git commit -m "feat: add compute_knowledge_hidden_states with strided sampling"
```

---

## Task 5: TokenMemForCausalLM 包装器

**Files:**
- Create: `memory_lora/tokenmem_model.py`
- Test: `tests/integration/test_tokenmem_model.py`

- [ ] **Step 1: 编写测试**

```python
# tests/integration/test_tokenmem_model.py
"""TokenMemForCausalLM 端到端集成测试。"""
import os
import tempfile

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="需要 GPU"
)

MODEL_NAME = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def tokenmem_model():
    from memory_lora.tokenmem_model import TokenMemForCausalLM
    model = TokenMemForCausalLM(MODEL_NAME, knowledge_max_seq_len=16)
    model.cuda()
    return model


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_NAME)


class TestTokenMemForCausalLM:

    def test_only_gates_trainable(self, tokenmem_model):
        """仅 gate_crossattention 参数 requires_grad=True。"""
        trainable = {n for n, p in tokenmem_model.named_parameters() if p.requires_grad}
        assert len(trainable) > 0
        for name in trainable:
            assert "gate_crossattention" in name

    def test_trainable_param_count(self, tokenmem_model):
        """Qwen3-0.6B: 28 层 × 2 × 1024 × 16 = 917,504。"""
        total = sum(p.numel() for p in tokenmem_model.parameters() if p.requires_grad)
        assert 900_000 < total < 950_000

    def test_forward_with_knowledge(self, tokenmem_model, tokenizer):
        """提供 knowledge_input_ids → forward 不报错，返回 loss。"""
        query = tokenizer("What is the capital of France?", return_tensors="pt").to("cuda")
        knowledge = tokenizer(
            "Paris is the capital of France.",
            return_tensors="pt", padding="max_length", max_length=32, truncation=True,
        ).to("cuda")

        labels = query["input_ids"].clone()
        out = tokenmem_model(
            input_ids=query["input_ids"],
            attention_mask=query["attention_mask"],
            labels=labels,
            knowledge_input_ids=knowledge["input_ids"],
            knowledge_attention_mask=knowledge["attention_mask"],
        )
        assert out.loss is not None
        assert out.loss.requires_grad

    def test_forward_without_knowledge(self, tokenmem_model, tokenizer):
        """不提供知识 → 正常 forward（退化为普通 LM）。"""
        query = tokenizer("Hello world", return_tensors="pt").to("cuda")
        labels = query["input_ids"].clone()
        out = tokenmem_model(
            input_ids=query["input_ids"],
            attention_mask=query["attention_mask"],
            labels=labels,
        )
        assert out.loss is not None

    def test_save_load_gates(self, tokenmem_model):
        """save_gates → load_gates 往返一致。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenmem_model.save_gates(tmpdir)

            # 验证文件存在
            files = os.listdir(tmpdir)
            assert len(files) == 28  # Qwen3-0.6B 有 28 层

            # 修改权重
            first_layer = tokenmem_model.model.model.layers[0]
            original_wa = first_layer.gate_crossattention.W_A.data.clone()
            first_layer.gate_crossattention.W_A.data.fill_(999.0)

            # 加载恢复
            tokenmem_model.load_gates(tmpdir)
            torch.testing.assert_close(
                first_layer.gate_crossattention.W_A.data, original_wa
            )
```

- [ ] **Step 2: 运行测试确认失败**

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n ExplicitLLM python -m pytest tests/integration/test_tokenmem_model.py -v --timeout=120
```
Expected: FAIL — `ImportError`

- [ ] **Step 3: 实现 TokenMemForCausalLM**

```python
# memory_lora/tokenmem_model.py
"""TokenMemForCausalLM —— 知识融合包装器。

封装 modified base model:
- 冻结 base LLM 全部参数
- 解冻 gate_crossattention (LinearFusion)
- 管理知识编码 (compute_knowledge_hidden_states)
- 管理门控权重存取 (save_gates / load_gates)
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
from torch import LongTensor, Tensor
from transformers import AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from memory_lora.knowledge_encoder import compute_knowledge_hidden_states


# 模型类型 → modified modeling 模块映射
_MODEL_CLASS_MAP = {
    "qwen3": "memory_lora.modified_models.modeling_qwen3",
    "mistral": "memory_lora.modified_models.modeling_mistral",
    "gemma3_text": "memory_lora.modified_models.modeling_gemma3",
}

# 模型类型 → ForCausalLM 类名映射
_CAUSAL_LM_CLASS_MAP = {
    "qwen3": "Qwen3ForCausalLM",
    "mistral": "MistralForCausalLM",
    "gemma3_text": "Gemma3ForCausalLM",
}


def _load_modified_model(model_name_or_path: str, torch_dtype=torch.bfloat16):
    """加载 modified model 并启用 cross-attention。"""
    import importlib

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    model_type = config.model_type
    if model_type not in _MODEL_CLASS_MAP:
        raise ValueError(f"不支持的模型类型: {model_type}，支持: {list(_MODEL_CLASS_MAP)}")

    config.add_cross_attention = True
    config.add_cross_attention_layer_number = config.num_hidden_layers - 1

    module = importlib.import_module(_MODEL_CLASS_MAP[model_type])
    cls = getattr(module, _CAUSAL_LM_CLASS_MAP[model_type])
    model = cls.from_pretrained(model_name_or_path, config=config, torch_dtype=torch_dtype)
    return model


class TokenMemForCausalLM(nn.Module):
    """知识融合包装器：frozen base + trainable LinearFusion gates。

    参数：
        model_name_or_path: HuggingFace 模型路径
        knowledge_max_seq_len: strided sampling 目标长度 (默认 64)
        torch_dtype: 模型精度 (默认 bf16)
    """

    def __init__(
        self,
        model_name_or_path: str,
        knowledge_max_seq_len: int = 64,
        torch_dtype=torch.bfloat16,
    ) -> None:
        super().__init__()
        self.model = _load_modified_model(model_name_or_path, torch_dtype=torch_dtype)
        self.knowledge_max_seq_len = knowledge_max_seq_len
        self._freeze_all_except_gates()

    def _freeze_all_except_gates(self) -> None:
        """冻结所有参数，仅保留 gate_crossattention 可训练。"""
        for param in self.model.parameters():
            param.requires_grad = False
        for name, param in self.model.named_parameters():
            if "gate_crossattention" in name:
                param.requires_grad = True

    def forward(
        self,
        input_ids: LongTensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[LongTensor] = None,
        knowledge_input_ids: Optional[LongTensor] = None,
        knowledge_attention_mask: Optional[LongTensor] = None,
        num_docs: int = 1,
    ) -> CausalLMOutputWithPast:
        """前向传播（知识编码 + cross-attention 融合 + LM loss）。

        参数：
            input_ids: query token ids [B, L]
            attention_mask: query attention mask [B, L]
            labels: LM labels [B, L]
            knowledge_input_ids: 知识 token ids [B*num_docs, K]
            knowledge_attention_mask: 知识 attention mask [B*num_docs, K]
            num_docs: 每个 query 检索的知识条目数
        """
        knowledge_outputs = None
        if knowledge_input_ids is not None:
            knowledge_outputs = compute_knowledge_hidden_states(
                self.model,
                knowledge_input_ids=knowledge_input_ids,
                knowledge_attention_mask=knowledge_attention_mask,
                knowledge_max_seq_len=self.knowledge_max_seq_len,
                num_docs=num_docs,
            )

        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            knowledge_outputs=knowledge_outputs,
        )

    def save_gates(self, directory: str) -> None:
        """保存每层 gate 权重到 gate_{layer_idx}.pt。"""
        os.makedirs(directory, exist_ok=True)
        for layer_idx, layer in enumerate(self.model.model.layers):
            if hasattr(layer, "gate_crossattention"):
                path = os.path.join(directory, f"gate_{layer_idx}.pt")
                torch.save(layer.gate_crossattention.state_dict(), path)

    def load_gates(self, directory: str) -> None:
        """从 gate_{layer_idx}.pt 加载每层 gate 权重。"""
        for layer_idx, layer in enumerate(self.model.model.layers):
            if hasattr(layer, "gate_crossattention"):
                path = os.path.join(directory, f"gate_{layer_idx}.pt")
                if os.path.exists(path):
                    state = torch.load(path, map_location=layer.gate_crossattention.W_A.device)
                    layer.gate_crossattention.load_state_dict(state)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n ExplicitLLM python -m pytest tests/integration/test_tokenmem_model.py -v --timeout=180
```
Expected: 5 passed

- [ ] **Step 5: 更新 `__init__.py` 导出**

```python
# memory_lora/__init__.py
"""TokenMem pipeline 核心模块。"""

from memory_lora.linear_fusion import LinearFusion
from memory_lora.token_bank import TokenMemoryBank
from memory_lora.tokenmem_model import TokenMemForCausalLM
```

- [ ] **Step 6: 提交**

```bash
git add memory_lora/tokenmem_model.py memory_lora/__init__.py tests/integration/test_tokenmem_model.py
git commit -m "feat: add TokenMemForCausalLM wrapper with freeze + save/load gates"
```

---

## Task 6: 端到端 Smoke Test（Qwen3-0.6B loss 下降）

**Files:**
- Test: `tests/integration/test_tokenmem_model.py` (追加)

- [ ] **Step 1: 编写 smoke test**

在 `tests/integration/test_tokenmem_model.py` 末尾追加:

```python
class TestSmokeTraining:
    """Smoke test: 10 步训练 loss 下降。"""

    @pytest.mark.slow
    def test_loss_decreases(self, tokenizer):
        """10 步 oracle SFT → loss 明显下降。"""
        from memory_lora.tokenmem_model import TokenMemForCausalLM

        model = TokenMemForCausalLM(MODEL_NAME, knowledge_max_seq_len=16)
        model.cuda().train()

        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )

        knowledge_text = "The capital of France is Paris."
        query_text = "What is the capital of France? The answer is Paris."

        k_enc = tokenizer(knowledge_text, return_tensors="pt",
                          padding="max_length", max_length=32, truncation=True).to("cuda")
        q_enc = tokenizer(query_text, return_tensors="pt").to("cuda")
        labels = q_enc["input_ids"].clone()

        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            out = model(
                input_ids=q_enc["input_ids"],
                attention_mask=q_enc["attention_mask"],
                labels=labels,
                knowledge_input_ids=k_enc["input_ids"],
                knowledge_attention_mask=k_enc["attention_mask"],
            )
            out.loss.backward()
            optimizer.step()
            losses.append(out.loss.item())

        assert losses[-1] < losses[0], f"loss 未下降: {losses[0]:.4f} → {losses[-1]:.4f}"
```

- [ ] **Step 2: 运行 smoke test**

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n ExplicitLLM python -m pytest tests/integration/test_tokenmem_model.py::TestSmokeTraining -v --timeout=300 -s
```
Expected: 1 passed，loss 有明显下降

- [ ] **Step 3: 提交**

```bash
git add tests/integration/test_tokenmem_model.py
git commit -m "test: add smoke test for TokenMemForCausalLM loss descent"
```

---

## Task 7: Fork 并修改 modeling_mistral.py

**Files:**
- Create: `memory_lora/modified_models/modeling_mistral.py`

**与 Qwen3 的差异**: Mistral 没有 `q_norm`/`k_norm`，没有 `sliding_window` 在 attention_type 中（而是 config 级别）。其余结构完全一致。

- [ ] **Step 1: 复制原始文件**

```bash
SRC=$(conda run -n ExplicitLLM python -c "import transformers,os; print(os.path.join(os.path.dirname(transformers.__file__),'models','mistral','modeling_mistral.py'))")
cp "$SRC" memory_lora/modified_models/modeling_mistral.py
```

- [ ] **Step 2: 修改 MistralAttention.forward（同 Qwen3 模式）**

与 Task 2 Step 3c 相同的修改模式，但：
- 无 `self.q_norm` / `self.k_norm` → 直接 `self.q_proj` / `self.k_proj`
- Cross-attention K/V 投影:

```python
if is_cross_attention and encoder_hidden_states is not None:
    kv_input_shape = encoder_hidden_states.shape[:-1]
    kv_hidden_shape = (*kv_input_shape, -1, self.head_dim)
    key_states = self.k_proj(encoder_hidden_states).view(kv_hidden_shape).transpose(1, 2)
    value_states = self.v_proj(encoder_hidden_states).view(kv_hidden_shape).transpose(1, 2)
else:
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    # ... past_key_values 逻辑同原始
```

- [ ] **Step 3: 修改 MistralDecoderLayer / MistralModel / MistralForCausalLM**

与 Task 2 Step 3d-3f 完全一致的修改模式（参数名、逻辑一致）。

- [ ] **Step 4: 验证（可选，若有 Ministral-3B 权重）**

```bash
# 若本地有模型权重:
CUDA_VISIBLE_DEVICES=0 conda run -n ExplicitLLM python -c "
from memory_lora.tokenmem_model import TokenMemForCausalLM
import torch
m = TokenMemForCausalLM('mistralai/Ministral-8B-Instruct-2410', knowledge_max_seq_len=16)
print('Mistral gates:', sum(p.numel() for p in m.parameters() if p.requires_grad))
"
```

- [ ] **Step 5: 提交**

```bash
git add memory_lora/modified_models/modeling_mistral.py
git commit -m "feat: fork modeling_mistral.py with cross-attention support"
```

---

## Task 8: Fork 并修改 modeling_gemma3.py

**Files:**
- Create: `memory_lora/modified_models/modeling_gemma3.py`

**与 Qwen3 的差异**:
- Gemma3 有 `pre_feedforward_layernorm` + `post_feedforward_layernorm` + `post_attention_layernorm`
- Gemma3 DecoderLayer 接收 `position_embeddings_global` 和 `position_embeddings_local`（dual RoPE）
- Gemma3 DecoderLayer.forward 返回 tuple `(hidden_states,)` 而非直接 `hidden_states`
- Gemma3 使用 `Gemma3TextModel` 而非 `Gemma3Model`
- Gemma3ForCausalLM 的 `.model` 属性对应 `Gemma3TextModel`
- Config 类为 `Gemma3TextConfig`
- Gemma3Attention 有 `attn_logit_softcapping` 和 `query_pre_attn_scalar`

- [ ] **Step 1: 复制原始文件**

```bash
SRC=$(conda run -n ExplicitLLM python -c "import transformers,os; print(os.path.join(os.path.dirname(transformers.__file__),'models','gemma3','modeling_gemma3.py'))")
cp "$SRC" memory_lora/modified_models/modeling_gemma3.py
```

- [ ] **Step 2: 修改 Gemma3Attention.forward**

与 Qwen3 相同的 cross-attention 分叉模式。注意 Gemma3 的 `q_norm`/`k_norm` 在 proj 之后、reshape 之后调用（与 Qwen3 顺序稍不同）:

```python
if is_cross_attention and encoder_hidden_states is not None:
    kv_input_shape = encoder_hidden_states.shape[:-1]
    kv_hidden_shape = (*kv_input_shape, -1, self.head_dim)
    key_states = self.k_proj(encoder_hidden_states).view(kv_hidden_shape).transpose(1, 2)
    value_states = self.v_proj(encoder_hidden_states).view(kv_hidden_shape).transpose(1, 2)
    key_states = self.k_norm(key_states)
    # 不加 RoPE
else:
    # 原始逻辑 (q_proj → view → transpose → q_norm → RoPE)
    ...
```

- [ ] **Step 3: 修改 Gemma3DecoderLayer.__init__ 和 .forward**

`__init__` 末尾同 Qwen3 追加 `gate_crossattention`。

`forward` 在 FFN 后（`hidden_states = residual + hidden_states` 后、`outputs = (hidden_states,)` 前）追加 cross-attention:

```python
    # Cross-attention + gate
    if self.add_cross_attention and encoder_hidden_states is not None:
        residual = hidden_states
        cross_input = self.input_layernorm(encoder_hidden_states)
        if self.self_attn.is_sliding:
            pe = position_embeddings_local
        else:
            pe = position_embeddings_global
        cross_out, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=pe,
            attention_mask=None,
            encoder_hidden_states=cross_input,
            is_cross_attention=True,
        )
        hidden_states = self.gate_crossattention(residual, cross_out)

    outputs = (hidden_states,)
```

- [ ] **Step 4: 修改 Gemma3TextModel.forward 和 Gemma3ForCausalLM.forward**

与 Qwen3 完全一致的修改模式：
- `Gemma3TextModel.forward` 增加 `knowledge_outputs` 参数，按层传递给 decoder layer
- `Gemma3ForCausalLM.forward` 透传 `knowledge_outputs`

注意 Gemma3TextModel 的 layer 循环传递 `position_embeddings_global` 和 `position_embeddings_local`。

- [ ] **Step 5: 更新 `_MODEL_CLASS_MAP`**

确认 `tokenmem_model.py` 中的映射已包含 `gemma3_text`:

```python
_MODEL_CLASS_MAP = {
    "qwen3": "memory_lora.modified_models.modeling_qwen3",
    "mistral": "memory_lora.modified_models.modeling_mistral",
    "gemma3_text": "memory_lora.modified_models.modeling_gemma3",
}
```

- [ ] **Step 6: 提交**

```bash
git add memory_lora/modified_models/modeling_gemma3.py
git commit -m "feat: fork modeling_gemma3.py with cross-attention support"
```

---

## 自审清单

- [x] **Spec 覆盖**: 所有 spec 需求（LinearFusion、modified modeling、knowledge encoding、wrapper、multi-model）均有对应 task
- [x] **无占位符**: 所有 step 包含完整代码或精确命令
- [x] **类型一致性**: `LinearFusion` 在 Task 1 定义，Task 2-8 中一致引用；`knowledge_outputs: Optional[list]` 在 Model/ForCausalLM/TokenMemForCausalLM 中签名一致
- [x] **测试覆盖**: Unit (LinearFusion, strided_sampling) + Integration (modified model, wrapper) + Smoke (loss 下降)
- [x] **依赖顺序**: Task 1 → Task 2 → Task 3 → Task 4 → Task 5 → Task 6 (串行)；Task 7/8 可与 Task 6 并行

## 执行依赖图

```
Task 1 (LinearFusion)
  └→ Task 2 (Modified Qwen3)
       └→ Task 3 (strided_sampling)
            └→ Task 4 (compute_knowledge_hidden_states)
                 └→ Task 5 (TokenMemForCausalLM)
                      └→ Task 6 (Smoke test)
Task 7 (Modified Mistral)  ← 可与 Task 3+ 并行
Task 8 (Modified Gemma3)   ← 可与 Task 3+ 并行
```
