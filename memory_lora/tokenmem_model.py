"""TokenMemForCausalLM —— 知识融合包装器。

封装 modified base model:
- 冻结 base LLM 全部参数
- 解冻 gate_crossattention (LinearFusion)
- 管理知识编码 (compute_knowledge_hidden_states)
- 管理门控权重存取 (save_gates / load_gates)
"""
from __future__ import annotations

import os
import importlib
from typing import Optional, List

import torch
import torch.nn as nn
from torch import LongTensor, Tensor
from transformers import AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from memory_lora.knowledge_encoder import compute_knowledge_hidden_states
from memory_lora.linear_fusion import LinearFusion


_MODEL_CLASS_MAP = {
    "qwen3": "memory_lora.modified_models.modeling_qwen3",
    "gemma3_text": "memory_lora.modified_models.modeling_gemma3",
    "ministral": "memory_lora.modified_models.modeling_ministral",
}

_CAUSAL_LM_CLASS_MAP = {
    "qwen3": "Qwen3ForCausalLM",
    "gemma3_text": "Gemma3ForCausalLM",
    "ministral": "MinistralForCausalLM",
}


def _load_modified_model(model_name_or_path: str, torch_dtype=torch.bfloat16):
    """加载 modified model 并启用 cross-attention。

    参数：
        model_name_or_path: HuggingFace 模型路径或名称
        torch_dtype: 模型权重精度

    返回：
        加载完成的 CausalLM 模型实例
    """
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    model_type = config.model_type
    if model_type not in _MODEL_CLASS_MAP:
        raise ValueError(
            f"不支持的模型类型: {model_type}，当前支持: {list(_MODEL_CLASS_MAP)}"
        )

    config.add_cross_attention = True
    config.add_cross_attention_layer_number = config.num_hidden_layers - 1

    module = importlib.import_module(_MODEL_CLASS_MAP[model_type])
    cls = getattr(module, _CAUSAL_LM_CLASS_MAP[model_type])
    model = cls.from_pretrained(model_name_or_path, config=config, torch_dtype=torch_dtype)
    return model


class TokenMemForCausalLM(nn.Module):
    """知识融合包装器：frozen base LLM + trainable LinearFusion gates。

    参数：
        model_name_or_path: HuggingFace 模型路径或名称
        knowledge_max_seq_len: strided sampling 目标长度 (默认 64)
        torch_dtype: 模型精度 (默认 bf16)

    用法示例：
        model = TokenMemForCausalLM("Qwen/Qwen3-0.6B")
        model.cuda()
        out = model(input_ids=..., knowledge_input_ids=..., labels=...)
        loss = out.loss
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
        # from_pretrained 的 _init_weights 会覆盖 LinearFusion 的自定义初始化，
        # 必须在加载后重新初始化所有 gate_crossattention
        self._reinit_gates()
        self._freeze_all_except_gates()

    def _reinit_gates(self) -> None:
        """重新初始化所有 gate_crossattention 模块，恢复正确的低秩初始化。

        from_pretrained 调用 _init_weights 会将 LinearFusion 的参数覆盖为
        标准 Xavier/kaiming 初始化，破坏 W_A(Gaussian σ=0.01) 和 W_B(zeros) 的约定。
        通过创建新的 LinearFusion 实例来修复。
        """
        for layer in self.model.model.layers:
            if hasattr(layer, "gate_crossattention"):
                hidden_size = layer.hidden_size
                device = layer.gate_crossattention.W_A.device
                dtype = layer.gate_crossattention.W_A.dtype
                layer.gate_crossattention = LinearFusion(hidden_size)
                layer.gate_crossattention.to(device=device, dtype=dtype)

    def _freeze_all_except_gates(self) -> None:
        """冻结所有参数，仅保留 gate_crossattention 模块可训练。

        冻结策略：
        - 所有 base LLM 参数 requires_grad=False
        - gate_crossattention.W_A 和 W_B requires_grad=True
        """
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
        """前向传播：知识编码 + cross-attention 融合 + LM loss。

        参数：
            input_ids: query token ids [B, L]
            attention_mask: query attention mask [B, L]
            labels: LM labels [B, L]，用于计算 cross-entropy loss
            knowledge_input_ids: 知识 token ids [B*num_docs, K]，None 时跳过知识编码
            knowledge_attention_mask: 知识 attention mask [B*num_docs, K]
            num_docs: 每个 query 检索的知识条目数 (默认 1)

        返回：
            CausalLMOutputWithPast，包含 loss / logits / past_key_values
        """
        knowledge_outputs: Optional[List[Tensor]] = None
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
        """保存每层 gate 权重到 {directory}/gate_{layer_idx}.pt。

        参数：
            directory: 权重保存目录（不存在时自动创建）
        """
        os.makedirs(directory, exist_ok=True)
        for layer_idx, layer in enumerate(self.model.model.layers):
            if hasattr(layer, "gate_crossattention"):
                path = os.path.join(directory, f"gate_{layer_idx}.pt")
                torch.save(layer.gate_crossattention.state_dict(), path)

    def load_gates(self, directory: str) -> None:
        """从 {directory}/gate_{layer_idx}.pt 加载每层 gate 权重。

        参数：
            directory: 权重加载目录
        """
        for layer_idx, layer in enumerate(self.model.model.layers):
            if hasattr(layer, "gate_crossattention"):
                path = os.path.join(directory, f"gate_{layer_idx}.pt")
                if os.path.exists(path):
                    state = torch.load(
                        path,
                        map_location=layer.gate_crossattention.W_A.device,
                    )
                    layer.gate_crossattention.load_state_dict(state)
