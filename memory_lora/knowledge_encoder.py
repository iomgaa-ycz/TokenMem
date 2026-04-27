"""知识编码工具 —— strided sampling + 逐层 hidden states 计算。

参考 DecoupledRAG (WWW 2025) 的知识编码管线:
- strided_sampling: 按 attention_mask 有效长度均匀采样到目标长度
- compute_knowledge_hidden_states: frozen LLM 全层 forward → 逐层 hidden states
"""
from __future__ import annotations

from typing import List

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

    DecoupledRAG 约定: layer i 使用 hidden_states[i]（索引从 0 开始，
    hidden_states[0] 为 embedding 层输出，hidden_states[i] 为第 i 层 Transformer block 输出）。

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

        # hidden_states 是 (num_layers+1) 个 tensor 的 tuple
        # 索引 [0..num_layers-1]: DecoupledRAG 约定 layer i 用 hidden_states[i]
        for layer_idx in range(num_layers):
            hs = outputs.hidden_states[layer_idx]
            sampled = strided_sampling(hs, batch_mask, knowledge_max_seq_len)
            accumulated[layer_idx].append(sampled)

    result: List[Tensor] = []
    for layer_idx in range(num_layers):
        layer_hs = torch.cat(accumulated[layer_idx], dim=0)
        if num_docs > 1:
            batch_size = total_samples // num_docs
            layer_hs = (
                layer_hs.view(batch_size, num_docs, knowledge_max_seq_len, -1)
                .reshape(batch_size, num_docs * knowledge_max_seq_len, -1)
            )
        result.append(layer_hs)

    return result
