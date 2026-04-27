"""TokenMemForCausalLM 端到端集成测试。

验证：
- 参数冻结策略（仅 gate_crossattention 可训练）
- 可训练参数量符合预期
- _reinit_gates 正确恢复 W_A/W_B 初始化
- 有知识输入的前向传播（含 loss）
- 无知识输入的前向传播
- save/load gates 权重持久化
"""
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
    """初始化并移到 GPU 的 TokenMemForCausalLM。"""
    from memory_lora.tokenmem_model import TokenMemForCausalLM
    model = TokenMemForCausalLM(MODEL_NAME, knowledge_max_seq_len=16)
    model.cuda()
    return model


@pytest.fixture(scope="module")
def tokenizer():
    """加载 Qwen3-0.6B tokenizer。"""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_NAME)


class TestTokenMemForCausalLM:

    def test_only_gates_trainable(self, tokenmem_model):
        """仅 gate_crossattention 参数应设置 requires_grad=True。"""
        trainable = {n for n, p in tokenmem_model.named_parameters() if p.requires_grad}
        assert len(trainable) > 0, "应有可训练参数"
        for name in trainable:
            assert "gate_crossattention" in name, (
                f"非 gate 参数被设为可训练: {name}"
            )

    def test_trainable_param_count(self, tokenmem_model):
        """可训练参数量应接近 28 * 2 * 1024 * 16 = 917,504。

        28 层 × (W_A: 1024*16 + W_B: 16*1024) = 28 × 32768 = 917,504
        """
        total = sum(p.numel() for p in tokenmem_model.parameters() if p.requires_grad)
        assert 900_000 < total < 950_000, (
            f"可训练参数量异常: {total}，预期约 917,504"
        )

    def test_gate_zero_init_correct(self, tokenmem_model):
        """验证 _reinit_gates 恢复了正确的初始化：W_B 全零，W_A 接近 σ=0.01。"""
        for layer in tokenmem_model.model.model.layers:
            if hasattr(layer, "gate_crossattention"):
                assert torch.all(layer.gate_crossattention.W_B == 0), (
                    "W_B 应为全零初始化"
                )
                assert layer.gate_crossattention.W_A.std() < 0.05, (
                    f"W_A std 异常: {layer.gate_crossattention.W_A.std():.4f}，预期 ~0.01"
                )

    def test_forward_with_knowledge(self, tokenmem_model, tokenizer):
        """有知识输入时前向传播应产生带梯度的 loss。"""
        query = tokenizer(
            "What is the capital of France?", return_tensors="pt"
        ).to("cuda")
        knowledge = tokenizer(
            "Paris is the capital of France.",
            return_tensors="pt",
            padding="max_length",
            max_length=32,
            truncation=True,
        ).to("cuda")
        labels = query["input_ids"].clone()

        out = tokenmem_model(
            input_ids=query["input_ids"],
            attention_mask=query["attention_mask"],
            labels=labels,
            knowledge_input_ids=knowledge["input_ids"],
            knowledge_attention_mask=knowledge["attention_mask"],
        )

        assert out.loss is not None, "有知识输入时 loss 不应为 None"
        assert out.loss.requires_grad, "loss 应有梯度（gate 参数可训练）"
        assert not torch.isnan(out.loss), "loss 不应为 NaN"

    def test_forward_without_knowledge(self, tokenmem_model, tokenizer):
        """无知识输入时前向传播应正常完成并产生 loss。"""
        query = tokenizer("Hello world", return_tensors="pt").to("cuda")
        labels = query["input_ids"].clone()

        out = tokenmem_model(
            input_ids=query["input_ids"],
            attention_mask=query["attention_mask"],
            labels=labels,
        )

        assert out.loss is not None, "无知识输入时 loss 不应为 None"
        assert not torch.isnan(out.loss), "loss 不应为 NaN"

    def test_save_load_gates(self, tokenmem_model):
        """save_gates 应写出 28 个文件，load_gates 应恢复权重。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenmem_model.save_gates(tmpdir)
            files = os.listdir(tmpdir)
            assert len(files) == 28, (
                f"应保存 28 个 gate 文件，实际: {len(files)}"
            )

            # 篡改第 0 层权重，验证 load_gates 能恢复
            first_layer = tokenmem_model.model.model.layers[0]
            original_wa = first_layer.gate_crossattention.W_A.data.clone()
            first_layer.gate_crossattention.W_A.data.fill_(999.0)

            tokenmem_model.load_gates(tmpdir)
            torch.testing.assert_close(
                first_layer.gate_crossattention.W_A.data,
                original_wa,
                msg="load_gates 应恢复被篡改的 W_A 权重",
            )
