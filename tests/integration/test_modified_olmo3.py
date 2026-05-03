"""Modified OLMo3 cross-attention 集成测试。"""
import pytest
import torch
from transformers import AutoTokenizer, AutoConfig

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="需要 GPU"
)

MODEL_NAME = "hugglingface_model/Olmo-3-7B-Instruct"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def modified_model():
    from memory_lora.modified_models.modeling_olmo3 import (
        Olmo3ForCausalLM as ModifiedOlmo3ForCausalLM,
    )
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.add_cross_attention = True
    config.add_cross_attention_layer_number = config.num_hidden_layers - 1
    model = ModifiedOlmo3ForCausalLM.from_pretrained(
        MODEL_NAME, config=config, torch_dtype=torch.bfloat16
    )
    # from_pretrained 的 _init_weights 会覆盖 LinearFusion 的自定义初始化，
    # 需要重新初始化 gate 参数（W_A: randn*0.01, W_B: zeros）
    for layer in model.model.layers:
        if hasattr(layer, "gate_crossattention"):
            gate = layer.gate_crossattention
            gate.W_A.data = torch.randn_like(gate.W_A) * 0.01
            gate.W_B.data.zero_()
    model.eval().cuda()
    return model


class TestModifiedOlmo3:
    def test_gate_zero_init_preserves_output(self, modified_model, tokenizer):
        """W_B 零初始化时，knowledge 注入不应改变输出。"""
        inputs = tokenizer("Hello world", return_tensors="pt").to("cuda")
        num_layers = modified_model.config.num_hidden_layers
        B, L = inputs["input_ids"].shape
        D = modified_model.config.hidden_size
        kv_len = 16
        knowledge_outputs = [
            torch.randn(B, kv_len, D, dtype=torch.bfloat16, device="cuda")
            for _ in range(num_layers)
        ]
        with torch.no_grad():
            out_no = modified_model(**inputs, knowledge_outputs=None)
            out_with = modified_model(**inputs, knowledge_outputs=knowledge_outputs)
        torch.testing.assert_close(out_no.logits, out_with.logits, atol=1e-4, rtol=1e-3)

    def test_nonzero_gate_changes_output(self, modified_model, tokenizer):
        """W_B 非零时，knowledge 注入应改变输出。"""
        inputs = tokenizer("Hello world", return_tensors="pt").to("cuda")
        num_layers = modified_model.config.num_hidden_layers
        B, L = inputs["input_ids"].shape
        D = modified_model.config.hidden_size
        kv_len = 16
        knowledge_outputs = [
            torch.randn(B, kv_len, D, dtype=torch.bfloat16, device="cuda")
            for _ in range(num_layers)
        ]
        for layer in modified_model.model.layers:
            if hasattr(layer, "gate_crossattention"):
                layer.gate_crossattention.W_B.data.fill_(0.1)
        with torch.no_grad():
            out_no = modified_model(**inputs, knowledge_outputs=None)
            out_with = modified_model(**inputs, knowledge_outputs=knowledge_outputs)
        assert not torch.allclose(out_no.logits, out_with.logits)
        # 恢复零初始化
        for layer in modified_model.model.layers:
            if hasattr(layer, "gate_crossattention"):
                layer.gate_crossattention.W_B.data.zero_()

    def test_all_layers_have_gate(self, modified_model):
        """所有层都应有 gate_crossattention 属性。"""
        for layer in modified_model.model.layers:
            assert hasattr(layer, "gate_crossattention")

    def test_gate_params_are_only_trainable(self, modified_model):
        """gate_crossattention 参数应为 requires_grad=True。"""
        for name, p in modified_model.named_parameters():
            if "gate_crossattention" in name:
                assert p.requires_grad
