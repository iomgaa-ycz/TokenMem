"""知识编码工具函数测试。"""
import pytest
import torch

from memory_lora.knowledge_encoder import strided_sampling


class TestStridedSampling:
    def test_short_input_pads_to_max_length(self):
        hidden = torch.randn(1, 10, 64)
        mask = torch.ones(1, 10)
        out = strided_sampling(hidden, mask, max_length=16)
        assert out.shape == (1, 16, 64)

    def test_long_input_samples_to_max_length(self):
        hidden = torch.arange(128).float().unsqueeze(0).unsqueeze(-1).expand(1, 128, 4)
        mask = torch.ones(1, 128)
        out = strided_sampling(hidden, mask, max_length=32)
        assert out.shape == (1, 32, 4)

    def test_exact_length_no_change(self):
        hidden = torch.randn(1, 64, 32)
        mask = torch.ones(1, 64)
        out = strided_sampling(hidden, mask, max_length=64)
        assert out.shape == (1, 64, 32)

    def test_left_padding_handled(self):
        hidden = torch.randn(1, 20, 8)
        mask = torch.zeros(1, 20)
        mask[0, 10:] = 1
        out = strided_sampling(hidden, mask, max_length=16)
        assert out.shape == (1, 16, 8)
        assert torch.all(out[0, :6, :] == 0)

    def test_batch_dimension(self):
        hidden = torch.randn(3, 50, 16)
        mask = torch.ones(3, 50)
        mask[1, :30] = 0
        out = strided_sampling(hidden, mask, max_length=32)
        assert out.shape == (3, 32, 16)


class TestComputeKnowledgeHiddenStates:
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
        assert result[0].shape == (1, 16, qwen3_model.config.hidden_size)
