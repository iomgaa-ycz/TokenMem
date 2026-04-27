"""Baseline 评测脚本的核心函数测试。"""
import json
import tempfile
from pathlib import Path

import pytest
import torch

from evaluation.eval_baseline import (
    build_mc_prompt,
    evaluate_logprob,
    load_samples_jsonl,
)


class TestBuildMcPrompt:
    def test_no_memory(self):
        options = {"A": "Paris", "B": "London", "C": "Berlin", "D": "Rome"}
        prompt = build_mc_prompt(
            question="What is the capital of France?",
            options=options,
            passage=None,
        )
        assert prompt.startswith("Question:")
        assert "A. Paris" in prompt
        assert prompt.endswith("Answer:")
        assert "Reference:" not in prompt

    def test_vanilla_rag(self):
        options = {"A": "Paris", "B": "London", "C": "Berlin", "D": "Rome"}
        prompt = build_mc_prompt(
            question="What is the capital of France?",
            options=options,
            passage="France is a country in Europe. Its capital is Paris.",
        )
        assert prompt.startswith("Reference:")
        assert "A. Paris" in prompt
        assert prompt.endswith("Answer:")


class TestLoadSamplesJsonl:
    def test_loads_correct_fields(self):
        data = {
            "id": "test_0", "dataset": "test", "question": "Q?",
            "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
            "correct_letter": "A", "passage": "Knowledge text.",
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(data) + "\n")
            tmp_path = f.name
        samples = load_samples_jsonl(Path(tmp_path))
        assert len(samples) == 1
        assert samples[0]["correct_letter"] == "A"
        Path(tmp_path).unlink()

    def test_n_samples_truncation(self):
        lines = []
        for i in range(10):
            lines.append(json.dumps({
                "id": f"t_{i}", "dataset": "t", "question": "Q",
                "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                "correct_letter": "A", "passage": "p",
            }))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("\n".join(lines) + "\n")
            tmp_path = f.name
        samples = load_samples_jsonl(Path(tmp_path), n_samples=3)
        assert len(samples) == 3
        Path(tmp_path).unlink()


class TestEvaluateLogprob:
    """用 tiny random model 验证 logprob 评分函数的形状和返回值。"""

    @pytest.fixture
    def tiny_model_and_tokenizer(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        name = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        tokenizer = AutoTokenizer.from_pretrained(name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(name)
        model.eval()
        return model, tokenizer

    def test_returns_valid_index(self, tiny_model_and_tokenizer):
        model, tokenizer = tiny_model_and_tokenizer
        prompt = "Question: What?\nA. a\nB. b\nC. c\nD. d\nAnswer:"
        pred = evaluate_logprob(model, tokenizer, prompt, device="cpu")
        assert pred in [0, 1, 2, 3]
