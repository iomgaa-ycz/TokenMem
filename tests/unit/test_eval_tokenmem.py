"""eval_tokenmem 核心函数单元测试。

覆盖：
- build_cot_prompt (passage=None)：TokenMem 用法，无 passage 拼入 prompt
- tokenize_knowledge：返回正确 key + shape
- normalize_options（从 eval_baseline import）
"""

from transformers import AutoTokenizer

from evaluation.eval_baseline import build_cot_prompt, normalize_options
from evaluation.eval_tokenmem import tokenize_knowledge

_QWEN3_PATH = "hugglingface_model/qwen3-0.6B"


class TestBuildCotPromptTokenMem:
    """验证 TokenMem 使用 build_cot_prompt(passage=None) 的行为。"""

    def test_no_passage_in_prompt(self) -> None:
        """passage=None 时 prompt 中不应含有任何段落文本。"""
        options = {"A": "Paris", "B": "London", "C": "Berlin", "D": "Rome"}
        prompt = build_cot_prompt(
            question="What is the capital of France?",
            options=options,
            passage=None,
        )
        assert "Reference:" not in prompt
        assert "Paris" in prompt

    def test_contains_cot_instruction(self) -> None:
        """prompt 必须包含 CoT 指令。"""
        options = {"A": "a", "B": "b"}
        prompt = build_cot_prompt(question="Test?", options=options, passage=None)
        assert "step by step" in prompt
        assert "The answer is" in prompt

    def test_contains_question_and_options(self) -> None:
        """prompt 中包含问题和所有选项。"""
        options = {"A": "Alpha", "B": "Beta", "C": "Gamma"}
        prompt = build_cot_prompt(question="Pick one?", options=options, passage=None)
        assert "Pick one?" in prompt
        assert "A. Alpha" in prompt
        assert "B. Beta" in prompt
        assert "C. Gamma" in prompt


class TestTokenizeKnowledge:
    """验证 tokenize_knowledge 返回正确的 key 和 tensor shape。"""

    def test_returns_required_keys(self) -> None:
        """返回字典必须含有 knowledge_input_ids 和 knowledge_attention_mask。"""
        tokenizer = AutoTokenizer.from_pretrained(_QWEN3_PATH, trust_remote_code=True)
        result = tokenize_knowledge(
            tokenizer=tokenizer,
            passage="This is a test passage about medicine.",
            max_len=64,
            device="cpu",
        )
        assert "knowledge_input_ids" in result
        assert "knowledge_attention_mask" in result

    def test_shape_batch_dim_is_one(self) -> None:
        """knowledge_input_ids shape 第一维应为 1（batch size = 1）。"""
        tokenizer = AutoTokenizer.from_pretrained(_QWEN3_PATH, trust_remote_code=True)
        result = tokenize_knowledge(
            tokenizer=tokenizer,
            passage="Short passage.",
            max_len=128,
            device="cpu",
        )
        assert result["knowledge_input_ids"].shape[0] == 1
        assert result["knowledge_attention_mask"].shape[0] == 1

    def test_max_len_truncation(self) -> None:
        """超长 passage 应被截断到 max_len。"""
        tokenizer = AutoTokenizer.from_pretrained(_QWEN3_PATH, trust_remote_code=True)
        long_passage = "word " * 500
        result = tokenize_knowledge(
            tokenizer=tokenizer,
            passage=long_passage,
            max_len=32,
            device="cpu",
        )
        assert result["knowledge_input_ids"].shape[1] <= 32
        assert result["knowledge_attention_mask"].shape[1] <= 32

    def test_input_ids_and_mask_same_shape(self) -> None:
        """knowledge_input_ids 和 knowledge_attention_mask 形状必须相同。"""
        tokenizer = AutoTokenizer.from_pretrained(_QWEN3_PATH, trust_remote_code=True)
        result = tokenize_knowledge(
            tokenizer=tokenizer,
            passage="Test passage for shape check.",
            max_len=64,
            device="cpu",
        )
        assert (
            result["knowledge_input_ids"].shape
            == result["knowledge_attention_mask"].shape
        )

    def test_empty_passage(self) -> None:
        """空 passage 不应抛出异常。"""
        tokenizer = AutoTokenizer.from_pretrained(_QWEN3_PATH, trust_remote_code=True)
        result = tokenize_knowledge(
            tokenizer=tokenizer,
            passage="",
            max_len=64,
            device="cpu",
        )
        assert result["knowledge_input_ids"].shape[0] == 1


class TestNormalizeOptions:
    """验证 normalize_options 的 key 归一化逻辑（从 eval_baseline import）。"""

    def test_numeric_keys_to_letters(self) -> None:
        options = {"1": "opt1", "2": "opt2", "3": "opt3", "4": "opt4"}
        norm_opts, norm_letter = normalize_options(options, "2")
        assert set(norm_opts.keys()) == {"A", "B", "C", "D"}
        assert norm_letter == "B"

    def test_letter_keys_unchanged(self) -> None:
        options = {"A": "alpha", "B": "beta", "C": "gamma"}
        norm_opts, norm_letter = normalize_options(options, "C")
        assert set(norm_opts.keys()) == {"A", "B", "C"}
        assert norm_letter == "C"
