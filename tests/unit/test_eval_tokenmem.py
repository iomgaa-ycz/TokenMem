"""eval_tokenmem 核心函数单元测试。

覆盖：
- build_mc_prompt：不含 Reference 前缀，以 Answer: 结尾
- tokenize_knowledge：返回正确 key + shape
- normalize_options：数字 key 转字母，字母 key 不变
"""

from transformers import AutoTokenizer

from evaluation.eval_tokenmem import (
    build_mc_prompt,
    normalize_options,
    tokenize_knowledge,
)

_QWEN3_PATH = "hugglingface_model/qwen3-0.6B"


class TestBuildMcPromptNoPassage:
    """验证 TokenMem prompt 格式：无 Reference 前缀，以 Answer: 结尾。"""

    def test_no_reference_prefix(self) -> None:
        """prompt 中不应含有 'Reference:' 字符串。"""
        options = {"A": "Paris", "B": "London", "C": "Berlin", "D": "Rome"}
        prompt = build_mc_prompt(
            question="What is the capital of France?",
            options=options,
        )
        assert "Reference:" not in prompt

    def test_ends_with_answer(self) -> None:
        """prompt 必须以 'Answer:' 结尾。"""
        options = {"A": "Paris", "B": "London", "C": "Berlin", "D": "Rome"}
        prompt = build_mc_prompt(
            question="What is the capital of France?",
            options=options,
        )
        assert prompt.endswith("Answer:")

    def test_starts_with_question(self) -> None:
        """prompt 必须以 'Question:' 开头。"""
        options = {"A": "a", "B": "b"}
        prompt = build_mc_prompt(question="Test?", options=options)
        assert prompt.startswith("Question:")

    def test_options_present(self) -> None:
        """所有选项行出现在 prompt 中。"""
        options = {"A": "Alpha", "B": "Beta", "C": "Gamma"}
        prompt = build_mc_prompt(question="Pick one?", options=options)
        assert "A. Alpha" in prompt
        assert "B. Beta" in prompt
        assert "C. Gamma" in prompt

    def test_three_options(self) -> None:
        """支持 3 个选项（非 4 个）。"""
        options = {"A": "one", "B": "two", "C": "three"}
        prompt = build_mc_prompt(question="Q?", options=options)
        assert prompt.endswith("Answer:")
        assert "A. one" in prompt


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
        long_passage = "word " * 500  # ~500 tokens
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
    """验证 normalize_options 的 key 归一化逻辑。"""

    def test_numeric_keys_to_letters(self) -> None:
        """数字 key（"1","2","3","4"）应转为字母（A/B/C/D）。"""
        options = {"1": "opt1", "2": "opt2", "3": "opt3", "4": "opt4"}
        norm_opts, norm_letter = normalize_options(options, "2")
        assert set(norm_opts.keys()) == {"A", "B", "C", "D"}
        assert norm_letter == "B"

    def test_letter_keys_unchanged(self) -> None:
        """字母 key 不应被改变。"""
        options = {"A": "alpha", "B": "beta", "C": "gamma"}
        norm_opts, norm_letter = normalize_options(options, "C")
        assert set(norm_opts.keys()) == {"A", "B", "C"}
        assert norm_letter == "C"

    def test_numeric_correct_letter_mapped(self) -> None:
        """correct_letter 为数字时应同步转为字母。"""
        options = {"1": "a", "2": "b", "3": "c"}
        _, norm_letter = normalize_options(options, "3")
        assert norm_letter == "C"

    def test_values_preserved(self) -> None:
        """数字转字母后，选项值不应改变。"""
        options = {"1": "Apple", "2": "Banana"}
        norm_opts, _ = normalize_options(options, "1")
        assert norm_opts["A"] == "Apple"
        assert norm_opts["B"] == "Banana"
