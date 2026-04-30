"""CoT 数据生成脚本单元测试。"""

import pytest

from tools.generate_cot_data import (
    build_user_prompt,
    extract_cot_answer,
    SYSTEM_PROMPT,
)

NEWS_ROW = {
    "question": "Who discovered penicillin?",
    "passage": "Alexander Fleming discovered penicillin in 1928.",
    "options": {"A": "Pasteur", "B": "Fleming", "C": "Koch", "D": "Jenner"},
    "correct_letter": "B",
}

CF_ROW = {
    "question": "What color is the sky?",
    "options": {"A": "Red", "B": "Blue", "C": "Green", "D": "Yellow"},
    "counterfactual_passage": "In this world, the sky is red due to iron oxide.",
    "target_letter": "A",
}


class TestBuildUserPrompt:
    def test_contains_passage(self):
        prompt = build_user_prompt(NEWS_ROW["passage"], NEWS_ROW)
        assert NEWS_ROW["passage"] in prompt

    def test_contains_question_and_options(self):
        prompt = build_user_prompt(NEWS_ROW["passage"], NEWS_ROW)
        assert "Question: Who discovered penicillin?" in prompt
        assert "A. Pasteur" in prompt
        assert "D. Jenner" in prompt

    def test_contains_cot_instruction(self):
        prompt = build_user_prompt(NEWS_ROW["passage"], NEWS_ROW)
        assert "Let's think step by step" in prompt
        assert 'The answer is X' in prompt

    def test_contains_label_list(self):
        prompt = build_user_prompt(NEWS_ROW["passage"], NEWS_ROW)
        assert "A, B, C, or D" in prompt

    def test_cf_uses_same_template(self):
        prompt = build_user_prompt(CF_ROW["counterfactual_passage"], CF_ROW)
        assert CF_ROW["counterfactual_passage"] in prompt
        assert "Let's think step by step" in prompt


class TestSystemPrompt:
    def test_contains_only_instruction(self):
        assert "ONLY" in SYSTEM_PROMPT or "only" in SYSTEM_PROMPT

    def test_not_overly_specific(self):
        assert "alternate universe" not in SYSTEM_PROMPT
        assert "news" not in SYSTEM_PROMPT.lower()


class TestExtractCotAnswer:
    def test_standard_format(self):
        assert extract_cot_answer("Reasoning... The answer is B") == "B"

    def test_lowercase_the(self):
        assert extract_cot_answer("So the answer is C") == "C"

    def test_answer_colon_format(self):
        assert extract_cot_answer("After analysis, Answer: D") == "D"

    def test_no_match_returns_none(self):
        assert extract_cot_answer("I think B is correct") is None

    def test_multiple_matches_uses_last(self):
        text = "The answer is A... wait, The answer is C"
        result = extract_cot_answer(text)
        assert result == "C"
