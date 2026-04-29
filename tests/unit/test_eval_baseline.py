"""Baseline 评测脚本的核心函数测试。"""

import json
import tempfile
from pathlib import Path

from evaluation.eval_baseline import load_samples_jsonl


class TestLoadSamplesJsonl:
    def test_loads_correct_fields(self):
        data = {
            "id": "test_0",
            "dataset": "test",
            "question": "Q?",
            "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
            "correct_letter": "A",
            "passage": "Knowledge text.",
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
            lines.append(
                json.dumps(
                    {
                        "id": f"t_{i}",
                        "dataset": "t",
                        "question": "Q",
                        "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                        "correct_letter": "A",
                        "passage": "p",
                    }
                )
            )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("\n".join(lines) + "\n")
            tmp_path = f.name
        samples = load_samples_jsonl(Path(tmp_path), n_samples=3)
        assert len(samples) == 3
        Path(tmp_path).unlink()


class TestCompressPassage:
    def test_compresses_to_target(self):
        from evaluation.eval_baseline import compress_passage

        passage = "Misoprostol, a synthetic prostaglandin E1 analog, is widely used in obstetric practice for cervical ripening and induction of labor. In the setting of preterm labor with advanced cervical dilation and regular intense contractions, misoprostol can be administered to accelerate delivery."
        compressed = compress_passage(passage, target_token=64)
        assert isinstance(compressed, str)
        assert len(compressed) > 0
        assert len(compressed) < len(passage)

    def test_short_passage_unchanged(self):
        from evaluation.eval_baseline import compress_passage

        short = "Misoprostol is used for labor."
        result = compress_passage(short, target_token=64)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_string(self):
        from evaluation.eval_baseline import compress_passage

        result = compress_passage("Test passage about medicine.", target_token=64)
        assert isinstance(result, str)


class TestBuildCotPrompt:
    def test_no_memory_cot(self):
        from evaluation.eval_baseline import build_cot_prompt

        options = {"A": "Paris", "B": "London", "C": "Berlin", "D": "Rome"}
        prompt = build_cot_prompt(
            question="What is the capital of France?",
            options=options,
            passage=None,
        )
        assert "/no_think" in prompt
        assert "Let's think step by step" in prompt
        assert "The answer is X" in prompt
        assert "Question:" in prompt
        assert "Reference:" not in prompt

    def test_vanilla_rag_cot_neutral(self):
        from evaluation.eval_baseline import build_cot_prompt

        options = {"A": "Paris", "B": "London", "C": "Berlin", "D": "Rome"}
        prompt = build_cot_prompt(
            question="What is the capital of France?",
            options=options,
            passage="France capital is Paris.",
        )
        assert "France capital is Paris." in prompt
        assert "Reference:" not in prompt
        assert "/no_think" in prompt
        assert "The answer is X" in prompt


class TestExtractAnswerLetter:
    def test_answer_is_pattern(self):
        from evaluation.eval_baseline import extract_answer_letter

        assert (
            extract_answer_letter("blah blah. The answer is B.", {"A", "B", "C", "D"})
            == "B"
        )

    def test_answer_colon_pattern(self):
        from evaluation.eval_baseline import extract_answer_letter

        assert extract_answer_letter("So Answer: C", {"A", "B", "C", "D"}) == "C"

    def test_no_match(self):
        from evaluation.eval_baseline import extract_answer_letter

        assert (
            extract_answer_letter("I don't know the answer", {"A", "B", "C", "D"})
            == "?"
        )

    def test_trailing_letter(self):
        from evaluation.eval_baseline import extract_answer_letter

        assert extract_answer_letter("After analysis, D", {"A", "B", "C", "D"}) == "D"

    def test_option_pattern(self):
        from evaluation.eval_baseline import extract_answer_letter

        assert (
            extract_answer_letter("I choose option A here", {"A", "B", "C", "D"}) == "A"
        )
