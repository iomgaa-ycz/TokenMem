"""NewsQAOracleDataset + make_collate_fn 单元测试。"""

import json
from pathlib import Path
from typing import List

import pytest
import torch
from transformers import AutoTokenizer

from training.data import (
    CounterfactualDataset,
    NewsQAOracleDataset,
    OversampledDataset,
    make_collate_fn,
)

# ---------------------------------------------------------------------------
# 公共 fixtures & mock 数据
# ---------------------------------------------------------------------------

MOCK_DATA: List[dict] = [
    {
        "question": "What color is the sky?",
        "passage": "The sky appears blue due to Rayleigh scattering of sunlight.",
        "correct_answer": "Blue",
        "correct_letter": "B",
        "options": {"A": "Red", "B": "Blue", "C": "Green", "D": "Yellow"},
    },
    {
        "question": "What is 2+2?",
        "passage": "Basic arithmetic: two plus two equals four.",
        "correct_answer": "4",
        "correct_letter": "C",
        "options": {"A": "3", "B": "5", "C": "4", "D": "6"},
    },
    {
        "question": "Which planet is closest to the Sun?",
        "passage": "Mercury orbits closest to the Sun at about 58 million km.",
        "correct_answer": "Mercury",
        "correct_letter": "A",
        "options": {"A": "Mercury", "B": "Venus", "C": "Earth", "D": "Mars"},
    },
]


MOCK_CF_DATA: List[dict] = [
    {
        "cf_id": "arc_00_cf_B",
        "source_id": "arc_00",
        "dataset": "arc_easy",
        "question": "What color is the sky?",
        "options": {"A": "Red", "B": "Blue", "C": "Green", "D": "Yellow"},
        "correct_letter": "B",
        "passage": "The sky appears blue due to Rayleigh scattering.",
        "target_letter": "A",
        "target_answer": "Red",
        "counterfactual_passage": "The sky appears red due to iron oxide particles.",
        "split": "train",
    },
    {
        "cf_id": "arc_00_cf_C",
        "source_id": "arc_00",
        "dataset": "arc_easy",
        "question": "What color is the sky?",
        "options": {"A": "Red", "B": "Blue", "C": "Green", "D": "Yellow"},
        "correct_letter": "B",
        "passage": "The sky appears blue due to Rayleigh scattering.",
        "target_letter": "C",
        "target_answer": "Green",
        "counterfactual_passage": "The sky appears green due to chlorophyll absorption.",
        "split": "train",
    },
    {
        "cf_id": "arc_01_cf_A",
        "source_id": "arc_01",
        "dataset": "arc_easy",
        "question": "What is 2+2?",
        "options": {"A": "3", "B": "5", "C": "4", "D": "6"},
        "correct_letter": "C",
        "passage": "Basic arithmetic: two plus two equals four.",
        "target_letter": "A",
        "target_answer": "3",
        "counterfactual_passage": "In modular arithmetic base 3, two plus two equals three.",
        "split": "test",
    },
]


@pytest.fixture()
def cf_jsonl_path(tmp_path: Path) -> Path:
    """写入 3 条反事实 JSONL（2 train + 1 test）。"""
    p = tmp_path / "cf.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for row in MOCK_CF_DATA:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return p


@pytest.fixture()
def jsonl_path(tmp_path: Path) -> Path:
    """写入标准 3 行 JSONL 临时文件。"""
    p = tmp_path / "qa.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for row in MOCK_DATA:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return p


@pytest.fixture()
def jsonl_with_blanks(tmp_path: Path) -> Path:
    """包含空行的 JSONL 文件（3 条有效数据 + 2 空行）。"""
    p = tmp_path / "qa_blanks.jsonl"
    with p.open("w", encoding="utf-8") as f:
        f.write("\n")  # 开头空行
        for row in MOCK_DATA:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.write("\n")  # 末尾空行
    return p


@pytest.fixture()
def jsonl_with_compressed(tmp_path: Path) -> Path:
    """带 compressed_text 字段的 JSONL。"""
    p = tmp_path / "qa_compressed.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for row in MOCK_DATA:
            row_copy = dict(row)
            row_copy["compressed_text"] = "COMPRESSED: " + row["passage"]
            f.write(json.dumps(row_copy, ensure_ascii=False) + "\n")
    return p


@pytest.fixture()
def tokenizer():
    """加载 Qwen3-0.6B tokenizer 并设置 pad_token。"""
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ===================================================================
# Task 1: TestNewsQAOracleDataset
# ===================================================================


class TestNewsQAOracleDataset:
    """NewsQAOracleDataset 基本功能测试。"""

    def test_length(self, jsonl_path: Path) -> None:
        """加载 3 条数据。"""
        ds = NewsQAOracleDataset(jsonl_path)
        assert len(ds) == 3

    def test_getitem_keys(self, jsonl_path: Path) -> None:
        """__getitem__ 返回正确的 key 集合。"""
        ds = NewsQAOracleDataset(jsonl_path)
        item = ds[0]
        assert set(item.keys()) == {"prompt", "answer", "knowledge_text"}

    def test_getitem_prompt_format(self, jsonl_path: Path) -> None:
        """prompt 格式: 以 'Question:' 开头, 包含 A./B./C./D., 以 'Answer:' 结尾。"""
        ds = NewsQAOracleDataset(jsonl_path)
        prompt = ds[0]["prompt"]
        assert prompt.startswith("Question:")
        for letter in ("A.", "B.", "C.", "D."):
            assert letter in prompt
        assert prompt.endswith("Answer:")

    def test_getitem_answer_is_letter(self, jsonl_path: Path) -> None:
        """answer 应为 correct_letter 字段值: B, C, A。"""
        ds = NewsQAOracleDataset(jsonl_path)
        assert ds[0]["answer"] == "B"
        assert ds[1]["answer"] == "C"
        assert ds[2]["answer"] == "A"

    def test_getitem_knowledge_text(self, jsonl_path: Path) -> None:
        """knowledge_text 应包含 passage 中的关键内容。"""
        ds = NewsQAOracleDataset(jsonl_path)
        assert "Rayleigh scattering" in ds[0]["knowledge_text"]
        assert "arithmetic" in ds[1]["knowledge_text"]
        assert "Mercury" in ds[2]["knowledge_text"]

    def test_custom_knowledge_field(self, jsonl_with_compressed: Path) -> None:
        """使用自定义 knowledge_field='compressed_text'。"""
        ds = NewsQAOracleDataset(
            jsonl_with_compressed, knowledge_field="compressed_text"
        )
        assert ds[0]["knowledge_text"].startswith("COMPRESSED:")

    def test_empty_lines_skipped(self, jsonl_with_blanks: Path) -> None:
        """空行不应被计入有效数据。"""
        ds = NewsQAOracleDataset(jsonl_with_blanks)
        assert len(ds) == 3


# ===================================================================
# Task 2: TestCollateFn
# ===================================================================


class TestCollateFn:
    """make_collate_fn 批量 tokenize + 动态 padding 测试。"""

    @pytest.fixture()
    def batch(self, jsonl_path: Path) -> List[dict]:
        """从 dataset 取全部 3 条作为 batch。"""
        ds = NewsQAOracleDataset(jsonl_path)
        return [ds[i] for i in range(len(ds))]

    @pytest.fixture()
    def collated(self, batch: List[dict], tokenizer) -> dict:
        """执行 collate 得到 tensor dict。"""
        fn = make_collate_fn(tokenizer, max_seq_len=128, knowledge_max_len=64)
        return fn(batch)

    # ---- 测试 ----

    def test_output_keys(self, collated: dict) -> None:
        """输出 dict 包含 5 个预期 key。"""
        expected = {
            "input_ids",
            "attention_mask",
            "labels",
            "knowledge_input_ids",
            "knowledge_attention_mask",
        }
        assert set(collated.keys()) == expected

    def test_output_shapes(self, collated: dict) -> None:
        """batch 维度 = 3，各 tensor 形状一致。"""
        assert collated["input_ids"].shape[0] == 3
        seq_len = collated["input_ids"].shape[1]
        assert collated["attention_mask"].shape == (3, seq_len)
        assert collated["labels"].shape == (3, seq_len)

        k_len = collated["knowledge_input_ids"].shape[1]
        assert collated["knowledge_attention_mask"].shape == (3, k_len)

    def test_output_dtypes(self, collated: dict) -> None:
        """所有 tensor 均为 torch.long。"""
        for key, tensor in collated.items():
            assert tensor.dtype == torch.long, f"{key} dtype={tensor.dtype}"

    def test_labels_prompt_masked(self, collated: dict) -> None:
        """labels 第一个 token（属于 prompt）应为 -100；且存在非 -100 token。"""
        labels = collated["labels"]
        # 每条样本的第一个 token 属于 prompt → -100
        assert (labels[:, 0] == -100).all()
        # 至少存在一些 answer token（非 -100）
        assert (labels != -100).any()

    def test_labels_padding_masked(self, collated: dict) -> None:
        """attention_mask==0 处 labels 必须为 -100。"""
        mask = collated["attention_mask"]
        labels = collated["labels"]
        assert (labels[mask == 0] == -100).all()

    def test_dynamic_padding(self, jsonl_path: Path, tokenizer) -> None:
        """不同 batch size 应产生不同序列长度（动态 padding 验证）。"""
        ds = NewsQAOracleDataset(jsonl_path)
        fn = make_collate_fn(tokenizer, max_seq_len=256, knowledge_max_len=128)

        batch_1 = [ds[0]]  # 单条
        batch_3 = [ds[i] for i in range(3)]  # 三条

        out_1 = fn(batch_1)
        out_3 = fn(batch_3)

        # 单条 batch pad 到自身长度，三条 batch pad 到最长那条
        # 两者序列长度不一定相同
        len_1 = out_1["input_ids"].shape[1]
        len_3 = out_3["input_ids"].shape[1]
        # 三条 batch 的最长序列 >= 单条序列
        assert len_3 >= len_1

    def test_knowledge_truncation(self, jsonl_path: Path, tokenizer) -> None:
        """knowledge_max_len 应限制 knowledge token 长度。"""
        ds = NewsQAOracleDataset(jsonl_path)
        short_limit = 8
        fn = make_collate_fn(tokenizer, max_seq_len=256, knowledge_max_len=short_limit)
        batch = [ds[i] for i in range(len(ds))]
        out = fn(batch)
        assert out["knowledge_input_ids"].shape[1] <= short_limit


# ===================================================================
# Task 3: TestCounterfactualDataset
# ===================================================================


class TestCounterfactualDataset:
    """CounterfactualDataset 基本功能测试。"""

    def test_length_filters_train_only(self, cf_jsonl_path: Path) -> None:
        """仅加载 split='train' 的行，3 条中应保留 2 条。"""
        ds = CounterfactualDataset(cf_jsonl_path, split="train")
        assert len(ds) == 2

    def test_getitem_keys(self, cf_jsonl_path: Path) -> None:
        """返回与 NewsQAOracleDataset 相同的 key 集合。"""
        ds = CounterfactualDataset(cf_jsonl_path, split="train")
        item = ds[0]
        assert set(item.keys()) == {"prompt", "answer", "knowledge_text"}

    def test_answer_is_target_letter(self, cf_jsonl_path: Path) -> None:
        """answer 应为 target_letter（非 correct_letter）。"""
        ds = CounterfactualDataset(cf_jsonl_path, split="train")
        assert ds[0]["answer"] == "A"  # target_letter of cf_B
        assert ds[1]["answer"] == "C"  # target_letter of cf_C

    def test_knowledge_is_counterfactual(self, cf_jsonl_path: Path) -> None:
        """knowledge_text 应为 counterfactual_passage。"""
        ds = CounterfactualDataset(cf_jsonl_path, split="train")
        assert "iron oxide" in ds[0]["knowledge_text"]
        assert "chlorophyll" in ds[1]["knowledge_text"]

    def test_prompt_format(self, cf_jsonl_path: Path) -> None:
        """prompt 格式与 NewsQAOracleDataset 一致。"""
        ds = CounterfactualDataset(cf_jsonl_path, split="train")
        prompt = ds[0]["prompt"]
        assert prompt.startswith("Question:")
        for letter in ("A.", "B.", "C.", "D."):
            assert letter in prompt
        assert prompt.endswith("Answer:")

    def test_collate_compatible(self, cf_jsonl_path: Path, tokenizer) -> None:
        """CounterfactualDataset 输出可被现有 collate_fn 处理。"""
        ds = CounterfactualDataset(cf_jsonl_path, split="train")
        fn = make_collate_fn(tokenizer, max_seq_len=128, knowledge_max_len=64)
        batch = [ds[i] for i in range(len(ds))]
        out = fn(batch)
        assert out["input_ids"].shape[0] == 2
        assert out["labels"].shape[0] == 2


# ===================================================================
# Task 4: TestOversampledDataset
# ===================================================================


class TestOversampledDataset:
    """OversampledDataset 过采样包装器测试。"""

    def test_length_factor_2(self, cf_jsonl_path: Path) -> None:
        """2× 过采样后长度翻倍。"""
        ds = CounterfactualDataset(cf_jsonl_path, split="train")
        oversampled = OversampledDataset(ds, factor=2)
        assert len(oversampled) == len(ds) * 2  # 2 * 2 = 4

    def test_length_factor_1(self, cf_jsonl_path: Path) -> None:
        """1× 不改变长度。"""
        ds = CounterfactualDataset(cf_jsonl_path, split="train")
        oversampled = OversampledDataset(ds, factor=1)
        assert len(oversampled) == len(ds)

    def test_getitem_wraps_around(self, cf_jsonl_path: Path) -> None:
        """索引超过原始长度时循环取模。"""
        ds = CounterfactualDataset(cf_jsonl_path, split="train")
        oversampled = OversampledDataset(ds, factor=2)
        assert oversampled[0] == oversampled[2]  # idx 0 == idx 0+len(ds)
        assert oversampled[1] == oversampled[3]  # idx 1 == idx 1+len(ds)

    def test_getitem_returns_same_format(self, cf_jsonl_path: Path) -> None:
        """返回格式与底层 dataset 一致。"""
        ds = CounterfactualDataset(cf_jsonl_path, split="train")
        oversampled = OversampledDataset(ds, factor=2)
        item = oversampled[0]
        assert set(item.keys()) == {"prompt", "answer", "knowledge_text"}


# ---------------------------------------------------------------------------
# CoT 模式测试
# ---------------------------------------------------------------------------

MOCK_COT_DATA: List[dict] = [
    {
        "question": "What color is the sky?",
        "passage": "The sky appears blue due to Rayleigh scattering of sunlight.",
        "correct_answer": "Blue",
        "correct_letter": "B",
        "options": {"A": "Red", "B": "Blue", "C": "Green", "D": "Yellow"},
        "cot_response": "The passage states the sky is blue due to Rayleigh scattering. The answer is B",
        "cot_extracted_letter": "B",
        "cot_valid": True,
    },
    {
        "question": "What is 2+2?",
        "passage": "Basic arithmetic: two plus two equals four.",
        "correct_answer": "4",
        "correct_letter": "C",
        "options": {"A": "3", "B": "5", "C": "4", "D": "6"},
        "cot_response": "Two plus two is four. The answer is C",
        "cot_extracted_letter": "C",
        "cot_valid": True,
    },
    {
        "question": "Which planet?",
        "passage": "Mercury orbits closest.",
        "correct_answer": "Mercury",
        "correct_letter": "A",
        "options": {"A": "Mercury", "B": "Venus", "C": "Earth", "D": "Mars"},
        "cot_response": "I think Venus. The answer is B",
        "cot_extracted_letter": "B",
        "cot_valid": False,
    },
]

MOCK_CF_COT_DATA: List[dict] = [
    {
        "cf_id": "arc_00_cf_A",
        "source_id": "arc_00",
        "dataset": "arc_easy",
        "question": "What color is the sky?",
        "options": {"A": "Red", "B": "Blue", "C": "Green", "D": "Yellow"},
        "correct_letter": "B",
        "passage": "The sky appears blue.",
        "target_letter": "A",
        "target_answer": "Red",
        "counterfactual_passage": "The sky appears red due to iron oxide.",
        "split": "train",
        "cot_response": "According to the passage, the sky is red. The answer is A",
        "cot_extracted_letter": "A",
        "cot_valid": True,
    },
]


class TestNewsQACoTMode:
    """NewsQAOracleDataset CoT 模式测试。"""

    def test_cot_prompt_format(self, tmp_path: Path):
        jsonl = tmp_path / "cot.jsonl"
        jsonl.write_text(
            "\n".join(json.dumps(r) for r in MOCK_COT_DATA), encoding="utf-8"
        )
        ds = NewsQAOracleDataset(jsonl, prompt_mode="cot")
        item = ds[0]
        assert "Let's think step by step" in item["prompt"]
        assert 'The answer is X' in item["prompt"]
        assert "Answer:" not in item["prompt"]

    def test_cot_answer_is_full_response(self, tmp_path: Path):
        jsonl = tmp_path / "cot.jsonl"
        jsonl.write_text(
            "\n".join(json.dumps(r) for r in MOCK_COT_DATA), encoding="utf-8"
        )
        ds = NewsQAOracleDataset(jsonl, prompt_mode="cot")
        item = ds[0]
        assert item["answer"] == MOCK_COT_DATA[0]["cot_response"]
        assert len(item["answer"]) > 10

    def test_cot_filters_invalid(self, tmp_path: Path):
        jsonl = tmp_path / "cot.jsonl"
        jsonl.write_text(
            "\n".join(json.dumps(r) for r in MOCK_COT_DATA), encoding="utf-8"
        )
        ds = NewsQAOracleDataset(jsonl, prompt_mode="cot")
        assert len(ds) == 2

    def test_direct_mode_unchanged(self, tmp_path: Path):
        jsonl = tmp_path / "cot.jsonl"
        jsonl.write_text(
            "\n".join(json.dumps(r) for r in MOCK_COT_DATA), encoding="utf-8"
        )
        ds = NewsQAOracleDataset(jsonl, prompt_mode="direct")
        item = ds[0]
        assert "Answer:" in item["prompt"]
        assert item["answer"] == "B"
        assert len(ds) == 3


class TestCFCoTMode:
    """CounterfactualDataset CoT 模式测试。"""

    def test_cot_prompt_format(self, tmp_path: Path):
        jsonl = tmp_path / "cf_cot.jsonl"
        jsonl.write_text(json.dumps(MOCK_CF_COT_DATA[0]), encoding="utf-8")
        ds = CounterfactualDataset(jsonl, split="train", prompt_mode="cot")
        item = ds[0]
        assert "Let's think step by step" in item["prompt"]
        assert "Answer:" not in item["prompt"]

    def test_cot_answer_is_full_response(self, tmp_path: Path):
        jsonl = tmp_path / "cf_cot.jsonl"
        jsonl.write_text(json.dumps(MOCK_CF_COT_DATA[0]), encoding="utf-8")
        ds = CounterfactualDataset(jsonl, split="train", prompt_mode="cot")
        item = ds[0]
        assert item["answer"] == MOCK_CF_COT_DATA[0]["cot_response"]

    def test_knowledge_text_is_cf_passage(self, tmp_path: Path):
        jsonl = tmp_path / "cf_cot.jsonl"
        jsonl.write_text(json.dumps(MOCK_CF_COT_DATA[0]), encoding="utf-8")
        ds = CounterfactualDataset(jsonl, split="train", prompt_mode="cot")
        item = ds[0]
        assert item["knowledge_text"] == MOCK_CF_COT_DATA[0]["counterfactual_passage"]


class TestCollateWithCoT:
    """collate_fn 处理多 token answer 的测试。"""

    def test_labels_mask_prompt_only(self, tmp_path: Path, tokenizer):
        jsonl = tmp_path / "cot.jsonl"
        jsonl.write_text(
            "\n".join(json.dumps(r) for r in MOCK_COT_DATA[:2]), encoding="utf-8"
        )
        ds = NewsQAOracleDataset(jsonl, prompt_mode="cot")
        collate = make_collate_fn(tokenizer, max_seq_len=256, knowledge_max_len=64)
        batch = collate([ds[0], ds[1]])
        labels = batch["labels"]
        non_masked = (labels != -100).sum().item()
        assert non_masked > 5
