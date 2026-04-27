"""TokenMemoryBank 单元测试。

使用 Qwen/Qwen3-0.6B tokenizer（轻量，无需模型权重）。
测试配置：capacity=100, fusion_length=32, emb_dim=64。
"""

from __future__ import annotations

import os
import tempfile
from typing import List, Tuple

import pytest
import torch
from torch import Tensor
from transformers import AutoTokenizer

from memory_lora.token_bank import TokenMemoryBank


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────


@pytest.fixture
def tokenizer():
    """加载 Qwen3-0.6B tokenizer，设置 pad_token。"""
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture
def bank(tokenizer):
    """创建小规模测试用 TokenMemoryBank。"""
    return TokenMemoryBank(
        tokenizer=tokenizer, capacity=100, fusion_length=32, emb_dim=64
    )


def _random_emb(dim: int = 64) -> Tensor:
    """生成随机嵌入向量 [emb_dim]。"""
    return torch.randn(dim)


def _make_entries(n: int, dim: int = 64) -> List[Tuple[str, Tensor]]:
    """批量生成 (text, embedding) 测试数据。"""
    texts = [f"This is test entry number {i}" for i in range(n)]
    embs = [_random_emb(dim) for _ in range(n)]
    return list(zip(texts, embs))


# ─────────────────────────────────────────────
# TestConstruction
# ─────────────────────────────────────────────


class TestConstruction:
    """测试 TokenMemoryBank 构造和初始状态。"""

    def test_empty_bank_len(self, bank: TokenMemoryBank):
        """空 bank 长度为 0。"""
        assert len(bank) == 0

    def test_capacity_stored(self, bank: TokenMemoryBank):
        """capacity 正确存储。"""
        assert bank.capacity == 100

    def test_fusion_length_stored(self, bank: TokenMemoryBank):
        """fusion_length 正确存储。"""
        assert bank.fusion_length == 32

    def test_emb_dim_stored(self, bank: TokenMemoryBank):
        """emb_dim 正确存储。"""
        assert bank.emb_dim == 64

    def test_tokens_shape(self, bank: TokenMemoryBank):
        """内部 _tokens 张量形状正确。"""
        assert bank._tokens.shape == (100, 32)
        assert bank._tokens.dtype == torch.long

    def test_embs_shape(self, bank: TokenMemoryBank):
        """内部 _embs 张量形状正确。"""
        assert bank._embs.shape == (100, 64)
        assert bank._embs.dtype == torch.float32

    def test_deleted_shape(self, bank: TokenMemoryBank):
        """内部 _deleted 张量形状正确。"""
        assert bank._deleted.shape == (100,)
        assert bank._deleted.dtype == torch.bool
        assert not bank._deleted.any()

    def test_faiss_index_created(self, bank: TokenMemoryBank):
        """FAISS 索引已创建。"""
        assert bank._index is not None


# ─────────────────────────────────────────────
# TestValidate
# ─────────────────────────────────────────────


class TestValidate:
    """测试 _validate 方法的输入校验。"""

    def test_valid_input_passes(self, bank: TokenMemoryBank):
        """合法输入不抛异常。"""
        token_ids = torch.zeros(32, dtype=torch.long)
        emb = torch.randn(64)
        bank._validate(token_ids, emb)  # 不应抛异常

    def test_wrong_emb_dim_raises(self, bank: TokenMemoryBank):
        """embedding 维度不匹配时抛 ValueError。"""
        token_ids = torch.zeros(32, dtype=torch.long)
        emb = torch.randn(128)  # 期望 64
        with pytest.raises(ValueError, match="emb"):
            bank._validate(token_ids, emb)

    def test_wrong_token_length_raises(self, bank: TokenMemoryBank):
        """token 长度不匹配时抛 ValueError。"""
        token_ids = torch.zeros(16, dtype=torch.long)  # 期望 32
        emb = torch.randn(64)
        with pytest.raises(ValueError, match="token"):
            bank._validate(token_ids, emb)

    def test_wrong_dtype_raises(self, bank: TokenMemoryBank):
        """token_ids dtype 不是 long 时抛 TypeError。"""
        token_ids = torch.zeros(32, dtype=torch.float32)
        emb = torch.randn(64)
        with pytest.raises(TypeError, match="long"):
            bank._validate(token_ids, emb)


# ─────────────────────────────────────────────
# TestTokenizeText
# ─────────────────────────────────────────────


class TestTokenizeText:
    """测试 _tokenize_text 方法。"""

    def test_output_shape(self, bank: TokenMemoryBank):
        """输出形状为 [fusion_length]。"""
        result = bank._tokenize_text("Hello world")
        assert result.shape == (32,)
        assert result.dtype == torch.long

    def test_truncation(self, bank: TokenMemoryBank):
        """长文本被截断到 fusion_length。"""
        long_text = "word " * 1000
        result = bank._tokenize_text(long_text)
        assert result.shape == (32,)

    def test_padding(self, bank: TokenMemoryBank):
        """短文本被填充到 fusion_length。"""
        result = bank._tokenize_text("Hi")
        assert result.shape == (32,)


# ─────────────────────────────────────────────
# TestAdd
# ─────────────────────────────────────────────


class TestAdd:
    """测试 add 方法（批量写入）。"""

    def test_add_single_entry(self, bank: TokenMemoryBank):
        """添加单条 entry 后长度为 1。"""
        entries = _make_entries(1)
        ids = bank.add(entries)
        assert len(ids) == 1
        assert ids[0] == 0
        assert len(bank) == 1

    def test_add_multiple_entries(self, bank: TokenMemoryBank):
        """添加多条 entry 后长度正确。"""
        entries = _make_entries(5)
        ids = bank.add(entries)
        assert len(ids) == 5
        assert ids == [0, 1, 2, 3, 4]
        assert len(bank) == 5

    def test_correct_tokens_stored(self, bank: TokenMemoryBank):
        """存储的 token_ids 与 tokenize 结果一致。"""
        text = "Hello, world!"
        emb = _random_emb()
        bank.add([(text, emb)])
        expected_tokens = bank._tokenize_text(text)
        stored_tokens = bank._tokens[0]
        assert torch.equal(stored_tokens, expected_tokens)

    def test_correct_emb_stored(self, bank: TokenMemoryBank):
        """存储的 embedding 与输入一致。"""
        emb = _random_emb()
        bank.add([("test", emb)])
        assert torch.allclose(bank._embs[0], emb)

    def test_faiss_updated(self, bank: TokenMemoryBank):
        """添加后 FAISS 索引中有对应条目。"""
        entries = _make_entries(3)
        bank.add(entries)
        assert bank._index.ntotal == 3

    def test_full_bank_raises(self, tokenizer):
        """容量满时抛 RuntimeError。"""
        small_bank = TokenMemoryBank(
            tokenizer=tokenizer, capacity=3, fusion_length=32, emb_dim=64
        )
        entries = _make_entries(3)
        small_bank.add(entries)
        with pytest.raises(RuntimeError, match="full|capacity"):
            small_bank.add(_make_entries(1))

    def test_wrong_emb_dim_raises(self, bank: TokenMemoryBank):
        """embedding 维度不匹配时抛 ValueError。"""
        bad_emb = torch.randn(128)
        with pytest.raises(ValueError, match="emb"):
            bank.add([("test", bad_emb)])


# ─────────────────────────────────────────────
# TestGetItem
# ─────────────────────────────────────────────


class TestGetItem:
    """测试 __getitem__ 方法。"""

    def test_returns_clone(self, bank: TokenMemoryBank):
        """返回的是克隆，不与内部存储共享内存。"""
        bank.add(_make_entries(1))
        tokens, emb = bank[0]
        tokens[0] = -1
        emb[0] = -999.0
        # 原始数据不应被修改
        assert bank._tokens[0, 0] != -1
        assert bank._embs[0, 0] != -999.0

    def test_out_of_range_raises(self, bank: TokenMemoryBank):
        """越界访问抛 IndexError。"""
        bank.add(_make_entries(2))
        with pytest.raises(IndexError):
            _ = bank[5]

    def test_deleted_raises(self, tokenizer):
        """访问已删除 entry 抛 ValueError。"""
        # 使用高 compact_threshold 避免自动 compact
        b = TokenMemoryBank(
            tokenizer=tokenizer,
            capacity=100,
            fusion_length=32,
            emb_dim=64,
            compact_threshold=1.0,
        )
        b.add(_make_entries(2))
        b.delete(0)
        with pytest.raises(ValueError, match="deleted"):
            _ = b[0]


# ─────────────────────────────────────────────
# TestGetTokenIds
# ─────────────────────────────────────────────


class TestGetTokenIds:
    """测试 get_token_ids 批量获取方法。"""

    def test_1d_input(self, bank: TokenMemoryBank):
        """1D 输入 [B] 返回 [B, fusion_length]。"""
        bank.add(_make_entries(5))
        ids = torch.tensor([0, 2, 4], dtype=torch.long)
        result = bank.get_token_ids(ids)
        assert result.shape == (3, 32)

    def test_2d_input(self, bank: TokenMemoryBank):
        """2D 输入 [B, k] 返回 [B, k, fusion_length]。"""
        bank.add(_make_entries(5))
        ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        result = bank.get_token_ids(ids)
        assert result.shape == (2, 2, 32)


# ─────────────────────────────────────────────
# TestRetrieve
# ─────────────────────────────────────────────


class TestRetrieve:
    """测试 retrieve FAISS 检索方法。"""

    def test_top1_finds_correct(self, bank: TokenMemoryBank):
        """top-1 检索应返回最相似的 entry。"""
        emb0 = torch.zeros(64)
        emb0[0] = 1.0  # 方向 [1,0,0,...]
        emb1 = torch.zeros(64)
        emb1[1] = 1.0  # 方向 [0,1,0,...]

        bank.add([("entry zero", emb0), ("entry one", emb1)])

        # 查询方向接近 emb0
        query = torch.zeros(1, 64)
        query[0, 0] = 1.0
        ids, scores = bank.retrieve(query, k=1)
        assert ids.shape == (1, 1)
        assert ids[0, 0].item() == 0

    def test_topk(self, bank: TokenMemoryBank):
        """top-k 检索返回正确数量的结果。"""
        bank.add(_make_entries(10))
        query = _random_emb().unsqueeze(0)  # [1, 64]
        ids, scores = bank.retrieve(query, k=3)
        assert ids.shape == (1, 3)
        assert scores.shape == (1, 3)

    def test_batch_queries(self, bank: TokenMemoryBank):
        """批量查询返回正确形状。"""
        bank.add(_make_entries(10))
        queries = torch.randn(4, 64)  # [4, 64]
        ids, scores = bank.retrieve(queries, k=2)
        assert ids.shape == (4, 2)
        assert scores.shape == (4, 2)

    def test_empty_bank_raises(self, bank: TokenMemoryBank):
        """空 bank 检索抛 RuntimeError。"""
        query = torch.randn(1, 64)
        with pytest.raises(RuntimeError, match="empty"):
            bank.retrieve(query, k=1)


# ─────────────────────────────────────────────
# TestEdit
# ─────────────────────────────────────────────


class TestEdit:
    """测试 edit 方法（更新 entry）。"""

    def test_updates_tokens_and_emb(self, bank: TokenMemoryBank):
        """edit 后 token_ids 和 embedding 被更新。"""
        bank.add(_make_entries(2))
        new_text = "completely new text content"
        new_emb = torch.ones(64) * 42.0
        bank.edit(0, new_text, new_emb)

        tokens, emb = bank[0]
        expected_tokens = bank._tokenize_text(new_text)
        assert torch.equal(tokens, expected_tokens)
        assert torch.allclose(emb, new_emb)

    def test_faiss_updated(self, bank: TokenMemoryBank):
        """edit 后 FAISS 索引被更新。"""
        # 创建两个正交方向的 entry
        emb0 = torch.zeros(64)
        emb0[0] = 1.0
        emb1 = torch.zeros(64)
        emb1[1] = 1.0
        bank.add([("a", emb0), ("b", emb1)])

        # 把 entry 0 改为方向 [0,1,0,...] (与 entry 1 同向)
        new_emb = torch.zeros(64)
        new_emb[1] = 1.0
        bank.edit(0, "new a", new_emb)

        # 查询方向 [0,1,0,...] 应返回 entry 0 或 1
        query = torch.zeros(1, 64)
        query[0, 1] = 1.0
        ids, _ = bank.retrieve(query, k=2)
        assert set(ids[0].tolist()) == {0, 1}

    def test_invalid_id_raises(self, bank: TokenMemoryBank):
        """无效 entry_id 抛 IndexError。"""
        bank.add(_make_entries(2))
        with pytest.raises(IndexError):
            bank.edit(10, "text", _random_emb())

    def test_deleted_id_raises(self, tokenizer):
        """已删除 entry 的 edit 抛 ValueError。"""
        # 使用高 compact_threshold 避免自动 compact
        b = TokenMemoryBank(
            tokenizer=tokenizer,
            capacity=100,
            fusion_length=32,
            emb_dim=64,
            compact_threshold=1.0,
        )
        b.add(_make_entries(2))
        b.delete(0)
        with pytest.raises(ValueError, match="deleted"):
            b.edit(0, "text", _random_emb())

    def test_preserves_len(self, bank: TokenMemoryBank):
        """edit 不改变 bank 长度。"""
        bank.add(_make_entries(3))
        bank.edit(1, "new text", _random_emb())
        assert len(bank) == 3


# ─────────────────────────────────────────────
# TestAudit
# ─────────────────────────────────────────────


class TestAudit:
    """测试 audit 方法（解码 token_ids 为文本）。"""

    def test_returns_readable_text(self, bank: TokenMemoryBank):
        """audit 返回可读文本。"""
        text = "The quick brown fox"
        bank.add([(text, _random_emb())])
        result = bank.audit(0)
        assert isinstance(result, str)
        assert len(result) > 0
        # 由于 truncation/padding，文本可能不完全相同，但应包含原文关键词
        assert "quick" in result or "brown" in result or "fox" in result

    def test_invalid_id_raises(self, bank: TokenMemoryBank):
        """无效 entry_id 抛 IndexError。"""
        with pytest.raises(IndexError):
            bank.audit(0)

    def test_deleted_id_raises(self, tokenizer):
        """已删除 entry 抛 ValueError。"""
        # 使用高 compact_threshold 避免自动 compact
        b = TokenMemoryBank(
            tokenizer=tokenizer,
            capacity=100,
            fusion_length=32,
            emb_dim=64,
            compact_threshold=1.0,
        )
        b.add(_make_entries(1))
        b.delete(0)
        with pytest.raises(ValueError, match="deleted"):
            b.audit(0)


# ─────────────────────────────────────────────
# TestMigrateTo
# ─────────────────────────────────────────────


class TestMigrateTo:
    """测试 migrate_to 方法（导出所有活跃 entry 文本）。"""

    def test_returns_active_texts_only(self, bank: TokenMemoryBank):
        """只返回未删除的 entry 文本。"""
        bank.add(_make_entries(5))
        bank.delete(1)
        bank.delete(3)
        texts = bank.migrate_to()
        assert len(texts) == 3

    def test_empty_bank(self, bank: TokenMemoryBank):
        """空 bank 返回空列表。"""
        texts = bank.migrate_to()
        assert texts == []


# ─────────────────────────────────────────────
# TestDelete
# ─────────────────────────────────────────────


class TestDelete:
    """测试 delete 方法（软删除）。"""

    def test_reduces_len(self, bank: TokenMemoryBank):
        """删除后活跃数量减少。"""
        bank.add(_make_entries(5))
        bank.delete(2)
        assert len(bank) == 4

    def test_marks_deleted(self, tokenizer):
        """删除后 _deleted 标记为 True。"""
        # 使用高 compact_threshold 避免自动 compact
        b = TokenMemoryBank(
            tokenizer=tokenizer,
            capacity=100,
            fusion_length=32,
            emb_dim=64,
            compact_threshold=1.0,
        )
        b.add(_make_entries(3))
        b.delete(1)
        assert b._deleted[1].item() is True

    def test_removes_from_faiss(self, bank: TokenMemoryBank):
        """删除后 FAISS 中对应 entry 被移除。"""
        bank.add(_make_entries(3))
        bank.delete(1)
        assert bank._index.ntotal == 2

    def test_invalid_id_raises(self, bank: TokenMemoryBank):
        """无效 entry_id 抛 IndexError。"""
        bank.add(_make_entries(2))
        with pytest.raises(IndexError):
            bank.delete(10)

    def test_already_deleted_raises(self, tokenizer):
        """重复删除抛 ValueError。"""
        # 使用高 compact_threshold 避免自动 compact
        b = TokenMemoryBank(
            tokenizer=tokenizer,
            capacity=100,
            fusion_length=32,
            emb_dim=64,
            compact_threshold=1.0,
        )
        b.add(_make_entries(2))
        b.delete(0)
        with pytest.raises(ValueError, match="already deleted"):
            b.delete(0)


# ─────────────────────────────────────────────
# TestCompact
# ─────────────────────────────────────────────


class TestCompact:
    """测试 _compact 和 _maybe_compact 方法。"""

    def test_reclaims_space(self, bank: TokenMemoryBank):
        """compact 后 _n 反映实际活跃数量。"""
        bank.add(_make_entries(10))
        bank.delete(2)
        bank.delete(5)
        bank.delete(7)
        bank._compact()
        assert bank._n == 7
        assert bank._n_deleted == 0
        assert not bank._deleted[: bank._n].any()

    def test_auto_triggers_at_threshold(self, tokenizer):
        """删除比例超过 compact_threshold 时自动触发 compact。"""
        # compact_threshold=0.3，10 条删 4 条 = 40% > 30%
        b = TokenMemoryBank(
            tokenizer=tokenizer,
            capacity=100,
            fusion_length=32,
            emb_dim=64,
            compact_threshold=0.3,
        )
        b.add(_make_entries(10))
        # 逐条删除，第 4 条时 4/10 = 0.4 > 0.3 触发 compact
        b.delete(0)  # 1/10 = 0.1
        b.delete(1)  # 2/10 = 0.2
        b.delete(2)  # 3/10 = 0.3 — 不触发（>=，非 >）
        assert b._n == 10  # compact 未触发
        b.delete(3)  # 4/10 = 0.4 > 0.3 — 触发
        # compact 后 _n 应该被重置
        assert b._n == 6
        assert b._n_deleted == 0

    def test_add_triggers_compact_when_needed(self, tokenizer):
        """容量满但有已删除 entry 时，add 触发 compact 后成功写入。"""
        b = TokenMemoryBank(
            tokenizer=tokenizer, capacity=5, fusion_length=32, emb_dim=64
        )
        b.add(_make_entries(5))
        b.delete(0)
        b.delete(1)
        # 容量 5，_n=5，但有 2 个删除，compact 后 _n=3，再 add 1 条
        ids = b.add(_make_entries(1))
        assert len(ids) == 1
        assert len(b) == 4  # 3 alive + 1 new


# ─────────────────────────────────────────────
# TestPersistence
# ─────────────────────────────────────────────


class TestPersistence:
    """测试 save/load 持久化。"""

    def test_save_load_roundtrip(self, bank: TokenMemoryBank, tokenizer):
        """save + load 后数据一致。"""
        entries = _make_entries(5)
        bank.add(entries)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            bank.save(path)
            new_bank = TokenMemoryBank(
                tokenizer=tokenizer, capacity=100, fusion_length=32, emb_dim=64
            )
            new_bank.load(path)
            assert len(new_bank) == 5
            for i in range(5):
                t1, e1 = bank[i]
                t2, e2 = new_bank[i]
                assert torch.equal(t1, t2)
                assert torch.allclose(e1, e2)
        finally:
            os.unlink(path)

    def test_save_load_with_deletions(self, tokenizer):
        """含删除的 bank save/load 后删除标记保留。"""
        # 使用高 compact_threshold 避免自动 compact
        bank = TokenMemoryBank(
            tokenizer=tokenizer,
            capacity=100,
            fusion_length=32,
            emb_dim=64,
            compact_threshold=1.0,
        )
        bank.add(_make_entries(5))
        bank.delete(1)
        bank.delete(3)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            bank.save(path)
            new_bank = TokenMemoryBank(
                tokenizer=tokenizer,
                capacity=100,
                fusion_length=32,
                emb_dim=64,
                compact_threshold=1.0,
            )
            new_bank.load(path)
            assert len(new_bank) == 3  # 5 - 2 deleted
            assert new_bank._deleted[1].item() is True
            assert new_bank._deleted[3].item() is True
        finally:
            os.unlink(path)

    def test_rebuilds_faiss(self, bank: TokenMemoryBank, tokenizer):
        """load 后 FAISS 索引被重建。"""
        bank.add(_make_entries(5))
        bank.delete(2)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            bank.save(path)
            new_bank = TokenMemoryBank(
                tokenizer=tokenizer, capacity=100, fusion_length=32, emb_dim=64
            )
            new_bank.load(path)
            assert new_bank._index.ntotal == 4  # 5 - 1 deleted

            # 检索应该正常工作
            query = torch.randn(1, 64)
            ids, scores = new_bank.retrieve(query, k=2)
            assert ids.shape == (1, 2)
        finally:
            os.unlink(path)

    def test_mismatched_fusion_length_raises(self, tokenizer):
        """fusion_length 不匹配时 load 抛 ValueError。"""
        bank1 = TokenMemoryBank(
            tokenizer=tokenizer, capacity=100, fusion_length=32, emb_dim=64
        )
        bank1.add(_make_entries(3))

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            bank1.save(path)
            bank2 = TokenMemoryBank(
                tokenizer=tokenizer, capacity=100, fusion_length=64, emb_dim=64
            )
            with pytest.raises(ValueError, match="fusion_length"):
                bank2.load(path)
        finally:
            os.unlink(path)

    def test_mismatched_emb_dim_raises(self, tokenizer):
        """emb_dim 不匹配时 load 抛 ValueError。"""
        bank1 = TokenMemoryBank(
            tokenizer=tokenizer, capacity=100, fusion_length=32, emb_dim=64
        )
        bank1.add(_make_entries(3))

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            bank1.save(path)
            bank2 = TokenMemoryBank(
                tokenizer=tokenizer, capacity=100, fusion_length=32, emb_dim=128
            )
            with pytest.raises(ValueError, match="emb_dim"):
                bank2.load(path)
        finally:
            os.unlink(path)

    def test_different_capacity_rebuilds_buffer(self, tokenizer):
        """capacity 不同时 load 重建内部 buffer。"""
        bank1 = TokenMemoryBank(
            tokenizer=tokenizer, capacity=50, fusion_length=32, emb_dim=64
        )
        bank1.add(_make_entries(3))

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            bank1.save(path)
            bank2 = TokenMemoryBank(
                tokenizer=tokenizer, capacity=200, fusion_length=32, emb_dim=64
            )
            bank2.load(path)
            # capacity 应被更新为保存时的值
            assert bank2.capacity == 50
            assert bank2._tokens.shape[0] == 50
            assert len(bank2) == 3
        finally:
            os.unlink(path)
