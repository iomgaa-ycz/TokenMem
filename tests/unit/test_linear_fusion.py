"""LinearFusion 门控模块单元测试。"""
import pytest
import torch

from memory_lora.linear_fusion import LinearFusion


class TestLinearFusionInit:
    def test_parameter_shapes(self):
        m = LinearFusion(hidden_dim=64, rank=8, alpha=32)
        assert m.W_A.shape == (64, 8)
        assert m.W_B.shape == (8, 64)

    def test_w_b_zero_init(self):
        m = LinearFusion(hidden_dim=64, rank=8)
        assert torch.all(m.W_B == 0)

    def test_w_a_gaussian_init(self):
        m = LinearFusion(hidden_dim=256, rank=16)
        assert m.W_A.std() < 0.05


class TestLinearFusionForward:
    def test_zero_init_is_identity(self):
        m = LinearFusion(hidden_dim=64, rank=8)
        m.eval()
        A = torch.randn(2, 10, 64)
        B = torch.randn(2, 10, 64)
        out = m(A, B)
        torch.testing.assert_close(out, A)

    def test_nonzero_weights_change_output(self):
        m = LinearFusion(hidden_dim=64, rank=8)
        m.eval()
        m.W_B.data.fill_(0.1)
        A = torch.randn(2, 10, 64)
        B = torch.randn(2, 10, 64)
        out = m(A, B)
        assert not torch.allclose(out, A)

    def test_output_shape_matches_input(self):
        m = LinearFusion(hidden_dim=128, rank=16)
        A = torch.randn(4, 20, 128)
        B = torch.randn(4, 20, 128)
        assert m(A, B).shape == (4, 20, 128)

    def test_dtype_preservation(self):
        m = LinearFusion(hidden_dim=64, rank=8)
        A = torch.randn(2, 10, 64, dtype=torch.bfloat16)
        B = torch.randn(2, 10, 64, dtype=torch.bfloat16)
        assert m(A, B).dtype == torch.bfloat16

    def test_alpha_scaling(self):
        m = LinearFusion(hidden_dim=64, rank=8, alpha=0)
        m.eval()
        m.W_B.data.fill_(1.0)
        A = torch.randn(2, 10, 64)
        B = torch.randn(2, 10, 64)
        torch.testing.assert_close(m(A, B), A)
