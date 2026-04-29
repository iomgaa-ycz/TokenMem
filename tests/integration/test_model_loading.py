"""6模型加载集成测试。

验证 _MODEL_CLASS_MAP / _CAUSAL_LM_CLASS_MAP 覆盖 3 个模型家族：
- qwen3      → Qwen3ForCausalLM
- gemma3_text → Gemma3ForCausalLM
- ministral   → MinistralForCausalLM

测试分两层：
1. 映射 & config 解析（CPU，无需完整权重）
2. 完整模型加载 + gate 验证（GPU，需本地权重）
"""
import importlib
import os

import pytest
import torch
from transformers import AutoConfig

from memory_lora.tokenmem_model import (
    _CAUSAL_LM_CLASS_MAP,
    _MODEL_CLASS_MAP,
    TokenMemForCausalLM,
)

# ── 模型注册表（全部走本地路径） ────────────────────────────────────
HF_MODEL_ROOT = "/home/iomgaa/Projects/Memory-LoRA/hugglingface_model"

MODEL_REGISTRY = {
    "qwen3_0.6b": {
        "path": os.path.join(HF_MODEL_ROOT, "qwen3-0.6B"),
        "model_type": "qwen3",
        "expected_hidden": 1024,
        "expected_layers": 28,
    },
    "qwen3_1.7b": {
        "path": os.path.join(HF_MODEL_ROOT, "qwen3-1.7B"),
        "model_type": "qwen3",
        "expected_hidden": 2048,
        "expected_layers": 28,
    },
    "qwen3_4b": {
        "path": os.path.join(HF_MODEL_ROOT, "qwen3-4B"),
        "model_type": "qwen3",
        "expected_hidden": 2560,
        "expected_layers": 36,
    },
    "qwen3_8b": {
        "path": os.path.join(HF_MODEL_ROOT, "qwen3-8B"),
        "model_type": "qwen3",
        "expected_hidden": 4096,
        "expected_layers": 36,
    },
    "gemma3_1b": {
        "path": os.path.join(HF_MODEL_ROOT, "gemma3-1b"),
        "model_type": "gemma3_text",
        "expected_hidden": 1152,
        "expected_layers": 26,
    },
    "ministral_3b": {
        "path": os.path.join(HF_MODEL_ROOT, "ministral-3-3b"),
        "model_type": "ministral",
        "expected_hidden": 2560,
        "expected_layers": 24,
    },
}


# ═══════════════════════════════════════════════════════════════════
#  Part 1: 映射 & Config（CPU，无权重依赖）
# ═══════════════════════════════════════════════════════════════════


class TestModelMappings:
    """验证 _MODEL_CLASS_MAP / _CAUSAL_LM_CLASS_MAP 的完整性和正确性。"""

    EXPECTED_TYPES = {"qwen3", "gemma3_text", "ministral"}

    def test_model_class_map_covers_all_families(self):
        """_MODEL_CLASS_MAP 应包含 3 个模型家族。"""
        assert set(_MODEL_CLASS_MAP.keys()) == self.EXPECTED_TYPES

    def test_causal_lm_class_map_covers_all_families(self):
        """_CAUSAL_LM_CLASS_MAP 应与 _MODEL_CLASS_MAP 键一致。"""
        assert set(_CAUSAL_LM_CLASS_MAP.keys()) == self.EXPECTED_TYPES

    @pytest.mark.parametrize("model_type", EXPECTED_TYPES)
    def test_module_importable(self, model_type: str):
        """每个 model_type 对应的 module 应可成功 import。"""
        module_path = _MODEL_CLASS_MAP[model_type]
        module = importlib.import_module(module_path)
        cls_name = _CAUSAL_LM_CLASS_MAP[model_type]
        cls = getattr(module, cls_name)
        assert cls is not None, f"{cls_name} 应存在于 {module_path}"

    @pytest.mark.parametrize("model_type", EXPECTED_TYPES)
    def test_class_has_from_pretrained(self, model_type: str):
        """加载的类应具有 from_pretrained 方法（继承自 PreTrainedModel）。"""
        module = importlib.import_module(_MODEL_CLASS_MAP[model_type])
        cls = getattr(module, _CAUSAL_LM_CLASS_MAP[model_type])
        assert hasattr(cls, "from_pretrained")


def _has_config(path: str) -> bool:
    """检查本地目录是否有 config.json。"""
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json"))


_CONFIG_AVAILABLE = [
    (name, info)
    for name, info in MODEL_REGISTRY.items()
    if _has_config(info["path"])
]


class TestConfigResolution:
    """验证 AutoConfig 能正确识别各模型的 model_type。"""

    @pytest.mark.parametrize("name,info", _CONFIG_AVAILABLE, ids=[n for n, _ in _CONFIG_AVAILABLE])
    def test_config_model_type(self, name: str, info: dict):
        """AutoConfig.from_pretrained 应返回正确的 model_type。"""
        config = AutoConfig.from_pretrained(info["path"], trust_remote_code=True)
        assert config.model_type == info["model_type"]
        assert config.hidden_size == info["expected_hidden"]
        assert config.num_hidden_layers == info["expected_layers"]

    @pytest.mark.parametrize("name,info", _CONFIG_AVAILABLE, ids=[n for n, _ in _CONFIG_AVAILABLE])
    def test_config_in_model_class_map(self, name: str, info: dict):
        """解析出的 model_type 应存在于 _MODEL_CLASS_MAP 中。"""
        config = AutoConfig.from_pretrained(info["path"], trust_remote_code=True)
        assert config.model_type in _MODEL_CLASS_MAP, (
            f"model_type={config.model_type} 未在 _MODEL_CLASS_MAP 中注册"
        )


# ═══════════════════════════════════════════════════════════════════
#  Part 2: 完整模型加载 + Gate 验证（需 GPU + 本地权重）
# ═══════════════════════════════════════════════════════════════════

def _has_weights(path: str) -> bool:
    """检查本地目录是否有模型权重文件。"""
    if not os.path.isdir(path):
        return False
    return any(
        f.endswith((".safetensors", ".bin"))
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f))
    )


LOADABLE_MODELS = [
    (name, info)
    for name, info in MODEL_REGISTRY.items()
    if _has_weights(info["path"])
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")
class TestFullModelLoading:
    """验证 TokenMemForCausalLM 能加载各家族模型并正确初始化 gate。"""

    @pytest.fixture(params=[
        pytest.param(
            (mt, info),
            id=mt,
            marks=pytest.mark.skipif(
                info["expected_hidden"] > 2048,
                reason=f"模型过大 (hidden={info['expected_hidden']})，跳过完整加载"
            ),
        )
        for mt, info in LOADABLE_MODELS
    ])
    def model_spec(self, request):
        return request.param

    def test_instantiation(self, model_spec):
        """TokenMemForCausalLM 应成功实例化。"""
        model_type, info = model_spec
        model = TokenMemForCausalLM(
            info["path"], knowledge_max_seq_len=16, torch_dtype=torch.bfloat16
        )
        assert model is not None

    def test_layer_access_path(self, model_spec):
        """model.model.model.layers 应可访问且层数正确。"""
        _, info = model_spec
        model = TokenMemForCausalLM(
            info["path"], knowledge_max_seq_len=16, torch_dtype=torch.bfloat16
        )
        layers = model.model.model.layers
        assert len(layers) == info["expected_layers"]

    def test_gates_exist_on_all_layers(self, model_spec):
        """每层 decoder layer 应都有 gate_crossattention 属性。"""
        _, info = model_spec
        model = TokenMemForCausalLM(
            info["path"], knowledge_max_seq_len=16, torch_dtype=torch.bfloat16
        )
        for idx, layer in enumerate(model.model.model.layers):
            assert hasattr(layer, "gate_crossattention"), (
                f"Layer {idx} 缺少 gate_crossattention"
            )

    def test_only_gates_trainable(self, model_spec):
        """仅 gate_crossattention 参数可训练。"""
        _, info = model_spec
        model = TokenMemForCausalLM(
            info["path"], knowledge_max_seq_len=16, torch_dtype=torch.bfloat16
        )
        trainable = {
            n for n, p in model.named_parameters() if p.requires_grad
        }
        assert len(trainable) > 0
        for name in trainable:
            assert "gate_crossattention" in name, (
                f"非 gate 参数可训练: {name}"
            )

    def test_gate_zero_init(self, model_spec):
        """W_B 应全零，W_A std 应接近 0.01。"""
        _, info = model_spec
        model = TokenMemForCausalLM(
            info["path"], knowledge_max_seq_len=16, torch_dtype=torch.bfloat16
        )
        for layer in model.model.model.layers:
            gate = layer.gate_crossattention
            assert torch.all(gate.W_B == 0), "W_B 应为零初始化"
            assert gate.W_A.float().std() < 0.05, (
                f"W_A std={gate.W_A.float().std():.4f}，预期 ~0.01"
            )
