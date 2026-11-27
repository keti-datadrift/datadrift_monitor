import unittest

import torch
from torch import nn

from negzerohoc.vision_lora import (
    LoRALinear,
    VisionLoRAConfig,
    inject_clip_vision_lora,
    load_vision_lora_state_dict,
    vision_lora_parameters,
    vision_lora_state_dict,
)


class DummyAttention(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)


class DummyLayer(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.self_attn = DummyAttention(width)


class DummyEncoder(nn.Module):
    def __init__(self, width: int, layers: int):
        super().__init__()
        self.layers = nn.ModuleList([DummyLayer(width) for _ in range(layers)])


class DummyVisionModel(nn.Module):
    def __init__(self, width: int, layers: int):
        super().__init__()
        self.encoder = DummyEncoder(width, layers)


class DummyCLIP(nn.Module):
    def __init__(self, width: int = 8, layers: int = 3):
        super().__init__()
        self.vision_model = DummyVisionModel(width, layers)
        self.text_projection = nn.Linear(width, width)


class VisionLoRATest(unittest.TestCase):
    def test_injection_targets_only_selected_vision_layers(self):
        model = DummyCLIP()
        config = VisionLoRAConfig(
            rank=2,
            alpha=4.0,
            dropout=0.0,
            target_modules=("q_proj", "v_proj"),
            layers=(1, 2),
        )
        replaced = inject_clip_vision_lora(model, config)

        self.assertEqual(len(replaced), 4)
        self.assertIsInstance(model.vision_model.encoder.layers[1].self_attn.q_proj, LoRALinear)
        self.assertIsInstance(model.vision_model.encoder.layers[2].self_attn.v_proj, LoRALinear)
        self.assertIsInstance(model.vision_model.encoder.layers[0].self_attn.q_proj, nn.Linear)
        self.assertFalse(model.vision_model.encoder.layers[1].self_attn.k_proj.weight.requires_grad)
        self.assertTrue(all(parameter.requires_grad for parameter in vision_lora_parameters(model)))

    def test_zero_initialized_lora_preserves_base_output_and_round_trips(self):
        torch.manual_seed(0)
        model = DummyCLIP(width=4, layers=1)
        inputs = torch.randn(3, 4)
        base_output = model.vision_model.encoder.layers[0].self_attn.q_proj(inputs).detach()
        inject_clip_vision_lora(
            model,
            VisionLoRAConfig(rank=2, alpha=2.0, dropout=0.0, target_modules=("q_proj",)),
        )
        adapted = model.vision_model.encoder.layers[0].self_attn.q_proj
        self.assertTrue(torch.allclose(base_output, adapted(inputs), atol=1e-7))

        with torch.no_grad():
            adapted.lora_b.weight.fill_(0.25)
        state = vision_lora_state_dict(model)

        restored = DummyCLIP(width=4, layers=1)
        inject_clip_vision_lora(
            restored,
            VisionLoRAConfig(rank=2, alpha=2.0, dropout=0.0, target_modules=("q_proj",)),
        )
        load_vision_lora_state_dict(restored, state)
        restored_module = restored.vision_model.encoder.layers[0].self_attn.q_proj
        self.assertTrue(torch.allclose(adapted.lora_a.weight, restored_module.lora_a.weight))
        self.assertTrue(torch.allclose(adapted.lora_b.weight, restored_module.lora_b.weight))


if __name__ == "__main__":
    unittest.main()
