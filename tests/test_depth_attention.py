from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from transformers.models.llama.configuration_llama import LlamaConfig

from tests.compat import install_default_rope_if_missing

install_default_rope_if_missing(torch)

from llamafactory.model.modeling import modeling_llama_depth_attention

ROOT = Path(__file__).resolve().parents[1]


def tiny_config(**overrides):
    data = json.loads((ROOT / "llama_config" / "depth_attention_tiny" / "config.json").read_text(encoding="utf-8"))
    data.update(overrides)
    config = LlamaConfig(**data)
    for key in ("rope_theta", "rope_scaling"):
        if key in data and not hasattr(config, key):
            setattr(config, key, data[key])
    return config


def test_depth_attention_split_mix_matches_reference_formula():
    torch.manual_seed(0)
    query = torch.randn(2, 3, 4, 5)
    cross_key = torch.randn(2, 2, 3, 4, 5)
    cross_value = torch.randn(2, 2, 3, 4, 5)
    self_key = torch.randn(2, 3, 4, 5)
    self_value = torch.randn(2, 3, 4, 5)
    scale = math.sqrt(query.shape[-1])

    actual = modeling_llama_depth_attention._depth_attention_split_mix(
        query, cross_key, cross_value, self_key, self_value, scale
    )

    cross_scores = torch.einsum("bhtd,nbhtd->bhtn", query, cross_key) / scale
    self_score = (query * self_key).sum(-1, keepdim=True) / scale
    weights = torch.softmax(torch.cat([cross_scores, self_score], dim=-1), dim=-1)
    expected = torch.einsum("bhtn,nbhtd->bhtd", weights[..., :-1], cross_value)
    expected = expected + weights[..., -1:] * self_value
    torch.testing.assert_close(actual, expected)


def test_depth_attention_tiny_gqa_forward_backward():
    model = modeling_llama_depth_attention.LlamaForCausalLM(tiny_config())
    input_ids = torch.tensor([[1, 14, 8, 18, 2]])
    outputs = model(input_ids=input_ids, labels=input_ids, use_cache=False)
    assert outputs.logits.shape == (1, 5, model.config.vocab_size)
    outputs.loss.backward()
    assert model.lm_head.weight.grad is not None


def test_depth_attention_cache_round_trip_shapes():
    model = modeling_llama_depth_attention.LlamaForCausalLM(tiny_config())
    first = torch.tensor([[1, 14, 8]])
    out = model(input_ids=first, attention_mask=torch.ones_like(first), use_cache=True)
    next_ids = torch.tensor([[18]])
    next_mask = torch.ones((1, 4), dtype=torch.long)
    out_next = model(
        input_ids=next_ids,
        attention_mask=next_mask,
        past_key_values=out.past_key_values,
        use_cache=True,
    )
    assert out_next.logits.shape[:2] == (1, 1)
    assert out_next.past_key_values is not None
