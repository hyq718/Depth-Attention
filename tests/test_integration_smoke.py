from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from transformers import AutoModelForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig

from tests.compat import install_default_rope_if_missing

install_default_rope_if_missing(torch)

from llamafactory.model.modeling import (
    modeling_llama_attnres,
    modeling_llama_denseformer,
    modeling_llama_depth_attention,
    modeling_llama_mhc,
)
from llamafactory.model.llama_patch import (
    patch_llama_attnres,
    patch_llama_denseformer,
    patch_llama_depth_attention,
    patch_llama_mhc,
)


def config_for(method: str):
    data = dict(
        vocab_size=64,
        hidden_size=32,
        head_dim=8,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        attention_bias=False,
        attention_dropout=0.0,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        pretraining_tp=1,
        use_cache=True,
        use_qk_norm=True,
        rope_theta=10000.0,
        scale_embeds=False,
    )
    if method == "depth_attention":
        data.update(
            recurrent_model=True,
            cross_layer_pattern="depth_softmax",
            cross_layer_mode="depth_softmax",
            depth_softmax_stride=1,
            depth_recent_window=0,
        )
    elif method == "attnres":
        data.update(recurrent_model=True, baseline_mode="attnres", attnres_block_size=4)
    elif method == "denseformer":
        data.update(recurrent_model=True, baseline_mode="denseformer")
    elif method == "mhc":
        data.update(recurrent_model=True, baseline_mode="mhc", residual_baseline_num_streams=2)
    config = LlamaConfig(**data)
    for key in ("rope_theta", "rope_scaling"):
        if key in data and not hasattr(config, key):
            setattr(config, key, data[key])
    return config


@pytest.mark.parametrize(
    ("method", "module"),
    [
        ("depth_attention", modeling_llama_depth_attention),
        ("attnres", modeling_llama_attnres),
        ("denseformer", modeling_llama_denseformer),
        ("mhc", modeling_llama_mhc),
    ],
)
def test_one_step_train_save_load_round_trip(tmp_path, method, module):
    model = module.LlamaForCausalLM(config_for(method))
    model.train()
    input_ids = torch.tensor([[1, 14, 8, 2]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
    loss.backward()
    optimizer.step()

    save_dir = tmp_path / method
    model.save_pretrained(save_dir)
    loaded = module.LlamaForCausalLM.from_pretrained(save_dir)
    loaded.eval()
    with torch.no_grad():
        logits = loaded(input_ids=input_ids, use_cache=False).logits
    assert logits.shape == (1, 4, 64)


@pytest.mark.parametrize(
    ("method", "patch_fn", "module"),
    [
        ("depth_attention", patch_llama_depth_attention, modeling_llama_depth_attention),
        ("attnres", patch_llama_attnres, modeling_llama_attnres),
        ("denseformer", patch_llama_denseformer, modeling_llama_denseformer),
        ("mhc", patch_llama_mhc, modeling_llama_mhc),
    ],
)
def test_auto_model_patch_selects_expected_llama_implementation(method, patch_fn, module):
    patch_fn()
    model = AutoModelForCausalLM.from_config(config_for(method), trust_remote_code=True)
    assert isinstance(model, module.LlamaForCausalLM)
