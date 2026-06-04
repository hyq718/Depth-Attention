from __future__ import annotations


def install_default_rope_if_missing(torch_module) -> None:
    """Patch older transformers test environments that lack default LLaMA RoPE."""
    from transformers import modeling_rope_utils

    if "default" in modeling_rope_utils.ROPE_INIT_FUNCTIONS:
        return

    def _compute_default_rope_parameters(config, device=None, seq_len=None, **rope_kwargs):
        del seq_len, rope_kwargs
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch_module.arange(0, head_dim, 2, dtype=torch_module.int64).float().to(device) / head_dim)
        )
        return inv_freq, 1.0

    modeling_rope_utils.ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters
