# Copyright 2024 the LlamaFactory team.
# Copyright 2026 LUMIA Group.
#
# Licensed under the Apache License, Version 2.0.

"""
Monkey-patch transformers' built-in Llama model class with the Depth-Attention
implementation and the paper baselines used for comparison.

Method mapping
--------------
| --patch_method    | Modeling file                         | Source implementation |
|-------------------|---------------------------------------|-----------------------|
| depth_attention   | modeling_llama_depth_attention.py     | depthsoftmax          |
| attnres           | modeling_llama_attnres.py             | baseline_final        |
| denseformer       | modeling_llama_denseformer.py         | depthsoftmax_baselines |
| mhc               | modeling_llama_mhc.py                 | depthsoftmax_baselines |
| vanilla           | transformers stock                    | transformers          |

The vanilla path leaves the stock ``transformers`` implementation in place.
"""


def _swap_llama(cls) -> None:
    from transformers import AutoModelForCausalLM
    from transformers.models.llama.configuration_llama import LlamaConfig
    import transformers.models.llama.modeling_llama as modeling_llama

    modeling_llama.LlamaForCausalLM = cls
    AutoModelForCausalLM.register(LlamaConfig, cls, exist_ok=True)


def patch_llama_depth_attention() -> None:
    """Use the Depth-Attention Llama implementation."""
    from .modeling.modeling_llama_depth_attention import LlamaForCausalLM

    _swap_llama(LlamaForCausalLM)


def patch_llama_attnres() -> None:
    """Use the block AttnRes baseline implementation."""
    from .modeling.modeling_llama_attnres import LlamaForCausalLM

    _swap_llama(LlamaForCausalLM)


def patch_llama_denseformer() -> None:
    """Use the DenseFormer / DenseTransformer baseline implementation."""
    from .modeling.modeling_llama_denseformer import LlamaForCausalLM

    _swap_llama(LlamaForCausalLM)


def patch_llama_mhc() -> None:
    """Use the multi-stream residual-connector (mHC) baseline implementation."""
    from .modeling.modeling_llama_mhc import LlamaForCausalLM

    _swap_llama(LlamaForCausalLM)


patch_llama = patch_llama_depth_attention
