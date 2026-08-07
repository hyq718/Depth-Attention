"""Qwen3 P2N-V1 Jacobi training patch.

The patch keeps the stock Qwen3 parameterization and replaces only
``Qwen3Model.forward``.  A forward consists of one injection-free warm pass
followed by K Jacobi updates.  During training K can be sampled uniformly and
is broadcast from rank zero so every data-parallel rank executes the same
graph.
"""

from types import MethodType
from typing import TYPE_CHECKING, Optional

import torch


if TYPE_CHECKING:
    from transformers import PretrainedConfig


_PATCHED = False


def p2n_shift_previous(hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Shift batch-first hidden states right without feeding sequence starts."""

    if hidden_states.ndim != 3:
        raise ValueError(f"P2N expects [batch, sequence, hidden], got {tuple(hidden_states.shape)}.")
    shifted = torch.cat((torch.zeros_like(hidden_states[:, :1]), hidden_states[:, :-1]), dim=1)
    if attention_mask is not None and attention_mask.ndim == 2:
        valid = attention_mask.to(device=shifted.device, dtype=torch.bool)
        previous_valid = torch.cat((torch.zeros_like(valid[:, :1]), valid[:, :-1]), dim=1)
        shifted = shifted * (valid & previous_valid).unsqueeze(-1).to(dtype=shifted.dtype)
    return shifted


def p2n_v1_mix(embedding: torch.Tensor, normalized_previous_hidden: torch.Tensor, attention_mask=None) -> torch.Tensor:
    """Match Megatron P2N-V1's fixed-scale embedding/feedback mixture."""

    if embedding.shape != normalized_previous_hidden.shape or embedding.ndim != 3:
        raise ValueError(
            "P2N expects equal [batch, sequence, hidden] tensors, got "
            f"{tuple(embedding.shape)} and {tuple(normalized_previous_hidden.shape)}."
        )
    hidden_size = embedding.shape[-1]
    if hidden_size < 1:
        raise ValueError("P2N requires a positive hidden size.")
    shifted = p2n_shift_previous(normalized_previous_hidden, attention_mask)
    inv_sqrt_two = 2.0**-0.5
    return embedding * inv_sqrt_two + shifted * (inv_sqrt_two / hidden_size**0.5)


def _sample_update_count(config: "PretrainedConfig", device: torch.device, training: bool) -> int:
    maximum = int(getattr(config, "p2n_jacobi_iterations", 3))
    minimum = int(getattr(config, "p2n_jacobi_iterations_min", 2))
    if maximum < 1 or not 1 <= minimum <= maximum:
        raise ValueError(f"P2N requires 1 <= min <= max, got min={minimum}, max={maximum}.")
    if not training or not bool(getattr(config, "p2n_random_iterations", False)):
        return maximum

    count = torch.empty(1, dtype=torch.long, device=device)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        count.random_(minimum, maximum + 1)
    if torch.distributed.is_initialized():
        torch.distributed.broadcast(count, src=0)
    return int(count.item())


def patch_qwen3_p2n() -> None:
    """Install the P2N-V1 model-forward patch exactly once per process."""

    global _PATCHED
    if _PATCHED:
        return

    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

    original_forward = Qwen3Model.forward

    def p2n_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        **kwargs,
    ):
        if not bool(getattr(self.config, "use_p2n", False)):
            return original_forward(
                self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                **kwargs,
            )
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("P2N requires exactly one of input_ids or inputs_embeds.")
        if past_key_values is not None or use_cache:
            raise RuntimeError("Qwen3 P2N cached decoding is not implemented; use full-sequence evaluation.")

        embedding = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        update_count = _sample_update_count(self.config, embedding.device, self.training)
        shared = dict(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        # Z^(0): one injection-free warm pass.  Qwen3Model returns the final
        # RMS-normalized state, exactly the feedback state needed by V1.
        state = original_forward(self, input_ids=None, inputs_embeds=embedding * (2.0**-0.5), **shared).last_hidden_state
        final_output = None
        for update_index in range(update_count):
            mixed = p2n_v1_mix(embedding, state, attention_mask)
            is_final = update_index + 1 == update_count
            final_output = original_forward(
                self,
                input_ids=None,
                inputs_embeds=mixed,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                output_attentions=output_attentions if is_final else False,
                output_hidden_states=output_hidden_states if is_final else False,
                return_dict=True,
                cache_position=cache_position,
                **kwargs,
            )
            state = final_output.last_hidden_state

        seen = int(getattr(self, "_p2n_debug_samples", 0))
        if seen < 32 and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            boundary_abs_max = float(p2n_shift_previous(state, attention_mask)[:, 0].abs().max().item())
            print(
                "[P2N] JACOBI_ACTIVE "
                f"variant=v1 warm_start=1 updates={update_count} layers={len(self.layers)} "
                f"layer_evaluations={(update_count + 1) * len(self.layers)} "
                f"embedding_scale={2.0**-0.5:.12f} "
                f"hidden_scale={(2.0**-0.5)/(embedding.shape[-1]**0.5):.12f} "
                f"boundary_injection_abs_max={boundary_abs_max:.1f}",
                flush=True,
            )
        self._p2n_debug_samples = seen + 1

        if return_dict is False:
            return final_output.to_tuple()
        return final_output

    Qwen3Model.forward = p2n_forward
    Qwen3Model._p2n_original_forward = original_forward
    _PATCHED = True

