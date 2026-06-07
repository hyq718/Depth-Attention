# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import _flash_attention_forward

import os as _os


def _trace_env_int(name: str, default: int) -> int:
    value = _os.environ.get(name)
    if value is None:
        return default
    try:
        return max(1, int(value))
    except ValueError:
        return default


def _trace_structured_samples(tensor: torch.Tensor) -> dict[str, dict[str, list[float]]]:
    if tensor.ndim != 4:
        return {}

    batch_limit = min(_trace_env_int("TRACE_SAMPLE_BATCH_LIMIT", 2), tensor.shape[0])
    head_limit = min(_trace_env_int("TRACE_SAMPLE_HEAD_LIMIT", 2), tensor.shape[1])
    token_limit = min(_trace_env_int("TRACE_SAMPLE_TOKEN_LIMIT", 2), tensor.shape[2])
    checksum_head_limit = min(_trace_env_int("TRACE_CHECKSUM_HEAD_LIMIT", 8), tensor.shape[1])
    checksum_token_limit = min(
        _trace_env_int("TRACE_CHECKSUM_TOKEN_LIMIT", 8), tensor.shape[2]
    )
    width_limit = min(_trace_env_int("TRACE_SAMPLE_WIDTH_LIMIT", 64), tensor.shape[3])

    sample_vectors: dict[str, list[float]] = {}
    for batch_idx in range(batch_limit):
        for head_idx in range(head_limit):
            for token_idx in range(token_limit):
                sample_vectors[f"b{batch_idx}_h{head_idx}_t{token_idx}"] = (
                    tensor[batch_idx, head_idx, token_idx, :width_limit].tolist()
                )

    token_checksums: dict[str, list[float]] = {}
    for batch_idx in range(batch_limit):
        for head_idx in range(head_limit):
            token_checksums[f"b{batch_idx}_h{head_idx}"] = (
                tensor[batch_idx, head_idx, :checksum_token_limit, :].sum(dim=-1).tolist()
            )

    head_checksums: dict[str, list[float]] = {}
    for batch_idx in range(batch_limit):
        for token_idx in range(token_limit):
            head_checksums[f"b{batch_idx}_t{token_idx}"] = (
                tensor[batch_idx, :checksum_head_limit, token_idx, :].sum(dim=-1).tolist()
            )

    return {
        "sample_vectors": sample_vectors,
        "token_checksums": token_checksums,
        "head_checksums": head_checksums,
    }


def _trace_tensor_stats(tensor: torch.Tensor, limit: int = 8192):
    detached = tensor.detach().float().cpu()
    flat = detached.reshape(-1)
    head = flat[: min(limit, flat.numel())]
    stats = {
        "shape": list(detached.shape),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "maxabs": float(flat.abs().max().item()),
        "l2": float(torch.linalg.vector_norm(flat).item()),
        "checksum_head": float(head.sum().item()),
        "head_values": head[: min(16, head.numel())].tolist(),
    }
    stats.update(_trace_structured_samples(detached))
    return stats

def _reset_attention_trace(module: nn.Module):
    if getattr(module, "_trace_probe_enabled", False):
        module._trace_last_probe = {}


def _capture_attention_trace(module: nn.Module, name: str, tensor: Optional[torch.Tensor]):
    if tensor is None or not getattr(module, "_trace_probe_enabled", False):
        return
    if getattr(module, "_trace_last_probe", None) is None:
        module._trace_last_probe = {}
    module._trace_last_probe[name] = _trace_tensor_stats(tensor)


def _capture_rope_phase_trace(
    module: nn.Module,
    cos: Optional[torch.Tensor],
    sin: Optional[torch.Tensor],
    phase: Optional[torch.Tensor] = None,
):
    if not getattr(module, "_trace_probe_enabled", False):
        return
    if phase is not None:
        phase = phase.unsqueeze(1) if phase.ndim == 3 else phase
        _capture_attention_trace(module, "query_rope_emb", phase)
        _capture_attention_trace(module, "key_rope_emb", phase)
    if cos is not None:
        _capture_attention_trace(
            module,
            "query_rope_cos",
            cos.unsqueeze(1) if cos.ndim == 3 else cos,
        )
    if sin is not None:
        _capture_attention_trace(
            module,
            "query_rope_sin",
            sin.unsqueeze(1) if sin.ndim == 3 else sin,
        )


# ---------------------------------------------------------------------------
# Compiled depth_softmax mixing: narrow-scope torch.compile for inductor fusion.
# Inspired by a fellow agent's approach in figs/modeling_llama_fast.py.
# Inputs/outputs are pure tensors so inductor can fuse the whole block.
# ---------------------------------------------------------------------------
def _depth_softmax_split_mix(query_states, cross_k_normed, cross_v, self_k_normed, self_v, scale):
    """Depth-wise softmax mixing using split-scoring (no 5D cat).
    Inputs:
      query_states: [B, H, T, d]
      cross_k_normed: [N, B, H, T, d] — already k_normed
      cross_v: [N, B, H, T, d]
      self_k_normed: [B, H, T, d]
      self_v: [B, H, T, d]
      scale: float (sqrt head_dim)
    Returns:
      mixed_v: [B, H, T, d]
    """
    cross_scores = torch.einsum("bhtd,nbhtd->bhtn", query_states, cross_k_normed) / scale
    self_score = (query_states * self_k_normed).sum(-1, keepdim=True) / scale
    depth_scores = torch.cat([cross_scores, self_score], dim=-1)
    depth_weights = torch.nn.functional.softmax(
        depth_scores, dim=-1, dtype=torch.float32
    ).to(query_states.dtype)
    cross_weights = depth_weights[..., :-1]
    self_weight = depth_weights[..., -1:]
    cross_mixed = torch.einsum("bhtn,nbhtd->bhtd", cross_weights, cross_v)
    return cross_mixed + self_weight * self_v


def _depth_softmax_stream_mix(query_states, cross_k_normed, cross_v, self_k_normed, self_v, scale):
    """Streaming online-softmax depth mixing (flash-attention style).
    No scores/weights materialization. Uses running max/denom/output state.
    Iterates over cross sources via unbind (unrolls with dynamic=False compile).

    Inputs same as _depth_softmax_split_mix.
    Returns mixed_v: [B, H, T, d]
    """
    dtype = query_states.dtype

    # Unbind cross K/V into tuples (views, no copy)
    cross_k_list = torch.unbind(cross_k_normed, dim=0)  # tuple of N [B,H,T,d]
    cross_v_list = torch.unbind(cross_v, dim=0)

    # Initialize with first cross source
    score = (query_states * cross_k_list[0]).sum(-1) / scale  # [B,H,T]
    max_score = score.float()
    denom = torch.ones_like(max_score)
    out_fp32 = cross_v_list[0].float()

    # Stream remaining cross sources with online softmax
    for i in range(1, len(cross_k_list)):
        score_i = ((query_states * cross_k_list[i]).sum(-1) / scale).float()
        new_max = torch.maximum(max_score, score_i)
        old_scale = torch.exp(max_score - new_max)
        new_scale = torch.exp(score_i - new_max)
        denom = denom * old_scale + new_scale
        out_fp32 = out_fp32 * old_scale.unsqueeze(-1) + cross_v_list[i].float() * new_scale.unsqueeze(-1)
        max_score = new_max

    # Add self contribution
    self_score = ((query_states * self_k_normed).sum(-1) / scale).float()
    new_max = torch.maximum(max_score, self_score)
    old_scale = torch.exp(max_score - new_max)
    new_scale = torch.exp(self_score - new_max)
    denom = denom * old_scale + new_scale
    out_fp32 = out_fp32 * old_scale.unsqueeze(-1) + self_v.float() * new_scale.unsqueeze(-1)

    return (out_fp32 / denom.unsqueeze(-1)).to(dtype)


# Cached compiled versions (keyed by mode + N for per-shape specialization)
_DEPTH_MIX_COMPILED = {}
_DEPTH_MIX_COMPILE_ENABLED = _os.environ.get("LLAMA_DEPTH_COMPILE_MIX", "0") == "1"
# Which implementation to use: "split" (2 einsums) or "stream" (online softmax)
_DEPTH_MIX_IMPL = _os.environ.get("LLAMA_DEPTH_MIX_IMPL", "split")


def _get_compiled_depth_mix():
    """Lazy-compile the depth mixing function once (dynamic=False for best fusion)."""
    global _DEPTH_MIX_COMPILED
    impl_name = _DEPTH_MIX_IMPL
    impl_fn = _depth_softmax_stream_mix if impl_name == "stream" else _depth_softmax_split_mix
    if not _DEPTH_MIX_COMPILE_ENABLED:
        return impl_fn
    cache_key = f"compiled_{impl_name}"
    if cache_key not in _DEPTH_MIX_COMPILED:
        try:
            _DEPTH_MIX_COMPILED[cache_key] = torch.compile(
                impl_fn, fullgraph=False, dynamic=False
            )
        except Exception:
            _DEPTH_MIX_COMPILED[cache_key] = impl_fn
    return _DEPTH_MIX_COMPILED[cache_key]


# ---------------------------------------------------------------------------
# Compiled mHC residual stream connector.
#
# The paper's efficient implementation depends on fusing a large number of
# small N=1/2/4 map/update ops. The default eager PyTorch implementation is kept
# below for readability and correctness tests; this path is opt-in through
# LLAMA_MHC_COMPILE=1 and keeps the same tensor formulas.
# ---------------------------------------------------------------------------
def _mhc_sinkhorn_tensor(logits: torch.Tensor, iters: int) -> torch.Tensor:
    matrix = logits.float()
    matrix = matrix - matrix.amax(dim=(-1, -2), keepdim=True)
    matrix = matrix.exp().clamp_min(1e-12)
    for _ in range(iters):
        matrix = matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        matrix = matrix / matrix.sum(dim=-2, keepdim=True).clamp_min(1e-12)
    return matrix


def _mhc_mix_n1_tensor(h_pre: torch.Tensor, residual_streams: torch.Tensor) -> torch.Tensor:
    return h_pre[..., 0].unsqueeze(-1) * residual_streams[:, :, 0, :]


def _mhc_update_n1_tensor(
    residual_streams: torch.Tensor,
    hidden_update: torch.Tensor,
    h_post: torch.Tensor,
    h_res: torch.Tensor,
) -> torch.Tensor:
    out0 = (
        h_res[..., 0, 0].unsqueeze(-1) * residual_streams[:, :, 0, :]
        + h_post[..., 0].unsqueeze(-1) * hidden_update
    )
    return out0.unsqueeze(2)


def _mhc_mix_n2_tensor(h_pre: torch.Tensor, residual_streams: torch.Tensor) -> torch.Tensor:
    r0, r1 = residual_streams.unbind(dim=2)
    return h_pre[..., 0].unsqueeze(-1) * r0 + h_pre[..., 1].unsqueeze(-1) * r1


def _mhc_update_n2_tensor(
    residual_streams: torch.Tensor,
    hidden_update: torch.Tensor,
    h_post: torch.Tensor,
    h_res: torch.Tensor,
) -> torch.Tensor:
    r0, r1 = residual_streams.unbind(dim=2)
    out0 = (
        h_res[..., 0, 0].unsqueeze(-1) * r0
        + h_res[..., 0, 1].unsqueeze(-1) * r1
        + h_post[..., 0].unsqueeze(-1) * hidden_update
    )
    out1 = (
        h_res[..., 1, 0].unsqueeze(-1) * r0
        + h_res[..., 1, 1].unsqueeze(-1) * r1
        + h_post[..., 1].unsqueeze(-1) * hidden_update
    )
    return torch.stack((out0, out1), dim=2)


def _mhc_mix_n4_tensor(h_pre: torch.Tensor, residual_streams: torch.Tensor) -> torch.Tensor:
    r0, r1, r2, r3 = residual_streams.unbind(dim=2)
    return (
        h_pre[..., 0].unsqueeze(-1) * r0
        + h_pre[..., 1].unsqueeze(-1) * r1
        + h_pre[..., 2].unsqueeze(-1) * r2
        + h_pre[..., 3].unsqueeze(-1) * r3
    )


def _mhc_update_n4_tensor(
    residual_streams: torch.Tensor,
    hidden_update: torch.Tensor,
    h_post: torch.Tensor,
    h_res: torch.Tensor,
) -> torch.Tensor:
    r0, r1, r2, r3 = residual_streams.unbind(dim=2)
    out0 = (
        h_res[..., 0, 0].unsqueeze(-1) * r0
        + h_res[..., 0, 1].unsqueeze(-1) * r1
        + h_res[..., 0, 2].unsqueeze(-1) * r2
        + h_res[..., 0, 3].unsqueeze(-1) * r3
        + h_post[..., 0].unsqueeze(-1) * hidden_update
    )
    out1 = (
        h_res[..., 1, 0].unsqueeze(-1) * r0
        + h_res[..., 1, 1].unsqueeze(-1) * r1
        + h_res[..., 1, 2].unsqueeze(-1) * r2
        + h_res[..., 1, 3].unsqueeze(-1) * r3
        + h_post[..., 1].unsqueeze(-1) * hidden_update
    )
    out2 = (
        h_res[..., 2, 0].unsqueeze(-1) * r0
        + h_res[..., 2, 1].unsqueeze(-1) * r1
        + h_res[..., 2, 2].unsqueeze(-1) * r2
        + h_res[..., 2, 3].unsqueeze(-1) * r3
        + h_post[..., 2].unsqueeze(-1) * hidden_update
    )
    out3 = (
        h_res[..., 3, 0].unsqueeze(-1) * r0
        + h_res[..., 3, 1].unsqueeze(-1) * r1
        + h_res[..., 3, 2].unsqueeze(-1) * r2
        + h_res[..., 3, 3].unsqueeze(-1) * r3
        + h_post[..., 3].unsqueeze(-1) * hidden_update
    )
    return torch.stack((out0, out1, out2, out3), dim=2)


def _mhc_mix_small_n_tensor(
    h_pre: torch.Tensor,
    residual_streams: torch.Tensor,
    num_streams: int,
) -> torch.Tensor:
    if num_streams == 1:
        return _mhc_mix_n1_tensor(h_pre, residual_streams)
    if num_streams == 2:
        return _mhc_mix_n2_tensor(h_pre, residual_streams)
    if num_streams == 4:
        return _mhc_mix_n4_tensor(h_pre, residual_streams)
    return torch.einsum("btn,btnd->btd", h_pre, residual_streams)


def _mhc_update_small_n_tensor(
    residual_streams: torch.Tensor,
    hidden_update: torch.Tensor,
    h_post: torch.Tensor,
    h_res: torch.Tensor,
) -> torch.Tensor:
    num_streams = residual_streams.shape[2]
    if num_streams == 1 and h_res.shape[-1] == 1:
        return _mhc_update_n1_tensor(residual_streams, hidden_update, h_post, h_res)
    if num_streams == 2 and h_res.shape[-1] == 2:
        return _mhc_update_n2_tensor(residual_streams, hidden_update, h_post, h_res)
    if num_streams == 4 and h_res.shape[-1] == 4:
        return _mhc_update_n4_tensor(residual_streams, hidden_update, h_post, h_res)
    mixed_streams = torch.einsum("btij,btjd->btid", h_res, residual_streams)
    return mixed_streams + h_post.unsqueeze(-1) * hidden_update.unsqueeze(2)


def _mhc_maps_mix_tensor(
    residual_streams: torch.Tensor,
    norm_weight: torch.Tensor,
    projection_weight: torch.Tensor,
    alpha_pre: torch.Tensor,
    alpha_post: torch.Tensor,
    alpha_res: torch.Tensor,
    pre_logits: torch.Tensor,
    post_logits: torch.Tensor,
    residual_logits: torch.Tensor,
    rms_norm_eps: float,
    num_streams: int,
    hidden_size: int,
    sinkhorn_iters: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat_streams = residual_streams.reshape(
        *residual_streams.shape[:2],
        num_streams * hidden_size,
    )
    input_dtype = flat_streams.dtype
    flat_streams_fp32 = flat_streams.to(torch.float32)
    variance = flat_streams_fp32.pow(2).mean(-1, keepdim=True)
    normed_streams = flat_streams_fp32 * torch.rsqrt(variance + rms_norm_eps)
    normed_streams = norm_weight * normed_streams.to(input_dtype)

    dynamic_logits = F.linear(normed_streams, projection_weight).float()
    pre_dynamic, post_dynamic, residual_dynamic = torch.split(
        dynamic_logits,
        [num_streams, num_streams, num_streams * num_streams],
        dim=-1,
    )
    residual_dynamic = residual_dynamic.reshape(
        *residual_streams.shape[:2],
        num_streams,
        num_streams,
    )

    h_pre_logits = alpha_pre.float() * pre_dynamic + pre_logits.float()
    h_post_logits = alpha_post.float() * post_dynamic + post_logits.float()
    h_res_logits = alpha_res.float() * residual_dynamic + residual_logits.float()

    h_pre = torch.sigmoid(h_pre_logits).to(dtype=residual_streams.dtype)
    h_post = (2.0 * torch.sigmoid(h_post_logits)).to(dtype=residual_streams.dtype)
    h_res = _mhc_sinkhorn_tensor(h_res_logits, sinkhorn_iters).to(dtype=residual_streams.dtype)
    if _MHC_FIXED_SMALL_N_ENABLED and num_streams in (1, 2, 4):
        mixed = _mhc_mix_small_n_tensor(h_pre, residual_streams, num_streams)
    else:
        mixed = torch.einsum("btn,btnd->btd", h_pre, residual_streams)
    return mixed, h_post, h_res


def _mhc_update_tensor(
    residual_streams: torch.Tensor,
    hidden_update: torch.Tensor,
    h_post: torch.Tensor,
    h_res: torch.Tensor,
) -> torch.Tensor:
    if _MHC_FIXED_SMALL_N_ENABLED and residual_streams.shape[2] in (1, 2, 4):
        return _mhc_update_small_n_tensor(residual_streams, hidden_update, h_post, h_res)
    mixed_streams = torch.einsum("btij,btjd->btid", h_res, residual_streams)
    return mixed_streams + h_post.unsqueeze(-1) * hidden_update.unsqueeze(2)


_MHC_COMPILED = {}
_MHC_COMPILE_ENABLED = _os.environ.get("LLAMA_MHC_COMPILE", "0") == "1"
_MHC_COMPILE_MODE = _os.environ.get("LLAMA_MHC_COMPILE_MODE", "default")
_MHC_COMPILE_CUDAGRAPHS = _os.environ.get("LLAMA_MHC_COMPILE_CUDAGRAPHS", "0") == "1"
_MHC_FIXED_N4_ENABLED = _os.environ.get("LLAMA_MHC_FIXED_N4", "1") == "1"
_MHC_FIXED_SMALL_N_ENABLED = _os.environ.get(
    "LLAMA_MHC_FIXED_SMALL_N",
    "1" if _MHC_FIXED_N4_ENABLED else "0",
) == "1"


def _maybe_compile_mhc(fn, cache_key: str):
    if not _MHC_COMPILE_ENABLED:
        return fn
    if cache_key not in _MHC_COMPILED:
        compile_kwargs = {
            "fullgraph": False,
            "dynamic": False,
        }
        if _MHC_COMPILE_CUDAGRAPHS:
            compile_kwargs["mode"] = _MHC_COMPILE_MODE
        elif _MHC_COMPILE_MODE == "max-autotune-no-cudagraphs":
            compile_kwargs["mode"] = _MHC_COMPILE_MODE
        else:
            compile_kwargs["options"] = {
                "triton.cudagraphs": False,
                "triton.cudagraph_trees": False,
            }
        try:
            _MHC_COMPILED[cache_key] = torch.compile(fn, **compile_kwargs)
        except Exception:
            _MHC_COMPILED[cache_key] = fn
    return _MHC_COMPILED[cache_key]


def _get_mhc_maps_mix_fn(num_streams: int):
    return _maybe_compile_mhc(_mhc_maps_mix_tensor, f"maps_mix_n{num_streams}")


def _get_mhc_update_fn(num_streams: int):
    return _maybe_compile_mhc(_mhc_update_tensor, f"update_n{num_streams}")


# ---------------------------------------------------------------------------
# Multi-segment flash-attention with exact joint-softmax gradients (LSE merge)
# Each segment does flash-attn forward(q, k_i, v_i), then merges by log-sum-exp.
# ---------------------------------------------------------------------------
class _MultiFlashAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, *args):
        try:
            from flash_attn.flash_attn_interface import _flash_attn_forward
        except Exception as e:
            raise RuntimeError("multi_flash_attn requires flash_attn >= 2.x") from e

        dropout_p = args[-3]
        softmax_scale = args[-2]
        causal = args[-1]
        kv = args[:-3]
        n_seg = len(kv) // 2
        k_list = list(kv[0::2])
        v_list = list(kv[1::2])
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        q = q.contiguous()
        k_list = [k.contiguous() for k in k_list]
        v_list = [v.contiguous() for v in v_list]

        out_list, lse_list, rng_list = [], [], []
        for k, v in zip(k_list, v_list):
            out_i, lse_i, _, rng_i = _flash_attn_forward(
                q, k, v, dropout_p, softmax_scale, causal,
                -1, -1, 0.0, None, False,
            )
            out_list.append(out_i)
            lse_list.append(lse_i)
            rng_list.append(rng_i)

        # LSE merge (fp32)
        lse_bl_h1 = [lse.transpose(1, 2).unsqueeze(-1).float() for lse in lse_list]
        lse_stack = torch.stack(lse_bl_h1, dim=0)
        mx = torch.amax(lse_stack, dim=0)
        exp_stack = torch.exp(lse_stack - mx)
        denom = torch.sum(exp_stack, dim=0)
        w_stack = exp_stack / denom

        out_merged_fp32 = torch.zeros_like(out_list[0], dtype=torch.float32)
        for i in range(n_seg):
            out_merged_fp32 = out_merged_fp32 + w_stack[i] * out_list[i].float()
        out_merged = out_merged_fp32.to(q.dtype).contiguous()

        to_save = [q, out_merged]
        to_save.extend(k_list)
        to_save.extend(v_list)
        to_save.extend(lse_list)
        w_list = [w_stack[i].contiguous() for i in range(n_seg)]
        to_save.extend(w_list)
        ctx.save_for_backward(*to_save)
        ctx.n_seg = n_seg
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.rng_list = rng_list
        return out_merged

    @staticmethod
    def backward(ctx, grad_output):
        try:
            from flash_attn.flash_attn_interface import _flash_attn_backward
        except Exception as e:
            raise RuntimeError("multi_flash_attn backward requires flash_attn") from e

        grad_output = grad_output.contiguous()
        saved = ctx.saved_tensors
        n = ctx.n_seg
        q = saved[0]; out_m = saved[1]
        idx = 2
        k_list = list(saved[idx:idx+n]); idx += n
        v_list = list(saved[idx:idx+n]); idx += n
        lse_list = list(saved[idx:idx+n]); idx += n
        w_list = list(saved[idx:idx+n]); idx += n

        dq_total = torch.zeros_like(q)
        dk_list, dv_list = [], []
        for i in range(n):
            w_i = w_list[i]
            dout_i = (w_i * grad_output.float()).to(q.dtype).contiguous()
            dq = torch.empty_like(q)
            dk = torch.empty_like(k_list[i])
            dv = torch.empty_like(v_list[i])
            _flash_attn_backward(
                dout_i, q, k_list[i], v_list[i], out_m, lse_list[i],
                dq, dk, dv, ctx.dropout_p, ctx.softmax_scale, ctx.causal,
                -1, -1, 0.0, None, False, ctx.rng_list[i],
            )
            dq_total.add_(dq)
            dk_list.append(dk)
            dv_list.append(dv)

        grads = [dq_total]
        for i in range(n):
            grads.append(dk_list[i])
            grads.append(dv_list[i])
        grads.extend([None, None, None])
        return tuple(grads)


def multi_flash_attn(q, kv_pairs, dropout_p=0.0, softmax_scale=None, causal=True):
    """q: [B,L,Hq,d], kv_pairs: List[(k,v)] each [B,Lk,Hkv,d]"""
    flat = []
    for k, v in kv_pairs:
        flat.append(k)
        flat.append(v)
    return _MultiFlashAttnFn.apply(q, *flat, dropout_p, softmax_scale, causal)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.models.llama.configuration_llama import LlamaConfig


# Baseline-final keeps only the residual-stream baselines used in the final
# comparison table. Select one from model config with `baseline_mode`.
# Legacy `residual_baseline` is accepted as a fallback for existing configs.

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "meta-llama/Llama-2-7b-hf"
_CONFIG_FOR_DOC = "LlamaConfig"

# ---------------------------------------------------------------------------
# Multi-segment flash-attention with exact joint-softmax gradients
# Generalization of your proven dual_flash_attn, supports N KV segments.
#
# Each segment i does a flash-attn forward(q, k_i, v_i) with causal=True,
# then we merge outputs by log-sum-exp across segments:
#   out = Σ_i w_i * out_i,   where w_i = exp(lse_i) / Σ_j exp(lse_j)
#
# Backward is exact joint-softmax by calling _flash_attn_backward with:
#   dout_i = w_i * grad_output
#   out    = out_merged (NOT out_i)
#   lse    = lse_i
# ---------------------------------------------------------------------------
class _MultiFlashAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, *args):
        """
        args layout:
          k1, v1, k2, v2, ..., kN, vN, dropout_p, softmax_scale, causal
        """
        try:
            from flash_attn.flash_attn_interface import _flash_attn_forward
        except Exception as e:
            raise RuntimeError(
                "multi_flash_attn 需要 flash_attn.flash_attn_interface "
                "(flash-attn >= 2.x). Import failed."
            ) from e

        if len(args) < 5:
            raise ValueError("multi_flash_attn requires at least 1 KV segment + (dropout_p, softmax_scale, causal).")

        dropout_p = args[-3]
        softmax_scale = args[-2]
        causal = args[-1]
        kv = args[:-3]
        if len(kv) % 2 != 0:
            raise ValueError("KV tensors must come in (k, v) pairs.")

        n_seg = len(kv) // 2
        k_list = list(kv[0::2])
        v_list = list(kv[1::2])

        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        # ensure contiguous
        q = q.contiguous()
        k_list = [k.contiguous() for k in k_list]
        v_list = [v.contiguous() for v in v_list]

        out_list = []
        lse_list = []
        rng_list = []

        # ---- forward per segment ----
        for k, v in zip(k_list, v_list):
            out_i, lse_i, _, rng_i = _flash_attn_forward(
                q, k, v,
                dropout_p,
                softmax_scale,
                causal,
                -1, -1,       # window_size_left, window_size_right
                0.0,          # softcap
                None,         # alibi_slopes
                False,        # return_softmax
            )
            out_list.append(out_i)
            lse_list.append(lse_i)
            rng_list.append(rng_i)

        # ---- LSE merge (fp32) ----
        # lse: [B, H, L] -> [B, L, H, 1]
        lse_bl_h1 = [lse.transpose(1, 2).unsqueeze(-1).float() for lse in lse_list]
        lse_stack = torch.stack(lse_bl_h1, dim=0)  # [N, B, L, H, 1]

        mx = torch.amax(lse_stack, dim=0)          # [B, L, H, 1]
        exp_stack = torch.exp(lse_stack - mx)      # [N, B, L, H, 1]
        denom = torch.sum(exp_stack, dim=0)        # [B, L, H, 1]
        w_stack = exp_stack / denom                # [N, B, L, H, 1]

        # out: each out_i is [B, L, Hq, d]
        # note: Hq can be >= Hkv (GQA/MQA) is ok in flash-attn; out_i matches q heads.
        out_merged_fp32 = torch.zeros_like(out_list[0], dtype=torch.float32)
        for i in range(n_seg):
            out_merged_fp32 = out_merged_fp32 + w_stack[i] * out_list[i].float()

        out_merged = out_merged_fp32.to(q.dtype).contiguous()

        # save tensors for backward
        # store: q, out_merged, then (k_i, v_i, lse_i, w_i) for each segment
        to_save = [q, out_merged]
        to_save.extend(k_list)
        to_save.extend(v_list)
        to_save.extend(lse_list)
        # save each w_i as tensor [B,L,H,1] (fp32)
        w_list = [w_stack[i].contiguous() for i in range(n_seg)]
        to_save.extend(w_list)

        ctx.save_for_backward(*to_save)
        ctx.n_seg = n_seg
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.rng_list = rng_list

        return out_merged

    @staticmethod
    def backward(ctx, grad_output):
        try:
            from flash_attn.flash_attn_interface import _flash_attn_backward
        except Exception as e:
            raise RuntimeError(
                "multi_flash_attn backward 需要 flash_attn.flash_attn_interface._flash_attn_backward"
            ) from e

        grad_output = grad_output.contiguous()

        saved = ctx.saved_tensors
        n = ctx.n_seg

        # unpack saved tensors
        # [0]=q, [1]=out_m,
        # next n: k_list
        # next n: v_list
        # next n: lse_list
        # next n: w_list
        q = saved[0]
        out_m = saved[1]

        idx = 2
        k_list = list(saved[idx:idx + n]); idx += n
        v_list = list(saved[idx:idx + n]); idx += n
        lse_list = list(saved[idx:idx + n]); idx += n
        w_list = list(saved[idx:idx + n]); idx += n

        dq_total = torch.zeros_like(q)
        dk_list = []
        dv_list = []

        # per segment backward with exact joint-softmax correction
        for i in range(n):
            w_i = w_list[i]  # fp32 [B,L,H,1]
            dout_i = (w_i * grad_output.float()).to(q.dtype).contiguous()

            dq = torch.empty_like(q)
            dk = torch.empty_like(k_list[i])
            dv = torch.empty_like(v_list[i])

            _flash_attn_backward(
                dout_i,
                q, k_list[i], v_list[i],
                out_m, lse_list[i],
                dq, dk, dv,
                ctx.dropout_p,
                ctx.softmax_scale,
                ctx.causal,
                -1, -1,       # window_size_left, window_size_right
                0.0,          # softcap
                None,         # alibi_slopes
                False,        # deterministic
                ctx.rng_list[i],
            )

            dq_total.add_(dq)
            dk_list.append(dk)
            dv_list.append(dv)

        # grads must match forward inputs: (q, k1, v1, ..., kN, vN, dropout_p, softmax_scale, causal)
        # non-tensor params -> None
        grads = [dq_total]
        for i in range(n):
            grads.append(dk_list[i])
            grads.append(dv_list[i])
        grads.extend([None, None, None])
        return tuple(grads)


def multi_flash_attn(q, kv_pairs, dropout_p=0.0, softmax_scale=None, causal=True):
    """
    kv_pairs: List[Tuple[k, v]] where each k/v is [B, Lk, Hkv, d]
    q: [B, Lq, Hq, d]
    """
    flat = []
    for k, v in kv_pairs:
        flat.append(k)
        flat.append(v)
    return _MultiFlashAttnFn.apply(q, *flat, dropout_p, softmax_scale, causal)


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[LlamaConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`LlamaRotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.46"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        self._last_trace_phase = emb

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "`LlamaLinearScalingRotaryEmbedding` is deprecated an will be removed in v4.46. Please use "
            "`LlamaRotaryEmbedding`, which now also does linear scaling (simply pass the model config to __init__)."
        )
        kwargs["rope_type"] = "linear"
        super().__init__(*args, **kwargs)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "`LlamaDynamicNTKScalingRotaryEmbedding` is deprecated an will be removed in v4.46. Please use "
            "`LlamaRotaryEmbedding`, which now also does dynamic ntk scaling (simply pass the model config to "
            "__init__)."
        )
        kwargs["rope_type"] = "dynamic"
        super().__init__(*args, **kwargs)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]
        self.down_proj._is_mlp_output = True  # 标记为MLP输出层
    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.o_proj._is_attention_output = True  # 标记为注意力输出层
        self._use_qk_norm = getattr(config, "use_qk_norm", True)
        if self._use_qk_norm:
            self.q_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
        self.cross_layer_mode = getattr(config, "cross_layer_mode", "residual")
        self.cross_layer_pattern = getattr(config, "cross_layer_pattern", None)
        if self.cross_layer_pattern == "dense":
            # Only dense mode uses v_norm (residual addition needs normalization)
            self.v_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # TODO (joao): remove in v4.46 (RoPE is computed in the model, not in the decoder layers)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)
        self._trace_probe_enabled = False
        self._trace_last_probe = None

    @staticmethod
    def _build_extra_attention_masks(base_mask: torch.Tensor, src_lens: List[int]) -> List[torch.Tensor]:
        if not src_lens:
            return []

        base_len = base_mask.shape[-1]
        # Fast path: all source lengths equal base length.
        if all(src_len == base_len for src_len in src_lens):
            return [base_mask] * len(src_lens)

        # Reuse generated slices/pads for repeated src_len values.
        mask_by_len = {}
        extra_masks = []
        for src_len in src_lens:
            if src_len not in mask_by_len:
                if src_len < base_len:
                    mask_by_len[src_len] = base_mask[:, :, :, -src_len:]
                elif src_len == base_len:
                    mask_by_len[src_len] = base_mask
                else:
                    pad = torch.zeros(
                        (base_mask.shape[0], 1, base_mask.shape[2], src_len - base_len),
                        dtype=base_mask.dtype,
                        device=base_mask.device,
                    )
                    mask_by_len[src_len] = torch.cat([pad, base_mask], dim=-1)
            extra_masks.append(mask_by_len[src_len])
        return extra_masks

    def _merge_cross_layer_kv(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        cross_layer_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        local_layer_kv = (key_states, value_states)
        if not cross_layer_kv:
            return key_states, value_states, attention_mask, local_layer_kv

        extra_k = []
        extra_v = []
        valid_cross_kv = []
        for kv in cross_layer_kv:
            if kv is None or len(kv) != 2:
                continue
            src_k, src_v = kv
            if src_k is None or src_v is None:
                continue
            extra_k.append(src_k)
            extra_v.append(src_v)
            valid_cross_kv.append((src_k, src_v))

        if not valid_cross_kv:
            return key_states, value_states, attention_mask, local_layer_kv

        key_states = torch.cat([key_states] + extra_k, dim=2)
        value_states = torch.cat([value_states] + extra_v, dim=2)

        if attention_mask is not None:
            # === Optimization: Check if mask is already extended ===
            if attention_mask.shape[-1] >= key_states.shape[-2]:
                return key_states, value_states, attention_mask[..., :key_states.shape[-2]], local_layer_kv
            # =====================================================

            base_mask = attention_mask[:, :, :, : local_layer_kv[0].shape[2]]
            src_lens = [src_k.shape[2] for src_k, _ in valid_cross_kv]
            extra_masks = self._build_extra_attention_masks(base_mask=base_mask, src_lens=src_lens)
            if extra_masks:
                attention_mask = torch.cat([attention_mask] + extra_masks, dim=-1)

        return key_states, value_states, attention_mask, local_layer_kv

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        cross_layer_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        return_layer_kv: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        _reset_attention_trace(self)

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        query_states = self.q_norm(query_states)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        rope_phase = None
        if getattr(self, "_trace_probe_enabled", False) and position_ids is not None:
            inv_freq_expanded = self.rotary_emb.inv_freq[None, :, None].float().expand(
                position_ids.shape[0], -1, 1
            )
            position_ids_expanded = position_ids[:, None, :].float()
            device_type = value_states.device.type
            device_type = (
                device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
            )
            with torch.autocast(device_type=device_type, enabled=False):
                freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
                rope_phase = torch.cat((freqs, freqs), dim=-1)
        if rope_phase is None and getattr(self, "_trace_probe_enabled", False):
            rope_phase = getattr(self, "_trace_shared_rope_phase", None)
        _capture_rope_phase_trace(self, cos, sin, phase=rope_phase)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        _capture_attention_trace(self, "query_post_rope", query_states)
        _capture_attention_trace(self, "key_post_rope", key_states)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Default layer_kv_for_reuse: 2-tuple for most modes, 3-tuple for depth-mix auxiliary layout
        layer_kv_for_reuse = (key_states, value_states)
        kv_expanded = False

        if cross_layer_kv:
            # Handle pre-stacked depth buffer format
            valid_cross = []
            for kv in cross_layer_kv:
                if kv is None or len(kv) < 2:
                    continue
                if kv[0] is None or kv[1] is None:
                    continue
                valid_cross.append((kv[0], kv[1]))  # always (K, V) for standard branches
            # For depth-mix auxiliary layout, also collect depth_k (3rd element) separately
            valid_cross_depth_k = []
            if valid_cross:
                self_k = repeat_kv(key_states, self.num_key_value_groups)
                self_v = repeat_kv(value_states, self.num_key_value_groups)

                if self.cross_layer_mode == "cross_attn":
                    # ============================================================
                    # Cross-Attention: multi-segment flash-attn with LSE merge.
                    # Each segment (cross layers + self) gets independent flash-attn,
                    # merged via log-sum-exp for exact joint-softmax.
                    # Falls back to concat+matmul if flash-attn unavailable.
                    # ============================================================
                    q_fa = query_states.transpose(1, 2).contiguous()  # [B, T, H, d]
                    kv_pairs = []
                    for k_src, v_src in valid_cross:
                        k_fa = repeat_kv(k_src, self.num_key_value_groups)
                        k_fa = self.k_norm(k_fa).transpose(1, 2).contiguous()
                        v_fa = repeat_kv(v_src, self.num_key_value_groups).transpose(1, 2).contiguous()
                        kv_pairs.append((k_fa, v_fa))
                    self_k_normed = self.k_norm(self_k).transpose(1, 2).contiguous()
                    self_v_fa = self_v.transpose(1, 2).contiguous()
                    kv_pairs.append((self_k_normed, self_v_fa))

                    try:
                        attn_output = multi_flash_attn(
                            q_fa, kv_pairs,
                            dropout_p=self.attention_dropout if self.training else 0.0,
                            softmax_scale=None, causal=True,
                        )
                        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
                        attn_output = self.o_proj(attn_output)
                        if return_layer_kv:
                            return attn_output, None, past_key_value, layer_kv_for_reuse
                        return attn_output, None, past_key_value

                    except Exception:
                        # Fallback: concat + manual attention
                        extra_k = [repeat_kv(k, self.num_key_value_groups) for k, _ in valid_cross]
                        extra_v = [repeat_kv(v, self.num_key_value_groups) for _, v in valid_cross]
                        key_states = self.k_norm(torch.cat(extra_k + [self_k], dim=2))
                        value_states = torch.cat(extra_v + [self_v], dim=2)
                        if attention_mask is not None:
                            base_mask = attention_mask[:, :, :, :q_len]
                            attention_mask = torch.cat([base_mask] * (len(valid_cross) + 1), dim=-1)
                        kv_expanded = True
                        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
                        if attention_mask is not None:
                            attn_weights = attn_weights + attention_mask[:, :, :, :key_states.shape[-2]]
                        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
                        attn_output = torch.matmul(attn_weights, value_states)
                        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
                        attn_output = self.o_proj(attn_output)
                        if not output_attentions:
                            attn_weights = None
                        if return_layer_kv:
                            return attn_output, attn_weights, past_key_value, layer_kv_for_reuse
                        return attn_output, attn_weights, past_key_value

                elif self.cross_layer_mode == "depth_softmax" and len(valid_cross) >= 1:
                    # Depth-wise Softmax Attention (eager path)
                    grp = self.num_key_value_groups
                    scale = math.sqrt(self.head_dim)

                    # k_norm self_k first, reuse for both depth scoring and self attention
                    self_k_normed = self.k_norm(self_k)

                    # Depth scoring: Q(q_norm) · K(k_norm)
                    all_k = [repeat_kv(self.k_norm(k), grp) for k, _ in valid_cross] + [repeat_kv(self_k_normed, grp)] if grp > 1 \
                            else [self.k_norm(k) for k, _ in valid_cross] + [self_k_normed]
                    depth_scores = torch.stack(
                        [(query_states * k).sum(-1) / scale for k in all_k], dim=-1
                    )
                    depth_weights = torch.nn.functional.softmax(
                        depth_scores, dim=-1, dtype=torch.float32
                    ).to(query_states.dtype)

                    # Weighted V via stack+einsum
                    all_v = [repeat_kv(v, grp) for _, v in valid_cross] + [self_v] if grp > 1 \
                            else [v for _, v in valid_cross] + [self_v]
                    stacked_v = torch.stack(all_v, dim=0)
                    value_states = torch.einsum("bhtn,nbhtd->bhtd", depth_weights, stacked_v)

                    layer_kv_for_reuse = (self_k, value_states)  # store depth-mixed V
                    key_states = self_k_normed  # already k_normed, no double norm
                    if hasattr(self, "v_norm"):
                        value_states = self.v_norm(value_states)
                elif self.cross_layer_mode == "depth_softmax_head0" and len(valid_cross) >= 1:
                    # Head-0 depth softmax (eager path): borrow head 0 for depth scoring,
                    # heads 1+ do normal self-attention. Zero extra parameters.
                    grp = self.num_key_value_groups
                    scale = math.sqrt(self.head_dim)

                    depth_q = query_states[:, 0:1, :, :]
                    self_k_normed = self.k_norm(self_k)
                    depth_self_k = self_k_normed[:, 0:1, :, :]

                    all_dk = [self.k_norm(repeat_kv(k, grp))[:, 0:1, :, :] for k, _ in valid_cross] + [depth_self_k]
                    all_v = [repeat_kv(v, grp) for _, v in valid_cross] + [self_v] if grp > 1 \
                            else [v for _, v in valid_cross] + [self_v]

                    depth_scores = torch.stack(
                        [(depth_q * dk).sum(-1) / scale for dk in all_dk], dim=-1
                    )
                    depth_weights = torch.nn.functional.softmax(
                        depth_scores, dim=-1, dtype=torch.float32
                    ).to(query_states.dtype)

                    stacked_v = torch.stack(all_v, dim=0)
                    mixed_v = torch.einsum("bhtn,nbhtd->bhtd", depth_weights.expand(-1, self.num_heads, -1, -1), stacked_v)

                    layer_kv_for_reuse = (self_k, mixed_v)  # store depth-mixed V

                    # Head 0: directly output depth-mixed V
                    head0_out = mixed_v[:, 0:1, :, :]

                    # Heads 1+: standard attention with depth-mixed V
                    attn_q = query_states[:, 1:, :, :]
                    attn_k = self_k_normed[:, 1:, :, :]
                    attn_v = mixed_v[:, 1:, :, :]

                    attn_weights_partial = torch.matmul(attn_q, attn_k.transpose(2, 3)) / scale
                    if attention_mask is not None:
                        causal_mask_partial = attention_mask[:, :, :, : attn_k.shape[-2]]
                        if causal_mask_partial.shape[1] == 1:
                            causal_mask_partial = causal_mask_partial.expand(-1, attn_q.shape[1], -1, -1)
                        elif causal_mask_partial.shape[1] > attn_q.shape[1]:
                            causal_mask_partial = causal_mask_partial[:, 1:, :, :]
                        attn_weights_partial = attn_weights_partial + causal_mask_partial
                    attn_weights_partial = nn.functional.softmax(attn_weights_partial, dim=-1, dtype=torch.float32).to(attn_q.dtype)
                    attn_weights_partial = nn.functional.dropout(attn_weights_partial, p=self.attention_dropout if self.training else 0.0, training=self.training)
                    attn_out = torch.matmul(attn_weights_partial, attn_v)

                    attn_output = torch.cat([head0_out, attn_out], dim=1)
                    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
                    attn_output = self.o_proj(attn_output)

                    if not output_attentions:
                        attn_weights_partial = None
                    if return_layer_kv:
                        return attn_output, attn_weights_partial, past_key_value, layer_kv_for_reuse
                    return attn_output, attn_weights_partial, past_key_value

                elif self.cross_layer_mode == "depth_softmax_head0_v2" and len(valid_cross) >= 1:
                    # Head-0 depth softmax v2 (eager path): heads 1+ use ORIGINAL self_v
                    # (not mixed). Zero extra parameters.
                    grp = self.num_key_value_groups
                    scale = math.sqrt(self.head_dim)

                    depth_q = query_states[:, 0:1, :, :]
                    self_k_normed = self.k_norm(self_k)
                    depth_self_k = self_k_normed[:, 0:1, :, :]

                    all_dk = [self.k_norm(repeat_kv(k, grp))[:, 0:1, :, :] for k, _ in valid_cross] + [depth_self_k]
                    all_v_head0 = [repeat_kv(v, grp)[:, 0:1, :, :] for _, v in valid_cross] + [self_v[:, 0:1, :, :]] if grp > 1 \
                                  else [v[:, 0:1, :, :] for _, v in valid_cross] + [self_v[:, 0:1, :, :]]

                    depth_scores = torch.stack(
                        [(depth_q * dk).sum(-1) / scale for dk in all_dk], dim=-1
                    )
                    depth_weights = torch.nn.functional.softmax(
                        depth_scores, dim=-1, dtype=torch.float32
                    ).to(query_states.dtype)

                    stacked_v_head0 = torch.stack(all_v_head0, dim=0)  # [N, B, 1, T, d]
                    head0_out = torch.einsum("bhtn,nbhtd->bhtd", depth_weights, stacked_v_head0)  # [B, 1, T, d]

                    # Compute full depth-mixed V (all heads) for downstream reuse storage.
                    # Mirrors head0 v1 full V mix semantics: weights are 1-head but expand over H.
                    all_v_full = [repeat_kv(v, grp) for _, v in valid_cross] + [self_v] if grp > 1 \
                                 else [v for _, v in valid_cross] + [self_v]
                    stacked_v_full = torch.stack(all_v_full, dim=0)  # [N, B, H, T, d]
                    mixed_v = torch.einsum(
                        "bhtn,nbhtd->bhtd",
                        depth_weights.expand(-1, self.num_heads, -1, -1),
                        stacked_v_full,
                    )  # [B, H, T, d]

                    layer_kv_for_reuse = (self_k, mixed_v)  # store depth-mixed V (full H)

                    # Heads 1+: standard attention with ORIGINAL self_v
                    attn_q = query_states[:, 1:, :, :]
                    attn_k = self_k_normed[:, 1:, :, :]
                    attn_v = self_v[:, 1:, :, :]

                    attn_weights_partial = torch.matmul(attn_q, attn_k.transpose(2, 3)) / scale
                    if attention_mask is not None:
                        causal_mask_partial = attention_mask[:, :, :, : attn_k.shape[-2]]
                        if causal_mask_partial.shape[1] == 1:
                            causal_mask_partial = causal_mask_partial.expand(-1, attn_q.shape[1], -1, -1)
                        elif causal_mask_partial.shape[1] > attn_q.shape[1]:
                            causal_mask_partial = causal_mask_partial[:, 1:, :, :]
                        attn_weights_partial = attn_weights_partial + causal_mask_partial
                    attn_weights_partial = nn.functional.softmax(attn_weights_partial, dim=-1, dtype=torch.float32).to(attn_q.dtype)
                    attn_weights_partial = nn.functional.dropout(attn_weights_partial, p=self.attention_dropout if self.training else 0.0, training=self.training)
                    attn_out = torch.matmul(attn_weights_partial, attn_v)

                    attn_output = torch.cat([head0_out, attn_out], dim=1)
                    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
                    attn_output = self.o_proj(attn_output)

                    if not output_attentions:
                        attn_weights_partial = None
                    if return_layer_kv:
                        return attn_output, attn_weights_partial, past_key_value, layer_kv_for_reuse
                    return attn_output, attn_weights_partial, past_key_value

                elif self.cross_layer_mode == "v0_mix" and len(valid_cross) >= 1:
                    # Simple V0 mixing: V = 0.5 * V_layer0 + 0.5 * V_self
                    cross_v = repeat_kv(valid_cross[0][1], self.num_key_value_groups)
                    value_states = 0.5 * cross_v + 0.5 * self_v
                    key_states = self_k
                    layer_kv_for_reuse = (key_states, value_states)  # store depth-mixed V
                    key_states = self.k_norm(key_states)
                    kv_expanded = True
                else:
                    if len(valid_cross) == 1:
                        cross_k = repeat_kv(valid_cross[0][0], self.num_key_value_groups)
                        cross_v = repeat_kv(valid_cross[0][1], self.num_key_value_groups)
                    else:
                        cross_k = torch.stack(
                            [repeat_kv(k, self.num_key_value_groups) for k, _ in valid_cross]
                        ).mean(0)
                        cross_v = torch.stack(
                            [repeat_kv(v, self.num_key_value_groups) for _, v in valid_cross]
                        ).mean(0)
                    key_states = self_k + cross_k
                    value_states = self_v + cross_v
                    layer_kv_for_reuse = (key_states, value_states)
                    key_states = self.k_norm(key_states)
                    if hasattr(self, "v_norm"):
                        value_states = self.v_norm(value_states)
                kv_expanded = True

        if not kv_expanded:
            # Use separate variables to keep raw key_states/value_states intact in layer_kv_for_reuse
            key_states_attn = self.k_norm(key_states)
            value_states_attn = value_states
            if hasattr(self, "v_norm"):
                value_states_attn = self.v_norm(value_states_attn)
            key_states = repeat_kv(key_states_attn, self.num_key_value_groups)
            value_states = repeat_kv(value_states_attn, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        if return_layer_kv:
            return attn_output, attn_weights, past_key_value, layer_kv_for_reuse
        return attn_output, attn_weights, past_key_value


class LlamaFlashAttention2(LlamaAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        cross_layer_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        return_layer_kv: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if cross_layer_kv:
            raise ValueError("cross_layer_kv 需要 eager/sdpa 注意力实现，flash_attention_2 不支持该模式。")
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        rope_phase = getattr(self, "_trace_shared_rope_phase", None)
        _capture_rope_phase_trace(self, cos, sin, phase=rope_phase)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        if return_layer_kv:
            return attn_output, attn_weights, past_key_value, None
        return attn_output, attn_weights, past_key_value


class LlamaSdpaAttention(LlamaAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cross_layer_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        return_layer_kv: bool = False,
        **kwargs,
    ):
        if output_attentions:
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` "
                "does not support `output_attentions=True`. Falling back to the manual attention implementation."
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                cross_layer_kv=cross_layer_kv,
                return_layer_kv=return_layer_kv,
            )

        bsz, q_len, _ = hidden_states.size()
        _reset_attention_trace(self)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)                 # [B,Hq,L,d]
        key_states   = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)         # [B,Hkv,L,d]
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)       # [B,Hkv,L,d]

        query_states = self.q_norm(query_states)
        # NOTE: k_norm is NOT applied here (before RoPE). Instead, each cross-layer
        # mode branch applies k_norm to self_k/mixed_k individually. This matches
        # the design where layer_kv_for_reuse stores K without k_norm, and k_norm
        # is applied just before the attention computation in each branch.

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        rope_phase = getattr(self, "_trace_shared_rope_phase", None)
        _capture_rope_phase_trace(self, cos, sin, phase=rope_phase)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        _capture_attention_trace(self, "query_post_rope", query_states)
        _capture_attention_trace(self, "key_post_rope", key_states)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Default layer_kv_for_reuse: 2-tuple for most modes, 3-tuple for depth-mix auxiliary layout
        # OPTIMIZATION: for depth_softmax variants, pre-normalize K here so downstream
        # layers can skip the expensive batched k_norm (~800ms/step savings).
        # Compute k_norm(key_states) ONCE here and reuse everywhere (branch + SDPA path).
        # NOTE: depth_softmax_head0_v2 is intentionally excluded — its branch still uses
        # the list-format valid_cross path and re-applies self.k_norm per layer.
        # OPTIMIZATION for 1head: main K (kv[0]) is never consumed by downstream 1head
        # layers (they only read kv[1]=V and kv[2]=depth_k). Store None for kv[0] to
        # skip the 11× H-head tensor reference and let the model-level incremental
        # stack skip building _depth_k_stacked entirely.
        # OPTIMIZATION for head0_v3: downstream only reads head 0 of K and V, so we
        # store ONLY head 0 slices (pre-normed K), giving 16× less cross-layer KV
        # memory + bandwidth vs head0_v2 which still stores full-H K/V.
        _is_depth_softmax_family = self.cross_layer_mode in (
            "depth_softmax", "depth_softmax_head0",
            "depth_softmax_head0_v3", "depth_softmax_head0_v4",        )
        _kn_precomputed = None  # pre-computed self.k_norm(key_states), only for grp==1 depth_softmax
        _depth_k_precomputed = None  # pre-computed self.depth_k_norm(depth_k_proj(x)), only for 1head
        if _is_depth_softmax_family and self.num_key_value_groups == 1:
            _kn_precomputed = self.k_norm(key_states)
            if self.cross_layer_mode in ("depth_softmax_head0_v3", "depth_softmax_head0_v4"):
                # Store ONLY head 0 of pre-normed K and head 0 of V. Downstream
                # head0_v3/v4 layers read just these 1-head slices.
                layer_kv_for_reuse = (
                    _kn_precomputed[:, 0:1, :, :].contiguous(),
                    value_states[:, 0:1, :, :].contiguous(),
                )
            else:
                layer_kv_for_reuse = (_kn_precomputed, value_states)
        else:
            layer_kv_for_reuse = (key_states, value_states)
        dropout_p = self.attention_dropout if self.training else 0.0

        # =========================
        # (A) cross-layer 路径
        # =========================
        # Detect stacked tuple format. Three possible layouts:
        #   regular:  (stacked_k,  stacked_v) or (stacked_k, stacked_v, stacked_dk)
        #   1head:    (None,       stacked_v, stacked_dk)  — main K omitted
        # vs list-of-tuples format: [(k1, v1), (k2, v2), ...]
        _stacked_cross_k = None
        _stacked_cross_v = None
        _stacked_cross_dk = None
        if cross_layer_kv is not None and isinstance(cross_layer_kv, tuple) \
                and len(cross_layer_kv) >= 2:
            # 1head layout: (None, stacked_v, stacked_dk)
            if cross_layer_kv[0] is None and len(cross_layer_kv) >= 3 \
                    and torch.is_tensor(cross_layer_kv[1]) and cross_layer_kv[1].ndim == 5:
                _stacked_cross_v = cross_layer_kv[1]
                if torch.is_tensor(cross_layer_kv[2]):
                    _stacked_cross_dk = cross_layer_kv[2]
            # Regular layout: (stacked_k, stacked_v[, stacked_dk])
            elif torch.is_tensor(cross_layer_kv[0]) and cross_layer_kv[0].ndim == 5:
                _stacked_cross_k = cross_layer_kv[0]
                _stacked_cross_v = cross_layer_kv[1]
                if len(cross_layer_kv) >= 3 and torch.is_tensor(cross_layer_kv[2]):
                    _stacked_cross_dk = cross_layer_kv[2]

        if cross_layer_kv:
            # For list format: build valid_cross list as before
            # For stacked format: mark valid_cross as non-empty sentinel
            valid_cross = []
            _is_1head = False
            if _stacked_cross_k is not None:
                # Use a single-element list as "non-empty" sentinel; downstream branches
                # check _stacked_cross_k first
                valid_cross = [(None, None)] * _stacked_cross_k.shape[0]
            elif _stacked_cross_v is not None:
                # stacked format: main K is None, use V's N for the sentinel
                valid_cross = [(None, None)] * _stacked_cross_v.shape[0]
            else:
                for kv in cross_layer_kv:
                    if kv is None or len(kv) < 2:
                        continue
                    if kv[1] is None:
                        continue
                    #                     if kv[0] is None and not _is_1head:
                        continue
                    valid_cross.append((kv[0], kv[1]))
            valid_cross_depth_k = []
            if _is_1head and _stacked_cross_dk is None:
                for kv in cross_layer_kv:
                    if kv is None or len(kv) < 2 or kv[1] is None:
                        continue
                    valid_cross_depth_k.append(kv[2] if len(kv) >= 3 else None)

            if valid_cross:
                self_k = repeat_kv(key_states, self.num_key_value_groups)
                self_v = repeat_kv(value_states, self.num_key_value_groups)

                if self.cross_layer_mode == "cross_attn":
                    # Cross-Attention (SDPA path): use multi_flash_attn with LSE merge
                    # Each segment (cross layers + self) gets independent flash-attn,
                    # then merged via log-sum-exp for exact joint-softmax gradients.
                    # flash-attn expects [B, L, H, d] layout
                    q_fa = query_states.transpose(1, 2).contiguous()  # [B, T, H, d]

                    kv_pairs = []
                    for k_src, v_src in valid_cross:
                        k_fa = repeat_kv(k_src, self.num_key_value_groups).transpose(1, 2).contiguous()
                        v_fa = repeat_kv(v_src, self.num_key_value_groups).transpose(1, 2).contiguous()
                        kv_pairs.append((self.k_norm(k_fa.transpose(1, 2)).transpose(1, 2).contiguous(), v_fa))
                    # Self segment as last
                    self_k_normed = self.k_norm(self_k).transpose(1, 2).contiguous()
                    self_v_fa = self_v.transpose(1, 2).contiguous()
                    kv_pairs.append((self_k_normed, self_v_fa))

                    try:
                        attn_output = multi_flash_attn(
                            q_fa, kv_pairs,
                            dropout_p=dropout_p, softmax_scale=None, causal=True,
                        )  # [B, T, H, d]
                        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
                        attn_output = self.o_proj(attn_output)
                        if return_layer_kv:
                            return attn_output, None, past_key_value, layer_kv_for_reuse
                        return attn_output, None, past_key_value

                    except Exception:
                        # Fallback: concat KV + extended causal mask (for non-flash-attn envs)
                        extra_k = [repeat_kv(k, self.num_key_value_groups) for k, _ in valid_cross]
                        extra_v = [repeat_kv(v, self.num_key_value_groups) for _, v in valid_cross]
                        mixed_k = self.k_norm(torch.cat(extra_k + [self_k], dim=2))
                        mixed_v = torch.cat(extra_v + [self_v], dim=2)
                        if attention_mask is not None:
                            base_mask = attention_mask[:, :, :, :q_len]
                            causal_mask = torch.cat([base_mask] * (len(valid_cross) + 1), dim=-1)
                        else:
                            causal_mask = None
                        if query_states.device.type == "cuda":
                            query_states = query_states.contiguous()
                            mixed_k = mixed_k.contiguous()
                            mixed_v = mixed_v.contiguous()
                        is_causal = True if causal_mask is None and q_len > 1 else False
                        attn_output = torch.nn.functional.scaled_dot_product_attention(
                            query_states, mixed_k, mixed_v,
                            attn_mask=causal_mask, dropout_p=dropout_p, is_causal=is_causal,
                        )
                        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
                        attn_output = self.o_proj(attn_output)
                        if return_layer_kv:
                            return attn_output, None, past_key_value, layer_kv_for_reuse
                        return attn_output, None, past_key_value

                elif self.cross_layer_mode == "depth_softmax" and len(valid_cross) >= 1:
                    # Depth-wise Softmax Attention (SDPA path) — stores pre-normed K
                    grp = self.num_key_value_groups
                    scale = math.sqrt(self.head_dim)

                    # Reuse pre-computed self.k_norm(key_states) if available (grp=1 case)
                    if _kn_precomputed is not None:
                        self_k_normed = _kn_precomputed
                    else:
                        self_k_normed = self.k_norm(self_k)

                    # --- Cross K is already pre-normed from source layers ---
                    # Use pre-stacked buffer if available, else stack from list
                    if _stacked_cross_k is not None:
                        cross_k_normed = _stacked_cross_k
                        cross_v = _stacked_cross_v
                    else:
                        cross_k_normed = torch.stack([k for k, _ in valid_cross], dim=0)
                        cross_v = torch.stack([v for _, v in valid_cross], dim=0)

                    if grp > 1:
                        ck_shape = cross_k_normed.shape
                        cross_k_normed = repeat_kv(cross_k_normed.reshape(-1, *ck_shape[2:]), grp).reshape(
                            ck_shape[0], bsz, self.num_heads, q_len, self.head_dim)
                        self_k_normed_exp = repeat_kv(self_k_normed, grp)
                        cross_v = repeat_kv(cross_v.reshape(-1, *ck_shape[2:]), grp).reshape(
                            ck_shape[0], bsz, self.num_heads, q_len, self.head_dim)
                    else:
                        self_k_normed_exp = self_k_normed

                    _capture_attention_trace(self, "key_for_attention", self_k_normed_exp)

                    # --- Depth mixing: dispatch to split/stream impl, optionally compiled ---
                    _mix_fn = _get_compiled_depth_mix()
                    mixed_v = _mix_fn(
                        query_states, cross_k_normed, cross_v,
                        self_k_normed_exp, self_v, scale,
                    )
                    _capture_attention_trace(self, "value_for_attention", mixed_v)

                    # Store pre-normed K (key optimization!) + depth-mixed V
                    layer_kv_for_reuse = (self_k_normed, mixed_v)
                    mixed_k = self_k_normed  # already k_normed
                    if hasattr(self, "v_norm"):
                        mixed_v = self.v_norm(mixed_v)
                elif self.cross_layer_mode == "depth_softmax_head0" and len(valid_cross) >= 1:
                    # Head-0 depth softmax (SDPA path) — batched for speed
                    grp = self.num_key_value_groups
                    scale = math.sqrt(self.head_dim)

                    depth_q = query_states[:, 0:1, :, :]  # [B, 1, T, d] (already q_normed)
                    if _kn_precomputed is not None:
                        self_k_normed = _kn_precomputed
                    else:
                        self_k_normed = self.k_norm(self_k)
                    depth_self_k = self_k_normed[:, 0:1, :, :]  # head 0 of self K

                    # Use pre-stacked buffer if available (cross K already pre-normed)
                    if _stacked_cross_k is not None:
                        cross_dk = _stacked_cross_k[:, :, 0:1, :, :]
                        cross_v_stacked = _stacked_cross_v
                    else:
                        if grp > 1:
                            cross_k_normed_full = torch.stack([repeat_kv(k, grp) for k, _ in valid_cross], dim=0)
                            cross_v_stacked = torch.stack([repeat_kv(v, grp) for _, v in valid_cross], dim=0)
                        else:
                            cross_k_normed_full = torch.stack([k for k, _ in valid_cross], dim=0)
                            cross_v_stacked = torch.stack([v for _, v in valid_cross], dim=0)
                        cross_dk = cross_k_normed_full[:, :, 0:1, :, :]

                    # --- Split-scoring: avoid 5D cat ---
                    cross_scores = torch.einsum("bhtd,nbhtd->bhtn", depth_q, cross_dk) / scale
                    self_score = (depth_q * depth_self_k).sum(-1, keepdim=True) / scale
                    depth_scores = torch.cat([cross_scores, self_score], dim=-1)
                    depth_weights = torch.nn.functional.softmax(
                        depth_scores, dim=-1, dtype=torch.float32
                    ).to(query_states.dtype)  # [B, 1, T, N+1]

                    # --- Split-weighted V ---
                    # Tier A opt: drop stride-0 .expand() in favor of squeezed [B,T,N]
                    # weights. Mathematically equivalent (see 1head branch comment).
                    cross_weights_btn = depth_weights[..., :-1].squeeze(1)  # [B, T, N]
                    self_weight = depth_weights[..., -1:]                    # [B, 1, T, 1]
                    cross_mixed = torch.einsum("btn,nbhtd->bhtd", cross_weights_btn, cross_v_stacked)
                    mixed_v = cross_mixed + self_weight * self_v

                    # Store pre-normed K + depth-mixed V
                    layer_kv_for_reuse = (self_k_normed, mixed_v)

                    # Head 0: directly output depth-mixed V (no self-attention)
                    head0_out = mixed_v[:, 0:1, :, :]  # [B, 1, T, d]

                    # Heads 1+: standard self-attention with depth-mixed V
                    attn_q = query_states[:, 1:, :, :]
                    attn_k = self_k_normed[:, 1:, :, :]
                    attn_v = mixed_v[:, 1:, :, :]

                    if query_states.device.type == "cuda":
                        attn_q = attn_q.contiguous()
                        attn_k = attn_k.contiguous()
                        attn_v = attn_v.contiguous()

                    attn_out = torch.nn.functional.scaled_dot_product_attention(
                        attn_q, attn_k, attn_v,
                        attn_mask=None, dropout_p=dropout_p, is_causal=True,
                    )  # [B, H-1, T, d]

                    attn_output = torch.cat([head0_out, attn_out], dim=1)  # [B, H, T, d]
                    attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
                    attn_output = self.o_proj(attn_output)

                    if return_layer_kv:
                        return attn_output, None, past_key_value, layer_kv_for_reuse
                    return attn_output, None, past_key_value

                elif self.cross_layer_mode == "depth_softmax_head0_v2" and len(valid_cross) >= 1:
                    # Head-0 depth softmax v2 (SDPA path): head 0 outputs depth-mixed V,
                    # but heads 1+ use ORIGINAL self_v (not depth-mixed). This isolates
                    # depth routing to head 0 only. Zero extra parameters.
                    grp = self.num_key_value_groups
                    scale = math.sqrt(self.head_dim)

                    depth_q = query_states[:, 0:1, :, :]  # [B, 1, T, d] (already q_normed)
                    self_k_normed = self.k_norm(self_k)
                    depth_self_k = self_k_normed[:, 0:1, :, :]

                    all_dk = [self.k_norm(repeat_kv(k, grp))[:, 0:1, :, :] for k, _ in valid_cross] + [depth_self_k]
                    # Only head 0 needs mixed V (not all heads like v1)
                    all_v_head0 = [repeat_kv(v, grp)[:, 0:1, :, :] for _, v in valid_cross] + [self_v[:, 0:1, :, :]] if grp > 1 \
                                  else [v[:, 0:1, :, :] for _, v in valid_cross] + [self_v[:, 0:1, :, :]]

                    depth_scores = torch.stack(
                        [(depth_q * dk).sum(-1) / scale for dk in all_dk], dim=-1
                    )
                    depth_weights = torch.nn.functional.softmax(
                        depth_scores, dim=-1, dtype=torch.float32
                    ).to(query_states.dtype)  # [B, 1, T, N]

                    stacked_v_head0 = torch.stack(all_v_head0, dim=0)  # [N, B, 1, T, d]
                    head0_out = torch.einsum("bhtn,nbhtd->bhtd", depth_weights, stacked_v_head0)  # [B, 1, T, d]

                    # Compute full depth-mixed V (all heads) for downstream reuse storage.
                    # Mirrors head0 v1 full V mix semantics: weights are 1-head but expand over H.
                    all_v_full = [repeat_kv(v, grp) for _, v in valid_cross] + [self_v] if grp > 1 \
                                 else [v for _, v in valid_cross] + [self_v]
                    stacked_v_full = torch.stack(all_v_full, dim=0)  # [N, B, H, T, d]
                    mixed_v = torch.einsum(
                        "bhtn,nbhtd->bhtd",
                        depth_weights.expand(-1, self.num_heads, -1, -1),
                        stacked_v_full,
                    )  # [B, H, T, d]

                    layer_kv_for_reuse = (self_k, mixed_v)  # store depth-mixed V (full H)

                    # Heads 1+: standard self-attention with ORIGINAL self_v (not mixed)
                    attn_q = query_states[:, 1:, :, :]
                    attn_k = self_k_normed[:, 1:, :, :]
                    attn_v = self_v[:, 1:, :, :]  # original, not mixed

                    if query_states.device.type == "cuda":
                        attn_q = attn_q.contiguous()
                        attn_k = attn_k.contiguous()
                        attn_v = attn_v.contiguous()

                    attn_out = torch.nn.functional.scaled_dot_product_attention(
                        attn_q, attn_k, attn_v,
                        attn_mask=None, dropout_p=dropout_p, is_causal=True,
                    )  # [B, H-1, T, d]

                    attn_output = torch.cat([head0_out, attn_out], dim=1)
                    attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
                    attn_output = self.o_proj(attn_output)

                    if return_layer_kv:
                        return attn_output, None, past_key_value, layer_kv_for_reuse
                    return attn_output, None, past_key_value

                elif self.cross_layer_mode == "depth_softmax_head0_v3" and len(valid_cross) >= 1:
                    # Head-0 depth softmax v3: same mixing as v2 but with 1-head cross
                    # KV storage (16× memory/bandwidth reduction). Numerically equivalent
                    # to v2 since only head 0 is ever consumed by downstream layers.
                    grp = self.num_key_value_groups
                    scale = math.sqrt(self.head_dim)

                    depth_q = query_states[:, 0:1, :, :]  # [B, 1, T, d] (already q_normed)
                    if _kn_precomputed is not None:
                        self_k_normed = _kn_precomputed
                    else:
                        self_k_normed = self.k_norm(self_k)
                    depth_self_k_h0 = self_k_normed[:, 0:1, :, :]  # [B, 1, T, d]
                    self_v_h0 = self_v[:, 0:1, :, :]               # [B, 1, T, d]

                    # Cross K/V: already head-0 slices from source layers.
                    if _stacked_cross_k is not None:
                        cross_dk_h0 = _stacked_cross_k
                        cross_v_h0 = _stacked_cross_v
                    else:
                        if grp > 1:
                            cross_dk_h0 = torch.stack(
                                [repeat_kv(k, grp)[:, 0:1, :, :] for k, _ in valid_cross], dim=0)
                            cross_v_h0 = torch.stack(
                                [repeat_kv(v, grp)[:, 0:1, :, :] for _, v in valid_cross], dim=0)
                        else:
                            cross_dk_h0 = torch.stack([k for k, _ in valid_cross], dim=0)
                            cross_v_h0 = torch.stack([v for _, v in valid_cross], dim=0)

                    # Depth scoring (1-head)
                    cross_scores = torch.einsum("bhtd,nbhtd->bhtn", depth_q, cross_dk_h0) / scale
                    self_score = (depth_q * depth_self_k_h0).sum(-1, keepdim=True) / scale
                    depth_scores = torch.cat([cross_scores, self_score], dim=-1)

                    depth_weights = torch.nn.functional.softmax(
                        depth_scores, dim=-1, dtype=torch.float32
                    ).to(query_states.dtype)

                    # V mix: head 0 only (cross_v_h0 is already 1-head) — should be 16×
                    # cheaper than 1head's V mix (which writes all H heads).
                    cross_weights_btn = depth_weights[..., :-1].squeeze(1)  # [B, T, N]
                    self_weight = depth_weights[..., -1:]                    # [B, 1, T, 1]
                    cross_mixed_h0 = torch.einsum("btn,nbhtd->bhtd", cross_weights_btn, cross_v_h0)
                    head0_out = cross_mixed_h0 + self_weight * self_v_h0     # [B, 1, T, d]

                    # Heads 1+: slice Q/K/V and make them contiguous for SDPA.
                    attn_q = query_states[:, 1:, :, :]
                    attn_k = self_k_normed[:, 1:, :, :]
                    attn_v = self_v[:, 1:, :, :]
                    if query_states.device.type == "cuda":
                        attn_q = attn_q.contiguous()
                        attn_k = attn_k.contiguous()
                        attn_v = attn_v.contiguous()

                    attn_out = torch.nn.functional.scaled_dot_product_attention(
                        attn_q, attn_k, attn_v,
                        attn_mask=None, dropout_p=dropout_p, is_causal=True,
                    )  # [B, H-1, T, d]

                    attn_output = torch.cat([head0_out, attn_out], dim=1)  # [B, H, T, d]
                    attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
                    attn_output = self.o_proj(attn_output)

                    # Storage: head-0 slices (set in default too; reaffirm for return).
                    layer_kv_for_reuse = (
                        depth_self_k_h0.contiguous(),
                        self_v_h0.contiguous(),
                    )

                    if return_layer_kv:
                        return attn_output, None, past_key_value, layer_kv_for_reuse
                    return attn_output, None, past_key_value

                elif self.cross_layer_mode == "depth_softmax_head0_v4" and len(valid_cross) >= 1:
                    # Head-0 depth softmax v4: 1-head cross KV storage, 1-head V mix,
                    # then cat head0_mixed_v with raw self_v[:, 1:, :, :] into a full
                    # mixed_v and fall through to the shared full-H SDPA path.
                    grp = self.num_key_value_groups
                    scale = math.sqrt(self.head_dim)

                    depth_q = query_states[:, 0:1, :, :]  # [B, 1, T, d]
                    if _kn_precomputed is not None:
                        self_k_normed = _kn_precomputed
                    else:
                        self_k_normed = self.k_norm(self_k)
                    depth_self_k_h0 = self_k_normed[:, 0:1, :, :]  # [B, 1, T, d]
                    self_v_h0 = self_v[:, 0:1, :, :]               # [B, 1, T, d]

                    if _stacked_cross_k is not None:
                        cross_dk_h0 = _stacked_cross_k   # [N, B, 1, T, d]
                        cross_v_h0 = _stacked_cross_v    # [N, B, 1, T, d]
                    else:
                        if grp > 1:
                            cross_dk_h0 = torch.stack(
                                [repeat_kv(k, grp)[:, 0:1, :, :] for k, _ in valid_cross], dim=0)
                            cross_v_h0 = torch.stack(
                                [repeat_kv(v, grp)[:, 0:1, :, :] for _, v in valid_cross], dim=0)
                        else:
                            cross_dk_h0 = torch.stack([k for k, _ in valid_cross], dim=0)
                            cross_v_h0 = torch.stack([v for _, v in valid_cross], dim=0)

                    # Depth scoring + softmax (all 1-head)
                    cross_scores = torch.einsum("bhtd,nbhtd->bhtn", depth_q, cross_dk_h0) / scale
                    self_score = (depth_q * depth_self_k_h0).sum(-1, keepdim=True) / scale
                    depth_scores = torch.cat([cross_scores, self_score], dim=-1)

                    depth_weights = torch.nn.functional.softmax(
                        depth_scores, dim=-1, dtype=torch.float32
                    ).to(query_states.dtype)  # [B, 1, T, N+1]

                    # V mix for head 0 (1-head)
                    cross_weights_btn = depth_weights[..., :-1].squeeze(1)  # [B, T, N]
                    self_weight = depth_weights[..., -1:]                    # [B, 1, T, 1]
                    cross_mixed_h0 = torch.einsum("btn,nbhtd->bhtd", cross_weights_btn, cross_v_h0)
                    head0_mixed_v = cross_mixed_h0 + self_weight * self_v_h0  # [B, 1, T, d]

                    # Build full V with head 0 replaced by mixed. Use torch.cat for a
                    # single allocation; avoids the slice+contig copies head0_v3 paid.
                    mixed_v = torch.cat([head0_mixed_v, self_v[:, 1:, :, :]], dim=1)  # [B, H, T, d]
                    mixed_k = self_k_normed

                    layer_kv_for_reuse = (
                        depth_self_k_h0.contiguous(),
                        self_v_h0.contiguous(),
                    )
                    if hasattr(self, "v_norm"):
                        mixed_v = self.v_norm(mixed_v)

                elif self.cross_layer_mode == "v0_mix" and len(valid_cross) >= 1:
                    # Simple V0 mixing (SDPA path): V = 0.5 * V_layer0 + 0.5 * V_self
                    cross_v = repeat_kv(valid_cross[0][1], self.num_key_value_groups)
                    mixed_v = 0.5 * cross_v + 0.5 * self_v
                    mixed_k = self_k
                    layer_kv_for_reuse = (mixed_k, mixed_v)  # store depth-mixed V
                    mixed_k = self.k_norm(mixed_k)
                else:
                    if len(valid_cross) == 1:
                        cross_k = repeat_kv(valid_cross[0][0], self.num_key_value_groups)
                        cross_v = repeat_kv(valid_cross[0][1], self.num_key_value_groups)
                    else:
                        cross_k = torch.stack(
                            [repeat_kv(k, self.num_key_value_groups) for k, _ in valid_cross]
                        ).mean(0)
                        cross_v = torch.stack(
                            [repeat_kv(v, self.num_key_value_groups) for _, v in valid_cross]
                        ).mean(0)
                    mixed_k = self_k + cross_k
                    mixed_v = self_v + cross_v
                    layer_kv_for_reuse = (mixed_k, mixed_v)
                    mixed_k = self.k_norm(mixed_k)
                    if hasattr(self, "v_norm"):
                        mixed_v = self.v_norm(mixed_v)

                if query_states.device.type == "cuda":
                    query_states = query_states.contiguous()
                    mixed_k = mixed_k.contiguous()
                    mixed_v = mixed_v.contiguous()

                _capture_attention_trace(self, "key_for_attention", mixed_k)
                _capture_attention_trace(self, "value_for_attention", mixed_v)
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states, mixed_k, mixed_v,
                    attn_mask=None, dropout_p=dropout_p, is_causal=True,
                )
                _capture_attention_trace(self, "core_attn_out", attn_output)

                attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
                attn_output = self.o_proj(attn_output)

                if return_layer_kv:
                    return attn_output, None, past_key_value, layer_kv_for_reuse
                return attn_output, None, past_key_value

        # =========================
        # (B) 原始 sdpa 路径
        # =========================
        # Use separate variables to keep raw key_states/value_states intact
        # in layer_kv_for_reuse (defensive: avoid any future bug from in-place ops)
        # Reuse pre-computed k_norm if available (saves a call for depth_softmax modes)
        if _kn_precomputed is not None:
            key_states_attn = _kn_precomputed
        else:
            key_states_attn = self.k_norm(key_states)
        value_states_attn = value_states
        if hasattr(self, "v_norm"):
            value_states_attn = self.v_norm(value_states_attn)
        _capture_attention_trace(self, "key_for_attention", key_states_attn)
        _capture_attention_trace(self, "value_for_attention", value_states_attn)
        key_states = repeat_kv(key_states_attn, self.num_key_value_groups)
        value_states = repeat_kv(value_states_attn, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        is_causal = True if causal_mask is None and q_len > 1 else False

        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch via is_causal when mask is None
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )
        _capture_attention_trace(self, "core_attn_out", attn_output)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if return_layer_kv:
            return attn_output, None, past_key_value, layer_kv_for_reuse
        return attn_output, None, past_key_value



LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaSdpaAttention,
}


_BASELINE_MODE_ALIASES = {
    "denseformer": "densetransformer",
    "densetransformer": "densetransformer",
    "attnres": "attnres",
    "block_attnres": "attnres",
    "block-attnres": "attnres",
    "attention_residual": "attnres",
    "attention-residual": "attnres",
    "mhc": "mhc",
}
_BASELINE_MODE_VALUES = set(_BASELINE_MODE_ALIASES.values())
_BASELINE_MODE_NONE_VALUES = {None, "", "none", "null", False}


def _normalize_baseline_mode_value(value) -> Optional[str]:
    if value in _BASELINE_MODE_NONE_VALUES:
        return None
    value = str(value).strip().lower()
    if value in _BASELINE_MODE_NONE_VALUES:
        return None
    return _BASELINE_MODE_ALIASES.get(value, value)


def _normalize_residual_baseline(config: LlamaConfig) -> Optional[str]:
    baseline_mode = _normalize_baseline_mode_value(getattr(config, "baseline_mode", None))
    legacy_baseline = _normalize_baseline_mode_value(getattr(config, "residual_baseline", None))

    if baseline_mode is not None and legacy_baseline is not None:
        if baseline_mode != legacy_baseline:
            raise ValueError(
                f"Conflicting baseline_mode={baseline_mode!r} and residual_baseline={legacy_baseline!r}. "
                "baseline_final expects one active baseline selector."
            )

    baseline = baseline_mode if baseline_mode is not None else legacy_baseline
    if baseline is None:
        raise ValueError(
            "baseline_final requires `baseline_mode` in the model config. "
            "Use one of: denseformer, attnres, or mhc."
        )
    if baseline not in _BASELINE_MODE_VALUES:
        raise ValueError(
            f"Unsupported baseline_mode={baseline!r}. "
            f"Expected one of {sorted(_BASELINE_MODE_ALIASES)} or null."
        )
    return baseline


def _positive_config_int(config: LlamaConfig, name: str, default: int) -> int:
    value = getattr(config, name, default)
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default
    return max(1, value)


def _positive_env_int(name: str, default: int) -> int:
    value = _os.environ.get(name)
    if value in (None, ""):
        return max(1, int(default))
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default
    return max(1, value)


def _positive_config_int_any(config: LlamaConfig, names: Tuple[str, ...], default: int) -> int:
    for name in names:
        if getattr(config, name, None) not in (None, ""):
            return _positive_config_int(config, name, default)
    return max(1, int(default))


def _residual_baseline_num_streams(config: LlamaConfig, baseline: Optional[str], default: int = 4) -> int:
    if baseline == "mhc":
        for name in ("residual_baseline_num_streams", "mhc_num_streams"):
            if getattr(config, name, None) not in (None, ""):
                return _positive_config_int(config, name, default)
        return _positive_env_int("MHC_NUM_STREAMS", default)
    return _positive_config_int(config, "residual_baseline_num_streams", default)


class ResidualStreamProjector(nn.Module):
    def __init__(self, num_streams: int):
        super().__init__()
        self.num_streams = num_streams
        self.reduce_logits = nn.Parameter(torch.zeros(num_streams))

    def expand(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.unsqueeze(2).expand(-1, -1, self.num_streams, -1).contiguous()

    def reduce(self, residual_streams: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.reduce_logits.float(), dim=0).to(residual_streams.dtype)
        return torch.einsum("n,btnd->btd", weights, residual_streams)


class ResidualStreamConnector(nn.Module):
    def __init__(self, config: LlamaConfig, baseline: str):
        super().__init__()
        self.baseline = baseline
        self.num_streams = _residual_baseline_num_streams(config, baseline, 4)
        self.hidden_size = config.hidden_size
        self.sinkhorn_iters = _positive_config_int(config, "mhc_sinkhorn_iters", 20)
        self.mhc_alpha_init = float(getattr(config, "mhc_alpha_init", 0.01))

        self.pre_logits = nn.Parameter(torch.zeros(self.num_streams))
        self.post_logits = nn.Parameter(torch.zeros(self.num_streams))
        self.residual_logits = nn.Parameter(torch.empty(self.num_streams, self.num_streams))

        if baseline == "mhc":
            flattened_hidden_size = self.num_streams * self.hidden_size
            self.mhc_input_norm = LlamaRMSNorm(flattened_hidden_size, eps=config.rms_norm_eps)
            self.mhc_dynamic_projection = nn.Linear(
                flattened_hidden_size,
                self.num_streams * self.num_streams + 2 * self.num_streams,
                bias=False,
            )
            self.alpha_pre = nn.Parameter(torch.tensor(self.mhc_alpha_init, dtype=torch.float32))
            self.alpha_post = nn.Parameter(torch.tensor(self.mhc_alpha_init, dtype=torch.float32))
            self.alpha_res = nn.Parameter(torch.tensor(self.mhc_alpha_init, dtype=torch.float32))
        else:
            self.mhc_input_norm = None
            self.mhc_dynamic_projection = None
            self.register_parameter("alpha_pre", None)
            self.register_parameter("alpha_post", None)
            self.register_parameter("alpha_res", None)

        self.reset_parameters()

    @staticmethod
    def _logit_probability(prob: float) -> float:
        prob = min(max(float(prob), 1e-6), 1.0 - 1e-6)
        return math.log(prob / (1.0 - prob))

    def reset_parameters(self):
        with torch.no_grad():
            if self.baseline == "mhc":
                self.pre_logits.fill_(self._logit_probability(1.0 / self.num_streams))
                self.alpha_pre.fill_(self.mhc_alpha_init)
                self.alpha_post.fill_(self.mhc_alpha_init)
                self.alpha_res.fill_(self.mhc_alpha_init)
            else:
                self.pre_logits.zero_()
            self.post_logits.zero_()

            residual = torch.eye(self.num_streams, dtype=self.residual_logits.dtype, device=self.residual_logits.device)
            if self.num_streams > 1:
                eps = 1e-2
                residual = residual * (1.0 - eps) + (1.0 - residual) * (eps / (self.num_streams - 1))
            self.residual_logits.copy_(residual.clamp_min(1e-6).log())

    @staticmethod
    def _sinkhorn(logits: torch.Tensor, iters: int) -> torch.Tensor:
        matrix = logits.float()
        matrix = matrix - matrix.amax(dim=(-1, -2), keepdim=True)
        matrix = matrix.exp().clamp_min(1e-12)
        for _ in range(iters):
            matrix = matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            matrix = matrix / matrix.sum(dim=-2, keepdim=True).clamp_min(1e-12)
        return matrix

    def _mhc_maps(
        self, residual_streams: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_streams = residual_streams.reshape(
            *residual_streams.shape[:2],
            self.num_streams * self.hidden_size,
        )
        dynamic_logits = self.mhc_dynamic_projection(self.mhc_input_norm(flat_streams)).float()
        pre_dynamic, post_dynamic, residual_dynamic = torch.split(
            dynamic_logits,
            [self.num_streams, self.num_streams, self.num_streams * self.num_streams],
            dim=-1,
        )
        residual_dynamic = residual_dynamic.reshape(
            *residual_streams.shape[:2],
            self.num_streams,
            self.num_streams,
        )

        h_pre_logits = self.alpha_pre.float() * pre_dynamic + self.pre_logits.float()
        h_post_logits = self.alpha_post.float() * post_dynamic + self.post_logits.float()
        h_res_logits = self.alpha_res.float() * residual_dynamic + self.residual_logits.float()

        h_pre = torch.sigmoid(h_pre_logits).to(dtype=residual_streams.dtype)
        h_post = (2.0 * torch.sigmoid(h_post_logits)).to(dtype=residual_streams.dtype)
        h_res = self._sinkhorn(h_res_logits, self.sinkhorn_iters).to(dtype=residual_streams.dtype)
        return h_pre, h_post, h_res

    def maps(self, residual_streams: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self.baseline == "mhc":
            return self._mhc_maps(residual_streams)
        return None

    def maps_mix(
        self,
        residual_streams: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        if self.baseline == "mhc" and _MHC_COMPILE_ENABLED:
            maps_mix_fn = _get_mhc_maps_mix_fn(self.num_streams)
            mixed, h_post, h_res = maps_mix_fn(
                residual_streams,
                self.mhc_input_norm.weight,
                self.mhc_dynamic_projection.weight,
                self.alpha_pre.float().clone(),
                self.alpha_post.float().clone(),
                self.alpha_res.float().clone(),
                self.pre_logits.float().clone(),
                self.post_logits.float().clone(),
                self.residual_logits.float().clone(),
                self.mhc_input_norm.variance_epsilon,
                self.num_streams,
                self.hidden_size,
                self.sinkhorn_iters,
            )
            return mixed, (None, h_post, h_res)

        maps = self.maps(residual_streams)
        return self.mix(residual_streams, maps), maps

    def mix(
        self,
        residual_streams: torch.Tensor,
        maps: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if self.baseline == "mhc":
            if maps is None:
                maps = self._mhc_maps(residual_streams)
            h_pre, _, _ = maps
            return torch.einsum("btn,btnd->btd", h_pre, residual_streams)

        weights = torch.softmax(self.pre_logits.float(), dim=-1).to(residual_streams.dtype)
        return torch.einsum("n,btnd->btd", weights, residual_streams)

    def _residual_matrix(self, residual_logits: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        matrix = torch.softmax(residual_logits.float(), dim=-1).to(device=device)
        return matrix.to(dtype=dtype)

    def update(
        self,
        residual_streams: torch.Tensor,
        hidden_update: torch.Tensor,
        maps: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if self.baseline == "mhc":
            if maps is None:
                maps = self._mhc_maps(residual_streams)
            _, h_post, h_res = maps
            if _MHC_COMPILE_ENABLED:
                update_fn = _get_mhc_update_fn(self.num_streams)
                return update_fn(residual_streams, hidden_update, h_post, h_res)
            mixed_streams = torch.einsum("btij,btjd->btid", h_res, residual_streams)
            return mixed_streams + h_post.unsqueeze(-1) * hidden_update.unsqueeze(2)

        residual_matrix = self._residual_matrix(self.residual_logits, residual_streams.dtype, residual_streams.device)
        mixed_streams = torch.einsum("ij,btjd->btid", residual_matrix, residual_streams)
        write_gains = (2.0 * torch.sigmoid(self.post_logits.float())).to(residual_streams.dtype)
        return mixed_streams + write_gains.unsqueeze(-1) * hidden_update.unsqueeze(2)


class DenseTransformerDepthWeightedAverager(nn.Module):
    def __init__(self, num_layers: int, dilation: int = 1, period: int = 1):
        super().__init__()
        self.num_layers = num_layers
        self.dilation = max(1, int(dilation))
        self.period = max(1, int(period))
        self.alphas = nn.ParameterList()
        for layer_idx in range(num_layers):
            current_depth = layer_idx + 1
            if current_depth % self.period == 0:
                num_inputs = current_depth // self.dilation + 1
                self.alphas.append(nn.Parameter(torch.empty(num_inputs)))
            else:
                self.alphas.append(nn.Parameter(torch.empty(0), requires_grad=False))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for alpha in self.alphas:
                if alpha.numel() == 0:
                    continue
                alpha.zero_()
                alpha[-1] = 1.0

    def forward(self, hidden_history: List[torch.Tensor], layer_idx: int) -> torch.Tensor:
        current_depth = layer_idx + 1
        if current_depth % self.period != 0:
            return hidden_history[-1]

        alpha = self.alphas[layer_idx].to(dtype=hidden_history[-1].dtype, device=hidden_history[-1].device)
        start_idx = current_depth % self.dilation
        mixed = hidden_history[start_idx] * alpha[0]
        alpha_idx = 1
        for history_idx in range(start_idx + self.dilation, current_depth + 1, self.dilation):
            mixed = mixed + hidden_history[history_idx] * alpha[alpha_idx]
            alpha_idx += 1
        return mixed


class BlockAttentionResidualAggregator(nn.Module):
    def __init__(self, hidden_size: int, eps: float, initializer_range: float = 0.02):
        super().__init__()
        self.hidden_size = hidden_size
        self.initializer_range = float(initializer_range)
        self.pseudo_query = nn.Parameter(torch.zeros(hidden_size))
        self.norm = LlamaRMSNorm(hidden_size, eps=eps)

    def reset_parameters(self):
        with torch.no_grad():
            self.pseudo_query.zero_()
            self.norm.weight.fill_(1.0)

    @staticmethod
    def compute_inv_rms(values: torch.Tensor, eps: float) -> torch.Tensor:
        return torch.rsqrt(values.float().pow(2).mean(-1) + eps)

    def forward(
        self,
        completed_blocks: List[torch.Tensor],
        completed_stack: Optional[torch.Tensor] = None,
        completed_inv_rms: Optional[torch.Tensor] = None,
        partial_block: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if len(completed_blocks) == 0:
            raise ValueError("Block AttnRes requires at least one residual source.")
        if partial_block is None and len(completed_blocks) == 1:
            return completed_blocks[0]

        if completed_stack is None:
            completed_stack = torch.stack(completed_blocks, dim=0)
        if completed_inv_rms is None:
            completed_inv_rms = self.compute_inv_rms(completed_stack, self.norm.variance_epsilon)

        query = (self.pseudo_query * self.norm.weight).to(dtype=completed_stack.dtype)
        completed_logits = torch.matmul(completed_stack, query).float() * completed_inv_rms
        logits = completed_logits
        partial_inv_rms = None
        if partial_block is not None:
            partial_inv_rms = self.compute_inv_rms(partial_block, self.norm.variance_epsilon)
            partial_logits = (torch.matmul(partial_block, query).float() * partial_inv_rms).unsqueeze(0)
            logits = torch.cat([completed_logits, partial_logits], dim=0)

        weights = torch.softmax(logits.float(), dim=0).to(dtype=completed_stack.dtype)
        mixed = (weights[: completed_stack.shape[0]].unsqueeze(-1) * completed_stack).sum(dim=0)
        if partial_block is not None:
            mixed = mixed + weights[-1].unsqueeze(-1) * partial_block
        return mixed


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.residual_baseline = _normalize_residual_baseline(config)
        self.residual_baseline_num_streams = _residual_baseline_num_streams(config, self.residual_baseline, 4)

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.no_residual = getattr(config, "no_residual", False)
        if self.residual_baseline in ("hyper_connection", "mhc"):
            self.attn_residual_connector = ResidualStreamConnector(config, self.residual_baseline)
            self.mlp_residual_connector = ResidualStreamConnector(config, self.residual_baseline)
        elif self.residual_baseline == "densenet":
            growth_size = _positive_config_int(config, "densenet_growth_size", 64)
            block_size = _positive_config_int(config, "densenet_block_size", 24)
            block_offset = layer_idx % block_size
            dense_width = config.hidden_size + block_offset * growth_size
            self.densenet_dense_width = dense_width
            self.densenet_input_norm = LlamaRMSNorm(dense_width, eps=config.rms_norm_eps)
            self.densenet_input_proj = nn.Linear(dense_width, config.hidden_size, bias=False)
            self.densenet_input_proj._is_densenet_input_proj = True
            self.densenet_emits_growth = layer_idx + 1 < config.num_hidden_layers and (layer_idx + 1) % block_size != 0
            if self.densenet_emits_growth:
                self.densenet_growth_proj = nn.Linear(config.hidden_size, growth_size, bias=False)
        elif self.residual_baseline == "attnres":
            self.attnres_attn_aggregator = BlockAttentionResidualAggregator(
                config.hidden_size,
                config.rms_norm_eps,
                initializer_range=config.initializer_range,
            )
            self.attnres_mlp_aggregator = BlockAttentionResidualAggregator(
                config.hidden_size,
                config.rms_norm_eps,
                initializer_range=config.initializer_range,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        cross_layer_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        collect_layer_kv: bool = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        layer_kv = None
        if collect_layer_kv:
            hidden_states, self_attn_weights, present_key_value, layer_kv = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                cross_layer_kv=cross_layer_kv,
                return_layer_kv=True,
                **kwargs,
            )
        else:
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                cross_layer_kv=cross_layer_kv,
                return_layer_kv=False,
                **kwargs,
            )
        if not self.no_residual:
            hidden_states = residual + hidden_states

        # Fully Connected
        if not self.no_residual:
            residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if not self.no_residual:
            hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
        if collect_layer_kv:
            outputs += (layer_kv,)

        return outputs

    def _expand_stream_update(self, hidden_update: torch.Tensor) -> torch.Tensor:
        return hidden_update.unsqueeze(2).expand(-1, -1, self.residual_baseline_num_streams, -1).contiguous()

    def forward_streams(
        self,
        residual_streams: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cross_layer_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        collect_layer_kv: bool = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        hidden_states, attn_maps = self.attn_residual_connector.maps_mix(residual_streams)
        hidden_states = self.input_layernorm(hidden_states)

        layer_kv = None
        if collect_layer_kv:
            hidden_update, self_attn_weights, present_key_value, layer_kv = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                cross_layer_kv=cross_layer_kv,
                return_layer_kv=True,
                **kwargs,
            )
        else:
            hidden_update, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                cross_layer_kv=cross_layer_kv,
                return_layer_kv=False,
                **kwargs,
            )
        if self.no_residual:
            residual_streams = self._expand_stream_update(hidden_update)
        else:
            residual_streams = self.attn_residual_connector.update(residual_streams, hidden_update, attn_maps)

        hidden_states, mlp_maps = self.mlp_residual_connector.maps_mix(residual_streams)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_update = self.mlp(hidden_states)
        if self.no_residual:
            residual_streams = self._expand_stream_update(hidden_update)
        else:
            residual_streams = self.mlp_residual_connector.update(residual_streams, hidden_update, mlp_maps)

        outputs = (residual_streams,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
        if collect_layer_kv:
            outputs += (layer_kv,)

        return outputs

    @staticmethod
    def _attnres_maybe_commit_block(
        completed_blocks: List[torch.Tensor],
        partial_block: Optional[torch.Tensor],
        partial_sublayers: int,
        block_size: int,
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor], int]:
        if partial_block is not None and partial_sublayers >= block_size:
            return completed_blocks + [partial_block], None, 0
        return completed_blocks, partial_block, partial_sublayers

    @staticmethod
    def _attnres_maybe_commit_block_cached(
        completed_blocks: List[torch.Tensor],
        completed_stack: torch.Tensor,
        completed_inv_rms: torch.Tensor,
        partial_block: Optional[torch.Tensor],
        partial_sublayers: int,
        block_size: int,
        eps: float,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, Optional[torch.Tensor], int]:
        if partial_block is not None and partial_sublayers >= block_size:
            completed_blocks = completed_blocks + [partial_block]
            completed_stack = torch.cat([completed_stack, partial_block.unsqueeze(0)], dim=0)
            partial_inv_rms = BlockAttentionResidualAggregator.compute_inv_rms(partial_block, eps)
            completed_inv_rms = torch.cat([completed_inv_rms, partial_inv_rms.unsqueeze(0)], dim=0)
            return completed_blocks, completed_stack, completed_inv_rms, None, 0
        return completed_blocks, completed_stack, completed_inv_rms, partial_block, partial_sublayers

    def forward_attnres(
        self,
        completed_blocks: List[torch.Tensor],
        completed_stack: torch.Tensor,
        completed_inv_rms: torch.Tensor,
        partial_block: Optional[torch.Tensor],
        partial_sublayers: int,
        attnres_block_size: int,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cross_layer_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        collect_layer_kv: bool = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, List[torch.Tensor], torch.Tensor, torch.Tensor, Optional[torch.Tensor], int]:
        completed_blocks, completed_stack, completed_inv_rms, partial_block, partial_sublayers = (
            self._attnres_maybe_commit_block_cached(
            completed_blocks,
            completed_stack,
            completed_inv_rms,
            partial_block,
            partial_sublayers,
            attnres_block_size,
            self.attnres_attn_aggregator.norm.variance_epsilon,
        )
        )
        hidden_states = self.attnres_attn_aggregator(
            completed_blocks,
            completed_stack,
            completed_inv_rms,
            partial_block,
        )
        hidden_states = self.input_layernorm(hidden_states)

        layer_kv = None
        if collect_layer_kv:
            hidden_update, self_attn_weights, present_key_value, layer_kv = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                cross_layer_kv=cross_layer_kv,
                return_layer_kv=True,
                **kwargs,
            )
        else:
            hidden_update, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                cross_layer_kv=cross_layer_kv,
                return_layer_kv=False,
                **kwargs,
            )
        partial_block = hidden_update if partial_block is None else partial_block + hidden_update
        partial_sublayers += 1

        completed_blocks, completed_stack, completed_inv_rms, partial_block, partial_sublayers = (
            self._attnres_maybe_commit_block_cached(
            completed_blocks,
            completed_stack,
            completed_inv_rms,
            partial_block,
            partial_sublayers,
            attnres_block_size,
            self.attnres_mlp_aggregator.norm.variance_epsilon,
        )
        )
        hidden_states = self.attnres_mlp_aggregator(
            completed_blocks,
            completed_stack,
            completed_inv_rms,
            partial_block,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_update = self.mlp(hidden_states)
        partial_block = hidden_update if partial_block is None else partial_block + hidden_update
        partial_sublayers += 1

        hidden_states = partial_block if partial_block is not None else completed_blocks[-1]
        outputs = (hidden_states, completed_blocks, completed_stack, completed_inv_rms, partial_block, partial_sublayers)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
        if collect_layer_kv:
            outputs += (layer_kv,)

        return outputs

    def forward_densenet_from_concat(
        self,
        hidden_states: torch.Tensor,
        dense_input: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cross_layer_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        collect_layer_kv: bool = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        dense_update = self.densenet_input_proj(self.densenet_input_norm(dense_input))
        hidden_states = hidden_states + dense_update
        outputs = self.forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            cross_layer_kv=cross_layer_kv,
            collect_layer_kv=collect_layer_kv,
            **kwargs,
        )
        if self.densenet_emits_growth:
            growth_state = self.densenet_growth_proj(outputs[0])
        else:
            growth_state = outputs[0].new_empty(0)
        return (outputs[0], growth_state) + outputs[1:]


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    # def _init_weights(self, module):
    #     """Initialize the weights"""
    #     if isinstance(module, nn.Linear):
    #         # 使用截断正态分布初始化权重，截断在3σ处，方差为2/(5*hidden_size)
    #         std = math.sqrt(2.0 / (5 * self.config.hidden_size))
            
    #         # 对于注意力层和MLP层的输出投影，根据网络深度进行额外缩放
    #         if hasattr(module, '_is_attention_output') or hasattr(module, '_is_mlp_output'):
    #             std = std / math.sqrt(2.0 * self.config.num_hidden_layers)
    #             print(f"small output std: {std}")
                
    #         nn.init.trunc_normal_(module.weight.data, mean=0.0, std=std, a=-3*std, b=3*std)
    #         if module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, nn.Embedding):
    #         # 使用截断正态分布初始化嵌入层权重
    #         std = math.sqrt(2.0 / (5 * self.config.hidden_size))
    #         nn.init.trunc_normal_(module.weight.data, mean=0.0, std=std, a=-3*std, b=3*std)
    #         if module.padding_idx is not None:
    #             module.weight.data[module.padding_idx].zero_()
    #     elif isinstance(module, nn.LayerNorm):
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance, see our
            [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache);
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.residual_baseline = _normalize_residual_baseline(config)
        self.residual_baseline_num_streams = _residual_baseline_num_streams(config, self.residual_baseline, 4)
        self.densenet_block_size = _positive_config_int(config, "densenet_block_size", 24)
        self.densetransformer_dwa_dilation = _positive_config_int_any(
            config, ("denseformer_dwa_dilation", "densetransformer_dwa_dilation"), 1
        )
        self.densetransformer_dwa_period = _positive_config_int_any(
            config, ("denseformer_dwa_period", "densetransformer_dwa_period"), 1
        )
        self.attnres_block_size = _positive_config_int(config, "attnres_block_size", 12)
        self.recurrent_model_enabled = bool(getattr(config, "recurrent_model", False))
        self.cross_layer_pattern = getattr(config, "cross_layer_pattern", None)
        self.layer_kv_reuse_map = self._normalize_layer_kv_reuse_map(
            getattr(config, "layer_kv_reuse_map", None), config.num_hidden_layers
        )
        if self.residual_baseline in ("hyper_connection", "mhc"):
            self.residual_stream_projector = ResidualStreamProjector(self.residual_baseline_num_streams)
        elif self.residual_baseline == "densetransformer":
            self.densetransformer_dwa = DenseTransformerDepthWeightedAverager(
                config.num_hidden_layers,
                dilation=self.densetransformer_dwa_dilation,
                period=self.densetransformer_dwa_period,
            )
        elif self.residual_baseline == "attnres":
            self.attnres_final_aggregator = BlockAttentionResidualAggregator(
                config.hidden_size,
                config.rms_norm_eps,
                initializer_range=config.initializer_range,
            )

        # Initialize weights and apply final processing
        self.post_init()
        self._reset_residual_baseline_parameters()
        # === 【新增】初始化打印标志位 ===
        self.has_printed_debug = False
        self._attnres_last_num_completed_blocks = 0
        self._attnres_last_partial_sublayers = 0

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _reset_residual_baseline_parameters(self):
        if self.residual_baseline in ("hyper_connection", "mhc"):
            with torch.no_grad():
                self.residual_stream_projector.reduce_logits.zero_()
            for layer in self.layers:
                layer.attn_residual_connector.reset_parameters()
                layer.mlp_residual_connector.reset_parameters()
        elif self.residual_baseline == "densenet":
            for layer in self.layers:
                with torch.no_grad():
                    layer.densenet_input_proj.weight.zero_()
        elif self.residual_baseline == "densetransformer":
            self.densetransformer_dwa.reset_parameters()
        elif self.residual_baseline == "attnres":
            self.attnres_final_aggregator.reset_parameters()
            for layer in self.layers:
                layer.attnres_attn_aggregator.reset_parameters()
                layer.attnres_mlp_aggregator.reset_parameters()

    @staticmethod
    def _normalize_layer_kv_reuse_map(raw_map, num_layers: int) -> dict:
        normalized = {}
        if raw_map is None:
            return normalized
        if isinstance(raw_map, (list, tuple)):
            for tgt_idx, src_layers in enumerate(raw_map):
                if src_layers is None:
                    continue
                if not isinstance(src_layers, (list, tuple)):
                    src_layers = [src_layers]
                valid_src = [int(s) for s in src_layers if 0 <= int(s) < tgt_idx]
                if 0 <= tgt_idx < num_layers and valid_src:
                    normalized[int(tgt_idx)] = valid_src
            return normalized
        if isinstance(raw_map, dict):
            for tgt, src_layers in raw_map.items():
                tgt_idx = int(tgt)
                if not (0 <= tgt_idx < num_layers):
                    continue
                if src_layers is None:
                    continue
                if not isinstance(src_layers, (list, tuple)):
                    src_layers = [src_layers]
                valid_src = [int(s) for s in src_layers if 0 <= int(s) < tgt_idx]
                if valid_src:
                    normalized[tgt_idx] = valid_src
        return normalized

    @staticmethod
    def _debug_tensor_seq_len(tensor) -> int:
        if torch.is_tensor(tensor) and tensor.ndim >= 3:
            return int(tensor.shape[-2])
        return 0

    @classmethod
    def _debug_cross_kv_seq_len(cls, cross_layer_kv) -> int:
        if cross_layer_kv is None:
            return 0

        # Stacked depth-softmax format: (stacked_k, stacked_v) or
        # (None, stacked_v, stacked_depth_k), each stacked tensor is [N,B,H,T,d].
        if isinstance(cross_layer_kv, tuple) and len(cross_layer_kv) >= 2:
            for item in cross_layer_kv:
                if torch.is_tensor(item) and item.ndim == 5:
                    return int(item.shape[0] * item.shape[-2])

            # Defensive handling for a single non-stacked KV tuple.
            if all(item is None or torch.is_tensor(item) for item in cross_layer_kv):
                for item in cross_layer_kv:
                    if torch.is_tensor(item):
                        return cls._debug_tensor_seq_len(item)
                return 0

        cross_len = 0
        for kv in cross_layer_kv:
            if kv is None:
                continue
            if torch.is_tensor(kv):
                cross_len += cls._debug_tensor_seq_len(kv)
                continue
            if not isinstance(kv, (list, tuple)):
                continue
            for item in kv:
                if torch.is_tensor(item):
                    cross_len += cls._debug_tensor_seq_len(item)
                    break
        return cross_len

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # === 【新增】Debug 信息打印逻辑 ===
        # 只在训练模式下，且只打印一次
        if self.training and not self.has_printed_debug:
            try:
                # 获取当前 rank，避免多卡训练时刷屏（如果获取失败则默认打印）
                import torch.distributed as dist
                is_main_process = not dist.is_initialized() or dist.get_rank() == 0
            except:
                is_main_process = True

            if is_main_process:
                print("\n" + "="*50)
                print(f"[Depth-Attention DEBUG] Layer KV Reuse Map Configuration")
                print("="*50)
                if not self.layer_kv_reuse_map:
                    print("Status: DISABLED (Map is empty)")
                else:
                    print(f"Status: ACTIVE")
                    print(f"Total Layers: {len(self.layers)}")
                    print(f"Reuse Map Content (Target Layer -> [Source Layers]):")
                    # 为了显示整齐，排序打印
                    for tgt_layer in sorted(self.layer_kv_reuse_map.keys()):
                        src_layers = self.layer_kv_reuse_map[tgt_layer]
                        print(f"  Layer {tgt_layer:02d} <== reads KV from ==> Layer {src_layers}")
                print("="*50 + "\n")
        # ==========================================

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        shared_rope_phase = getattr(self.rotary_emb, "_last_trace_phase", None)
        if shared_rope_phase is not None:
            for layer in self.layers:
                layer.self_attn._trace_shared_rope_phase = shared_rope_phase
        use_stream_baseline = self.residual_baseline in ("hyper_connection", "mhc")
        use_densenet_baseline = self.residual_baseline == "densenet"
        use_densetransformer_baseline = self.residual_baseline == "densetransformer"
        use_attnres_baseline = self.residual_baseline == "attnres"
        if use_stream_baseline:
            hidden_states = self.residual_stream_projector.expand(hidden_states)
        dense_growth_states = []
        dense_block_base = None
        densetransformer_hidden_history = [hidden_states] if use_densetransformer_baseline else None
        attnres_completed_blocks = [hidden_states] if use_attnres_baseline else None
        attnres_completed_stack = hidden_states.unsqueeze(0) if use_attnres_baseline else None
        attnres_completed_inv_rms = (
            BlockAttentionResidualAggregator.compute_inv_rms(
                hidden_states,
                self.attnres_final_aggregator.norm.variance_epsilon,
            ).unsqueeze(0)
            if use_attnres_baseline
            else None
        )
        attnres_partial_block = None
        attnres_partial_sublayers = 0

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        recurrent_model_enabled = bool(getattr(self.config, "recurrent_model", self.recurrent_model_enabled))
        if self.residual_baseline is not None:
            recurrent_model_enabled = False
        if use_attnres_baseline and self.gradient_checkpointing and self.training:
            raise ValueError("Block AttnRes baseline requires --disable_gradient_checkpointing true.")
        use_dense_cross = recurrent_model_enabled and self.cross_layer_pattern == "dense"
        use_depth_softmax = recurrent_model_enabled and self.cross_layer_pattern == "depth_softmax"
        use_all_attention = recurrent_model_enabled and self.cross_layer_pattern == "all_attention"
        collect_layer_kv = recurrent_model_enabled and (len(self.layer_kv_reuse_map) > 0 or use_dense_cross or use_depth_softmax or use_all_attention)
        if collect_layer_kv and self.config._attn_implementation == "flash_attention_2":
            raise ValueError("cross-layer KV 需要 eager/sdpa 注意力实现，flash_attention_2 不支持。")
        layer_kv_cache = [None] * len(self.layers) if (collect_layer_kv and not use_dense_cross and not use_all_attention) else None

        # Dense running mean state
        kv_running_sum_k = None
        kv_running_sum_v = None
        kv_count = 0

        # Depth softmax / All attention: collect all previous layers' KV as a list
        depth_kv_list = [] if use_depth_softmax else None
        all_attention_kv_list = [] if use_all_attention else None

        # OPTIMIZATION: incremental stacked buffer for depth_softmax with stride>1.
        # Instead of re-stacking cross K/V at each layer, maintain a growing buffer
        # that's only updated when a new stride-aligned layer is processed.
        _ds_mode = getattr(self.config, "cross_layer_mode", "")
        _ds_stride = int(getattr(self.config, "depth_softmax_stride", 1))
        _ds_recent_window = int(getattr(self.config, "depth_recent_window", 0))
        _use_incremental_stack = (
            use_depth_softmax
            and _ds_stride > 1
            and _ds_recent_window <= 0
            # depth_softmax_head0_v2 branch expects list format with raw K and
            # re-applies self.k_norm per layer; it does not support stacked tuples.
            and _ds_mode not in ("v0_mix", "depth_softmax_head0_v2")
        )
        _depth_k_stacked = None  # [N, B, H, T, d]
        _depth_v_stacked = None
        _depth_dk_stacked = None  # [N, B, 1, T, d] only for depth-mix auxiliary layout

        try:
            import torch.distributed as dist
            is_main_process = not dist.is_initialized() or dist.get_rank() == 0
        except:
            is_main_process = True

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                if use_stream_baseline:
                    all_hidden_states += (self.residual_stream_projector.reduce(hidden_states),)
                else:
                    all_hidden_states += (hidden_states,)

            cross_layer_kv = None
            if use_all_attention and len(all_attention_kv_list) > 0:
                # All Attention: pass ALL previous layers' KV as cross_layer_kv
                # The attention module (cross_attn mode) will concat them along seq dim
                cross_layer_kv = all_attention_kv_list
            elif use_depth_softmax and len(depth_kv_list) > 0:
                if _ds_mode == "v0_mix":
                    # v0_mix: only need layer 0's KV
                    cross_layer_kv = [depth_kv_list[0]]
                elif _use_incremental_stack and _depth_v_stacked is not None:
                    # OPTIMIZATION: use pre-built stacked buffer instead of re-stacking.
                    # For depth-mix auxiliary layout, _depth_k_stacked is None (main K is not
                    # consumed downstream); pass None in position 0 so SDPA detects the
                    # 1head stacked layout.
                    if _depth_k_stacked is None and _depth_dk_stacked is not None:
                        cross_layer_kv = (None, _depth_v_stacked, _depth_dk_stacked)
                    elif _depth_dk_stacked is not None:
                        cross_layer_kv = (_depth_k_stacked, _depth_v_stacked, _depth_dk_stacked)
                    else:
                        cross_layer_kv = (_depth_k_stacked, _depth_v_stacked)
                else:
                    # Fallback: list format (stride=1 or recent_window>0)
                    depth_stride = _ds_stride
                    depth_recent_window = _ds_recent_window
                    if depth_stride <= 1 and depth_recent_window <= 0:
                        cross_layer_kv = depth_kv_list
                    else:
                        n = len(depth_kv_list)
                        selected = set()
                        if depth_stride > 1:
                            selected.update(i for i in range(n) if i % depth_stride == 0)
                        else:
                            selected.update(range(n))
                        if depth_recent_window > 0:
                            selected.update(range(max(0, n - depth_recent_window), n))
                        cross_layer_kv = [depth_kv_list[i] for i in sorted(selected)]
                        if len(cross_layer_kv) == 0:
                            cross_layer_kv = None
            elif use_dense_cross and kv_count > 0:
                # Dense: cross-attend to running mean of all previous layers' KV
                cross_layer_kv = [(kv_running_sum_k / kv_count, kv_running_sum_v / kv_count)]
            elif not use_dense_cross and not use_depth_softmax and collect_layer_kv and layer_idx in self.layer_kv_reuse_map:
                cross_layer_kv = []
                for src_idx in self.layer_kv_reuse_map[layer_idx]:
                    src_kv = layer_kv_cache[src_idx]
                    if src_kv is not None:
                        cross_layer_kv.append(src_kv)
                if len(cross_layer_kv) == 0:
                    cross_layer_kv = None
            # ============================================================
            # 【新增】简单粗暴的 Debug：打印每一层的 KV 长度
            # ============================================================
            # 只有在训练的第一步打印 (has_printed_debug 还没变 True 时)
            if self.training and not getattr(self, "has_printed_debug", False) and is_main_process:
                # 1. 计算当前层的序列长度 (Base)
                base_len = hidden_states.shape[1] 
                
                # 2. 计算跨层的序列长度 (Cross)
                cross_len = self._debug_cross_kv_seq_len(cross_layer_kv)
                
                # 3. 打印结果
                total_len = base_len + cross_len
                print(f"[Layer {layer_idx:02d}] Input: {base_len} + Cross: {cross_len} ==> Total KV: {total_len}")
            # ============================================================
            
            if use_densenet_baseline and layer_idx % self.densenet_block_size == 0:
                dense_block_base = hidden_states
                dense_growth_states = []
            if use_densenet_baseline and dense_growth_states:
                dense_input = torch.cat([dense_block_base] + dense_growth_states, dim=-1)
            else:
                dense_input = dense_block_base if use_densenet_baseline else hidden_states

            if self.gradient_checkpointing and self.training:
                if use_stream_baseline:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.forward_streams,
                        hidden_states,
                        causal_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        cache_position,
                        position_embeddings,
                        cross_layer_kv,
                        collect_layer_kv,
                    )
                elif use_densenet_baseline:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.forward_densenet_from_concat,
                        hidden_states,
                        dense_input,
                        causal_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        cache_position,
                        position_embeddings,
                        cross_layer_kv,
                        collect_layer_kv,
                    )
                else:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        causal_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        cache_position,
                        position_embeddings,
                        cross_layer_kv,
                        collect_layer_kv,
                    )
            else:
                if use_stream_baseline:
                    layer_outputs = decoder_layer.forward_streams(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        cross_layer_kv=cross_layer_kv,
                        collect_layer_kv=collect_layer_kv,
                    )
                elif use_densenet_baseline:
                    layer_outputs = decoder_layer.forward_densenet_from_concat(
                        hidden_states,
                        dense_input,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        cross_layer_kv=cross_layer_kv,
                        collect_layer_kv=collect_layer_kv,
                    )
                elif use_attnres_baseline:
                    layer_outputs = decoder_layer.forward_attnres(
                        attnres_completed_blocks,
                        attnres_completed_stack,
                        attnres_completed_inv_rms,
                        attnres_partial_block,
                        attnres_partial_sublayers,
                        self.attnres_block_size,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        cross_layer_kv=cross_layer_kv,
                        collect_layer_kv=collect_layer_kv,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        cross_layer_kv=cross_layer_kv,
                        collect_layer_kv=collect_layer_kv,
                    )

            hidden_states = layer_outputs[0]
            offset = 1
            if use_densenet_baseline:
                growth_state = layer_outputs[offset]
                offset += 1
                if decoder_layer.densenet_emits_growth:
                    dense_growth_states.append(growth_state)
            elif use_densetransformer_baseline:
                densetransformer_hidden_history.append(hidden_states)
                hidden_states = self.densetransformer_dwa(densetransformer_hidden_history, layer_idx)
            elif use_attnres_baseline:
                attnres_completed_blocks = layer_outputs[offset]
                offset += 1
                attnres_completed_stack = layer_outputs[offset]
                offset += 1
                attnres_completed_inv_rms = layer_outputs[offset]
                offset += 1
                attnres_partial_block = layer_outputs[offset]
                offset += 1
                attnres_partial_sublayers = layer_outputs[offset]
                offset += 1
            if output_attentions:
                all_self_attns += (layer_outputs[offset],)
                offset += 1
            if use_cache:
                next_decoder_cache = layer_outputs[offset]
                offset += 1
            if collect_layer_kv:
                layer_kv = layer_outputs[offset]
                if use_all_attention:
                    all_attention_kv_list.append(layer_kv)
                elif use_depth_softmax:
                    depth_kv_list.append(layer_kv)
                    # Incremental stacked buffer update: only when this layer is stride-aligned.
                    # Guard every append on "not None" because 1head stores None for main K
                    # (layer_kv[0]) and skipping the torch.cat there saves ~138 MB/step and
                    # an extra cat op per stride-aligned layer.
                    if _use_incremental_stack and layer_idx % _ds_stride == 0:
                        _k = layer_kv[0]
                        _v = layer_kv[1]
                        _dk = layer_kv[2] if len(layer_kv) >= 3 else None
                        if _depth_v_stacked is None:
                            if _k is not None:
                                _depth_k_stacked = _k.unsqueeze(0)
                            _depth_v_stacked = _v.unsqueeze(0)
                            if _dk is not None:
                                _depth_dk_stacked = _dk.unsqueeze(0)
                        else:
                            if _k is not None:
                                _depth_k_stacked = torch.cat([_depth_k_stacked, _k.unsqueeze(0)], dim=0)
                            _depth_v_stacked = torch.cat([_depth_v_stacked, _v.unsqueeze(0)], dim=0)
                            if _dk is not None:
                                _depth_dk_stacked = torch.cat([_depth_dk_stacked, _dk.unsqueeze(0)], dim=0)
                elif use_dense_cross:
                    # Accumulate running sum for dense pattern
                    k, v = layer_kv
                    if kv_running_sum_k is None:
                        kv_running_sum_k = k
                        kv_running_sum_v = v
                    else:
                        kv_running_sum_k = kv_running_sum_k + k
                        kv_running_sum_v = kv_running_sum_v + v
                    kv_count += 1
                else:
                    layer_kv_cache[layer_idx] = layer_kv

        if use_stream_baseline:
            hidden_states = self.residual_stream_projector.reduce(hidden_states)
        elif use_attnres_baseline:
            (
                attnres_completed_blocks,
                attnres_completed_stack,
                attnres_completed_inv_rms,
                attnres_partial_block,
                attnres_partial_sublayers,
            ) = (
                LlamaDecoderLayer._attnres_maybe_commit_block_cached(
                    attnres_completed_blocks,
                    attnres_completed_stack,
                    attnres_completed_inv_rms,
                    attnres_partial_block,
                    attnres_partial_sublayers,
                    self.attnres_block_size,
                    self.attnres_final_aggregator.norm.variance_epsilon,
                )
            )
            self._attnres_last_num_completed_blocks = len(attnres_completed_blocks)
            self._attnres_last_partial_sublayers = attnres_partial_sublayers
            hidden_states = self.attnres_final_aggregator(
                attnres_completed_blocks,
                attnres_completed_stack,
                attnres_completed_inv_rms,
                attnres_partial_block,
            )
        hidden_states = self.norm(hidden_states)
        
        # === 【新增】关闭 Debug 开关 ===
        if self.training and is_main_process:
            self.has_printed_debug = True
        # ============================

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # ln_eps = getattr(config, "layer_norm_eps", getattr(config, "rms_norm_eps", 1e-6))
        # self.lm_head_layernorm = nn.LayerNorm(config.hidden_size, eps=ln_eps)
        self.lm_head_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Initialize weights and apply final processing
        self.post_init()
        self.model._reset_residual_baseline_parameters()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def _router_forward(
        self,
        input_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.FloatTensor],
        high_entropy_mask: Optional[torch.LongTensor],
        position_ids: Optional[torch.LongTensor],
        inputs_embeds: Optional[torch.FloatTensor],
        past_key_values: Optional[Union[Cache, Tuple[Tuple[torch.FloatTensor]]]],
        labels: Optional[torch.LongTensor],
        use_cache: Optional[bool],
        output_attentions: Optional[bool],
        output_hidden_states: Optional[bool],
        return_dict: Optional[bool],
        cache_position: Optional[torch.LongTensor],
        global_step: Optional[int],
        eval_prune_threshold: Optional[float] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        user_requested_output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if self.training and use_cache:
            use_cache = False

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids must be provided when inputs_embeds is None")
            initial_embeds_raw = self.model.embed_tokens(input_ids)
        else:
            initial_embeds_raw = inputs_embeds

        _, _, H = initial_embeds_raw.shape
        scaled_initial_embeds = initial_embeds_raw
        if hasattr(self.config, "scale_embeds") and self.config.scale_embeds:
            device = initial_embeds_raw.device
            embed_scale = torch.sqrt(torch.tensor(H, dtype=initial_embeds_raw.dtype, device=device))
            scaled_initial_embeds = initial_embeds_raw * embed_scale

        self.last_lm_loss = None
        model_outputs = self.model(
            inputs_embeds=scaled_initial_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=user_requested_output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
        )
        final_hidden = model_outputs.last_hidden_state
        logits_full = self.lm_head(final_hidden)
        final_logits = logits_full
        if num_logits_to_keep > 0:
            final_logits = logits_full[:, -num_logits_to_keep:, :]

        loss = None
        if labels is not None:
            shift_logits = logits_full[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        try:
            self.last_lm_loss = float(loss.detach().cpu().item())
        except Exception:
            self.last_lm_loss = None
        
        if not return_dict:
            items = [final_logits]
            if model_outputs.past_key_values is not None:
                items.append(model_outputs.past_key_values)
            if user_requested_output_hidden_states:
                items.append(model_outputs.hidden_states)
            if output_attentions:
                items.append(model_outputs.attentions)
            tup = tuple(items)
            return ((loss,) + tup) if loss is not None else tup

        return CausalLMOutputWithPast(
            loss=loss,
            logits=final_logits,
            past_key_values=model_outputs.past_key_values,
            hidden_states=model_outputs.hidden_states if user_requested_output_hidden_states else None,
            attentions=model_outputs.attentions if output_attentions else None,
        )

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        global_step: Optional[int] = None,
        eval_prune_threshold: Optional[float] = None,
        high_entropy_mask: Optional[torch.LongTensor] = None,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        return self._router_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            high_entropy_mask=high_entropy_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            global_step=global_step,
            eval_prune_threshold=eval_prune_threshold,
            num_logits_to_keep=num_logits_to_keep,
        )


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """
The Llama Model transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForQuestionAnswering(LlamaPreTrainedModel):
    base_model_prefix = "transformer"

    # Copied from transformers.models.bloom.modeling_bloom.BloomForQuestionAnswering.__init__ with Bloom->Llama
    def __init__(self, config):
        super().__init__(config)
        self.transformer = LlamaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        loss = None
        if start_positions is not None and end_positions is not None:
            loss = self.loss_function(start_logits, end_logits, start_positions, end_positions, **kwargs)

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    The Llama Model transformer with a token classification head on top (a linear layer on top of the hidden-states
    output) e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForTokenClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
