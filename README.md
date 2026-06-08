# Depth-Attention
### Cross-Layer Value Mixing for Language Models

[![Paper](https://img.shields.io/badge/arXiv-2606.05014-b31b1b.svg)](https://arxiv.org/abs/2606.05014)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![LLaMA-Factory](https://img.shields.io/badge/Built%20on-LLaMA--Factory-2563eb.svg)](https://github.com/hiyouga/LLaMA-Factory)

> **TL;DR.** Transformers can choose information flexibly across the sequence,
> but across depth they mostly rely on residual addition. **Depth-Attention**
> moves cross-layer selection into attention itself: before a layer performs
> causal self-attention, its query scores same-position keys from selected
> earlier layers and mixes their values into the value read by self-attention.
> It reuses the normal Q/K/V projections and KV-cache slots, adding no model
> parameters and no persistent inference state beyond the standard KV cache.
> The paper reports consistent gains from 360M to 3B models, including up to
> 2.3 average downstream accuracy points over vanilla Transformers.

```text
vanilla:
  layer l value  v_l  ---------------------------> causal self-attention

Depth-Attention:
  q_l scores same-token depth candidates:
      (k_0, v_0), (k_4, v_4), ..., (k_l, v_l)
            softmax over depth -> mixed v_l -> causal self-attention
```

---

## News

- **2026-06-03**: [Depth-Attention: Cross-Layer Value Mixing for Language Models](https://arxiv.org/abs/2606.05014) released on arXiv.

---

## What's in this repo

A slim fork of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for
pretraining Llama-style models with Depth-Attention and the paper baselines.
All methods share the same CLI and switch with one flag:

| `--patch_method` | Method | Modeling file |
|---|---|---|
| `depth_attention` | **Depth-Attention / depth-softmax** | `modeling_llama_depth_attention.py` |
| `attnres` | Block AttnRes baseline | `modeling_llama_attnres.py` |
| `denseformer` | DenseFormer / DenseTransformer baseline | `modeling_llama_denseformer.py` |
| `mhc` | Multi-stream residual connector baseline | `modeling_llama_mhc.py` |
| `vanilla` | Stock Llama | Hugging Face Transformers |

The Llama implementation files are synchronized with the internal reference
implementations used for the paper, including native GQA support, q/k head
RMSNorm, Hugging Face cache classes, and the AttnRes / DenseFormer / mHC
comparison paths.

---

## Repository tour

```text
Depth-Attention/
├── scripts/                         # quickstart + one launcher per method
├── llama_config/                    # tiny and 410M-class GQA4x configs
├── llama_config/released/           # 1.5B / 3B released checkpoint configs
├── examples/deepspeed/              # ZeRO configs
├── examples/released_models/        # training recipes for released models
├── data/dataset_info.json           # smoke_text, minipile, smallpile, testpile
└── src/llamafactory/model/
    ├── llama_patch.py               # method swap entry point
    └── modeling/
        ├── modeling_llama_depth_attention.py
        ├── modeling_llama_attnres.py
        ├── modeling_llama_denseformer.py
        └── modeling_llama_mhc.py
```

---

## Released checkpoints

The following trained checkpoints are released under
[zeng123](https://huggingface.co/zeng123) on Hugging Face:

| Size | Method | Checkpoint | Model config | Training recipe |
|---|---|---|---|---|
| 1.5B | DenseFormer | [zeng123/1.5b-denseformer-qknorm-gqa-4x](https://huggingface.co/zeng123/1.5b-denseformer-qknorm-gqa-4x) | `llama_config/released/1.5b_denseformer_qknorm_gqa4x` | `examples/released_models/train_1.5b_denseformer_qknorm_gqa4x.yaml` |
| 1.5B | mHC | [zeng123/1.5b-mhc-qknorm-gqa-4x](https://huggingface.co/zeng123/1.5b-mhc-qknorm-gqa-4x) | `llama_config/released/1.5b_mhc_qknorm_gqa4x` | `examples/released_models/train_1.5b_mhc_qknorm_gqa4x.yaml` |
| 3B | DenseFormer | [zeng123/3b-denseformer-qknorm-gqa-4x](https://huggingface.co/zeng123/3b-denseformer-qknorm-gqa-4x) | `llama_config/released/3b_denseformer_qknorm_gqa4x` | `examples/released_models/train_3b_denseformer_qknorm_gqa4x.yaml` |
| 3B | mHC | [zeng123/3b-mhc-qknorm-gqa-4x](https://huggingface.co/zeng123/3b-mhc-qknorm-gqa-4x) | `llama_config/released/3b_mhc_qknorm_gqa4x` | `examples/released_models/train_3b_mhc_qknorm_gqa4x.yaml` |

The 1.5B configs use 48 layers, hidden size 1536, 24 attention heads, and
6 KV heads. The 3B configs use 48 layers, hidden size 2048, 32 attention
heads, and 8 KV heads. All four checkpoints use QK norm and GQA4x.

The released training recipes are assembled from each checkpoint's exported
`config.json` and Trainer-generated model card. They record the public
smallpile recipe: 32 devices, per-device batch size 4, gradient accumulation
8, global batch size 1024, AdamW with betas `(0.9, 0.95)`, cosine-with-min-lr
scheduling, warmup ratio 0.02, and 1 epoch. Use `HF_ENDPOINT=https://hf-mirror.com`
when downloading from networks that require the Hugging Face mirror.

---

## Install

```bash
git clone https://github.com/LUMIA-Group/Depth-Attention.git
cd Depth-Attention
conda create -n depth-attention python=3.10 -y
conda activate depth-attention
pip install -e ".[torch,metrics,deepspeed]"
pip install wandb
```

For FlashAttention training, install the build matching your CUDA/PyTorch
stack. For example:

```bash
pip install flash-attn==2.7.2.post1 --no-build-isolation
```

This repository follows the LLaMA-Factory dependency range
`transformers>=4.41.2,<=4.46.1`.

---

## Quickstart - tiny Llama on local smoke data

A one-step CPU/GPU smoke run is included so you can confirm the install and
patch path before launching larger experiments:

```bash
bash scripts/quickstart_depth_attention_tiny.sh
```

Behind the scenes:

```bash
llamafactory-cli train \
    --model_name_or_path llama_config/depth_attention_tiny \
    --tokenizer_name_or_path data/tiny_tokenizer \
    --stage pt --do_train --finetuning_type full \
    --train_from_scratch true \
    --dataset smoke_text --dataset_dir data --template default \
    --cutoff_len 48 --max_steps 1 \
    --per_device_train_batch_size 2 \
    --learning_rate 1.0e-3 \
    --bf16 false --fp16 false --flash_attn disabled \
    --disable_gradient_checkpointing true \
    --patch_method depth_attention \
    --cross_layer_mode depth_softmax \
    --depth_attention_stride 2 \
    --depth_attention_recent_window 0 \
    --output_dir saves/smoke/depth_attention_tiny
```

The equivalent YAML entry point is:

```bash
llamafactory-cli train examples/train_depth_attention_tiny.yaml
```

For a vanilla reference, use `--patch_method vanilla` and remove the
Depth-Attention-specific flags.

---

## Smaller-compute companion - Llama-410M on smallpile

The repository ships 410M-class GQA4x configs and one launcher per method. This
is the easiest way to sanity-check Depth-Attention against the included
cross-layer baselines before running larger paper-scale experiments.

The companion scripts expect a tokenized `smallpile` directory:

```bash
mkdir -p data/tokenized_data
huggingface-cli download hyq718/uint16smallpile \
    --repo-type dataset --local-dir data/tokenized_data/smallpile
```

Launch Depth-Attention:

```bash
export MODEL_CFG=llama_config/depth_attention_410m
export TOKENIZER=/path/to/your/tokenizer
export TOKENIZED=data/tokenized_data/smallpile
export OUTPUT_DIR=saves/smallpile/depth_attention_410m

bash scripts/train_depth_attention_llama_410m.sh
```

Run the comparison baselines:

```bash
bash scripts/train_attnres_llama_410m.sh
bash scripts/train_denseformer_llama_410m.sh
bash scripts/train_mhc_llama_410m.sh
```

Each script picks up `MODEL_CFG`, `TOKENIZER`, `TOKENIZED`, and `OUTPUT_DIR`, so
you can redirect configs, tokenizer path, data, or checkpoint location without
editing the scripts.

---

## Reproducing paper-style runs

The paper evaluates Depth-Attention on Qwen3-style decoders from 360M to 3B
parameters and compares against vanilla Transformers and strong cross-layer
baselines. This release focuses on the Llama-Factory training path and the
reference Llama modeling variants used to reproduce the method-level behavior.

For Depth-Attention, the key config fields are:

```json
{
  "recurrent_model": true,
  "cross_layer_pattern": "depth_softmax",
  "cross_layer_mode": "depth_softmax",
  "depth_softmax_stride": 16,
  "depth_recent_window": 0,
  "use_qk_norm": true
}
```

The public 410M config uses GQA4x:

```json
{
  "num_attention_heads": 16,
  "num_key_value_heads": 4
}
```

This exercises the native-KV GQA path used by the reference implementation.

---

## Hyperparameters

| Flag | Default | Method | Meaning |
|---|---:|---|---|
| `--patch_method` | unset | all | Which Llama implementation to swap in. |
| `--cross_layer_mode` | config / `depth_softmax` | Depth-Attention | Depth scoring/mixing mode. This release supports `depth_softmax`. |
| `--depth_attention_stride` | config / half the layer count | Depth-Attention | Select every Nth previous layer for depth mixing. |
| `--depth_attention_recent_window` | config / `0` | Depth-Attention | Always include this many most-recent previous layers. |
| `--use_qk_norm` | config | Depth-Attention / baselines | Override q/k head RMSNorm. |
| `--attnres_block_size` | config / `12` | AttnRes | Block size for the block residual-attention baseline. |
| `--attnres_recency_bias_init` | config / `3.0` | AttnRes | Initial recency bias for the AttnRes aggregator. |
| `--denseformer_dwa_dilation` | config / `1` | DenseFormer | Depth-weighted averaging dilation. |
| `--denseformer_dwa_period` | config / `1` | DenseFormer | Depth-weighted averaging period. |
| `--residual_baseline_num_streams` | config / `4` | mHC | Number of residual streams. |
| `--train_from_scratch` | `False` | all | Random-init from config rather than loading pretrained weights. |

Everything else - optimizer, scheduler, DeepSpeed, FlashAttention, dataset
flags - is plain LLaMA-Factory. See the scripts for the values used in the
included 410M-class runs.

---

## Inference

For Llama checkpoints trained from this repo, patch the class before loading:

```python
from llamafactory.model.llama_patch import patch_llama_depth_attention

patch_llama_depth_attention()

from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("/path/to/tokenizer")
model = AutoModelForCausalLM.from_pretrained(
    "/path/to/depth_attention_checkpoint",
    trust_remote_code=True,
)
```

The Llama path supports the standard Hugging Face cache interface, so
`model.generate(..., use_cache=True)` works as usual for supported checkpoints.

For baseline checkpoints, use the matching patch function:

```python
from llamafactory.model.llama_patch import patch_llama_denseformer

patch_llama_denseformer()

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "zeng123/3b-denseformer-qknorm-gqa-4x",
    trust_remote_code=True,
)
```

For mHC checkpoints, use `patch_llama_mhc()` and the corresponding
`zeng123/*-mhc-qknorm-gqa-4x` checkpoint.

---

## Development checks

```bash
python -m compileall -q src tests
python -m pytest -q
```

The test suite covers the depth-softmax formula, tiny GQA forward/backward,
save/load, and `AutoModelForCausalLM` patch selection for all four modeling
files. Torch-dependent tests are skipped automatically when PyTorch is
unavailable.

---

## Citation

```bibtex
@article{zeng2026depthattention,
  title={Depth-Attention: Cross-Layer Value Mixing for Language Models},
  author={Zeng, Boyi and Hao, Yiqin and Wang, Zitong and Song, Shixiang and Li, He and Song, Feichen and Liu, Yifan and He, Ziwei and Wang, Xinbing and Lin, Zhouhan},
  journal={arXiv preprint arXiv:2606.05014},
  year={2026}
}
```

---

## Acknowledgements

Built on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
(Apache-2.0).

## Contact

Open a GitHub issue for questions or reproduction problems.
