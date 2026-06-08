# Depth-Attention

[论文](https://arxiv.org/abs/2606.05014) / [PDF](https://arxiv.org/pdf/2606.05014)

Depth-Attention 是一个 LLaMA-Factory 风格的预训练仓库，用于训练
Depth-Attention LLaMA 以及 AttnRes、DenseFormer、mHC
三个对比 baseline。

## 快速开始

```bash
git clone https://github.com/LUMIA-Group/Depth-Attention.git
cd Depth-Attention
conda create -n depth-attention python=3.10 -y
conda activate depth-attention
pip install -e ".[torch,metrics,deepspeed]"
bash scripts/quickstart_depth_attention_tiny.sh
```

## 方法切换

`--patch_method` 支持：

```text
depth_attention
attnres
denseformer
mhc
vanilla
```

Depth-Attention 使用参考实现字段：

```json
{
  "recurrent_model": true,
  "use_depth_attention": true,
  "depth_attention_stride": 16,
  "depth_attention_recent_window": 0
}
```

CLI 兼容别名：

```bash
--depth_attention_stride 16
--depth_attention_recent_window 0
```

## 已开源 checkpoint

已训练 checkpoint 发布在 [zeng123](https://huggingface.co/zeng123)：

| 规模 | 方法 | Hugging Face | 模型配置 | 训练 recipe |
|---|---|---|---|---|
| 1.5B | DenseFormer | [zeng123/1.5b-denseformer-qknorm-gqa-4x](https://huggingface.co/zeng123/1.5b-denseformer-qknorm-gqa-4x) | `llama_config/released/1.5b_denseformer_qknorm_gqa4x` | `examples/released_models/train_1.5b_denseformer_qknorm_gqa4x.yaml` |
| 1.5B | mHC | [zeng123/1.5b-mhc-qknorm-gqa-4x](https://huggingface.co/zeng123/1.5b-mhc-qknorm-gqa-4x) | `llama_config/released/1.5b_mhc_qknorm_gqa4x` | `examples/released_models/train_1.5b_mhc_qknorm_gqa4x.yaml` |
| 3B | DenseFormer | [zeng123/3b-denseformer-qknorm-gqa-4x](https://huggingface.co/zeng123/3b-denseformer-qknorm-gqa-4x) | `llama_config/released/3b_denseformer_qknorm_gqa4x` | `examples/released_models/train_3b_denseformer_qknorm_gqa4x.yaml` |
| 3B | mHC | [zeng123/3b-mhc-qknorm-gqa-4x](https://huggingface.co/zeng123/3b-mhc-qknorm-gqa-4x) | `llama_config/released/3b_mhc_qknorm_gqa4x` | `examples/released_models/train_3b_mhc_qknorm_gqa4x.yaml` |

四个 checkpoint 都使用 QK norm 和 GQA4x。1.5B 配置为 48 层、hidden size
1536、24 个 attention heads、6 个 KV heads；3B 配置为 48 层、hidden size
2048、32 个 attention heads、8 个 KV heads。训练 recipe 根据 checkpoint
导出的 `config.json` 和 Trainer model card 整理；集群下载可设置
`HF_ENDPOINT=https://hf-mirror.com`。

## 目录结构

- `src/llamafactory/model/modeling/modeling_llama_depth_attention.py`：Depth-Attention 主实现。
- `src/llamafactory/model/modeling/modeling_llama_attnres.py`：Block AttnRes baseline。
- `src/llamafactory/model/modeling/modeling_llama_denseformer.py`：DenseFormer baseline。
- `src/llamafactory/model/modeling/modeling_llama_mhc.py`：mHC baseline。
- `llama_config/`：tiny、410M-class 和 released 1.5B/3B GQA4x 配置。
- `scripts/`：quickstart 和各方法训练脚本。
- `examples/deepspeed/`：ZeRO 配置。
- `examples/released_models/`：released checkpoint 的训练 recipe。

## 检查

```bash
python -m compileall -q src tests
python -m pytest -q
```

本仓库基于 LLaMA-Factory。
