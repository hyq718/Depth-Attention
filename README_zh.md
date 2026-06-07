# Depth-Attention

Depth-Attention 是一个 LLaMA-Factory 风格的预训练仓库，用于训练
Depth-Attention / depth-softmax LLaMA 以及 AttnRes、DenseFormer、mHC
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
  "cross_layer_pattern": "depth_softmax",
  "cross_layer_mode": "depth_softmax",
  "depth_softmax_stride": 16,
  "depth_recent_window": 0
}
```

CLI 兼容别名：

```bash
--depth_attention_stride 16
--depth_attention_recent_window 0
```

## 目录结构

- `src/llamafactory/model/modeling/modeling_llama_depth_attention.py`：Depth-Attention 主实现。
- `src/llamafactory/model/modeling/modeling_llama_attnres.py`：Block AttnRes baseline。
- `src/llamafactory/model/modeling/modeling_llama_denseformer.py`：DenseFormer baseline。
- `src/llamafactory/model/modeling/modeling_llama_mhc.py`：mHC baseline。
- `llama_config/`：tiny 和 410M-class GQA4x 配置。
- `scripts/`：quickstart 和各方法训练脚本。
- `examples/deepspeed/`：ZeRO 配置。

## 检查

```bash
python -m compileall -q src tests
python -m pytest -q
```

本仓库基于 LLaMA-Factory。
