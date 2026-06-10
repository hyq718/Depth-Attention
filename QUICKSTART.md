# Quickstart

This guide starts from a fresh clone and ends with a tiny local training smoke
test plus the larger 410M-class launchers.

## 1. Install

```bash
git clone https://github.com/LUMIA-Group/Depth-Attention.git
cd Depth-Attention
conda create -n depth-attention python=3.10 -y
conda activate depth-attention
pip install -e ".[torch,metrics,deepspeed]"
```

Install a matching FlashAttention build if you plan to use `--flash_attn fa2`.

## 2. Local Smoke Test

```bash
bash scripts/quickstart_depth_attention_tiny.sh
```

This trains a tiny GQA Depth-Attention Llama model for one step using only
`data/smoke_text.jsonl` and `data/tiny_tokenizer`.

## 3. Larger Runs

Prepare a tokenizer and tokenized dataset, then launch:

```bash
export MODEL_CFG=llama_config/depth_attention_410m
export TOKENIZER=/path/to/tokenizer
export TOKENIZED=data/tokenized_data/smallpile
export OUTPUT_DIR=saves/smallpile/depth_attention_410m

bash scripts/train_depth_attention_llama_410m.sh
```

Baselines:

```bash
bash scripts/train_attnres_llama_410m.sh
bash scripts/train_denseformer_llama_410m.sh
bash scripts/train_mhc_llama_410m.sh
```

Depth-Attention overrides:

```bash
export STRIDE=4
export RECENT_WINDOW=0
```
