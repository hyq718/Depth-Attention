#!/bin/bash
# Quickstart: one tiny Depth-Attention pretraining smoke run on local toy data.

set -e

cd "$(dirname "$0")/.."

OUTPUT_DIR=${OUTPUT_DIR:-saves/smoke/depth_attention_tiny}

llamafactory-cli train \
    --model_name_or_path llama_config/depth_attention_tiny \
    --tokenizer_name_or_path data/tiny_tokenizer \
    --stage pt \
    --do_train \
    --finetuning_type full \
    --train_from_scratch true \
    --dataset smoke_text \
    --dataset_dir data \
    --template default \
    --cutoff_len 48 \
    --max_steps 1 \
    --save_steps 1 \
    --logging_steps 1 \
    --overwrite_output_dir true \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1.0e-3 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --weight_decay 0.0 \
    --bf16 false \
    --fp16 false \
    --flash_attn disabled \
    --disable_gradient_checkpointing true \
    --preprocessing_num_workers 1 \
    --patch_method depth_attention \
    --cross_layer_mode depth_softmax \
    --depth_attention_stride 2 \
    --depth_attention_recent_window 0 \
    --output_dir "$OUTPUT_DIR"
