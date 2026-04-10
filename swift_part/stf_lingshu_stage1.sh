#!/bin/bash
NNODES=1 \
NODE_RANK=0 \
NPROC_PER_NODE=3 \
CUDA_VISIBLE_DEVICES=0,1,2 \
swift sft \
    --custom_register_path /data/scy/SCY/SonoVLM_V2/swift_part/my_register.py \
    --model  /data/scy/SCY/Model_weights/Qwen2.5-VL-3B-Instruct\
    --model_type qwen2_5_vl \
    --template qwen2_5_vl \
    --train_type full \
    --freeze_vit true \
    --freeze_llm true \
    --freeze_aligner false \
    --dataset Ultrasound \
    --max_pixels 153664 \
    --padding_free true \
    --attn_impl flash_attention_2 \
    --torch_dtype bfloat16 \
    --load_from_cache_file True \
    --split_dataset_ratio 0.0 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 16 \
    --eval_steps -1 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 16 \
    --output_dir swift_output/stage1/lingshu \
    --dataset_num_proc 64 \
    --deepspeed zero2 \
#    --packing true \
#    --max_length 2048 \
#    --lora_rank 8 \
#    --lora_alpha 32 \
#    --target_modules all-linear \