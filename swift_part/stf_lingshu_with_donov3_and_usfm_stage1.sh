#!/bin/bash
NNODES=1 \
NODE_RANK=0 \
NPROC_PER_NODE=3 \
CUDA_VISIBLE_DEVICES=0,1,2 \
swift sft \
    --custom_register_path /data/scy/SCY/SonoVLM_V2/swift_part/my_register.py \
    --model /data/scy/SCY/Model_weights/Qwen2.5-VL-3B-Instruct \
    --train_type full \
    --freeze_vit true \
    --freeze_llm true \
    --freeze_aligner false \
    --model_type lingshu_with_donov3_and_usfm \
    --template lingshu_with_donov3_and_usfm_template \
    --attn_impl flash_attention_2 \
    --torch_dtype bfloat16 \
    --dataset Ultrasound \
    --load_from_cache_file True \
    --split_dataset_ratio 0.00 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 16 \
    --eval_steps -1 \
    --save_steps 1000 \
    --save_total_limit None \
    --logging_steps 1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --output_dir swift_output/stage1/lingshu_with_donov3_and_usfm \
    --dataset_num_proc 8 \
    --deepspeed zero3 \
#    --packing true \
#    --padding_free true \
#    --max_length 2048 \
#    --lora_rank 8 \
#    --lora_alpha 32 \
#    --target_modules all-linear \