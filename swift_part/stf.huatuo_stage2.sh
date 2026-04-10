#!/bin/bash
NNODES=1 \
NODE_RANK=0 \
NPROC_PER_NODE=4 \
MAX_PIXELS=112896 \
ROOT_IMAGE_DIR=/XYFS01/HOME/sysu_ldchen/sysu_ldchen_1/VLM/01DATA \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --custom_register_path /XYFS01/HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/swift_part/my_register_a800.py \
    --model  /XYFS01/HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/checkpoints/simple/compare/HuatuoGPT_compare_stage1_pubmed_only_us_stage1_5_20wr_publici_multicenterr/HuatuoGPT-Vision \
    --model_type qwen2_5_vl \
    --template qwen2_5_vl \
    --train_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit false \
    --freeze_llm false \
    --freeze_aligner false \
    --dataset  compare \
    --max_pixels 112896 \
    --attn_impl flash_attention_2 \
    --torch_dtype bfloat16 \
    --load_from_cache_file True \
    --split_dataset_ratio 0.0 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 8 \
    --save_steps 20 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --warmup_ratio 0.1 \
    --dataloader_num_workers 8 \
    --output_dir /XYFS01/HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/checkpoints/simple/compare/stage2/huatuo \
    --dataset_num_proc 8 \
    --deepspeed zero3 \
    --max_length 10240 \
#    --resume_from_checkpoint /XYFS01/HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/checkpoints/simple/compare/stage2/huatuo/v2-20251106-012616/checkpoint-200 \
#    --lora_rank 8 \
#    --lora_alpha 32 \
#    --target_modules all-linear \