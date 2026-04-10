#!/bin/bash
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NNODES=1 \
NODE_RANK=0 \
NPROC_PER_NODE=8 \
MAX_PIXELS=132496 \
ROOT_IMAGE_DIR=/XYFS01/HOME/sysu_ldchen/sysu_ldchen_1/VLM/01DATA \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --custom_register_path /XYFS01/HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/swift_part/my_register.py \
    --model  /XYFS01/HOME/sysu_ldchen/sysu_ldchen_1/VLM/Model_weights/Lingshu-7B \
    --model_type qwen2_5_vl \
    --template qwen2_5_vl \
    --train_type lora \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --freeze_vit false \
    --freeze_llm false \
    --freeze_aligner false \
    --dataset  ablation_data_percentage_70 \
    --attn_impl flash_attention_2 \
    --torch_dtype bfloat16 \
    --load_from_cache_file True \
    --split_dataset_ratio 0.0 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 8 \
    --save_steps 30 \
    --logging_steps 1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 16 \
    --output_dir /XYFS01/HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/checkpoints/ablation/ablation_data/ablation_data_percentage_70 \
    --dataset_num_proc 128 \
    --deepspeed zero2 \
    --max_length 10240 \
    --use_dora True \
    --padding_free True \
    --packing True \
    --save_total_limit 5 \
#    --resume_from_checkpoint /XYFS01/HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/checkpoints/simple/compare/stage2/huatuo/v2-20251106-012616/checkpoint-200 \
#    --lora_rank 8 \
#    --lora_alpha 32 \
#    --target_modules all-linear \