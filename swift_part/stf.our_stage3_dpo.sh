#!/bin/bash
NNODES=1 \
NODE_RANK=0 \
NPROC_PER_NODE=4 \
MAX_PIXELS=1048576 \
ROOT_IMAGE_DIR=/XYFS01/HOME/sysu_ldchen/sysu_ldchen_1/VLM/01DATA \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift rlhf \
    --rlhf_type dpo \
    --custom_register_path /XYFS01/HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/swift_part/my_register_a800.py \
    --model  /XYFS01/HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/checkpoints/simple/compare/Lingshu_compare_stage1_pubmed_only_us_stage1_5_20wr_publici_multicenterr/Lingshu \
    --model_type qwen2_5_vl \
    --template qwen2_5_vl \
    --train_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit false \
    --freeze_llm false \
    --freeze_aligner false \
    --dataset Ultrasound_dpo \
    --attn_impl flash_attention_2 \
    --torch_dtype bfloat16 \
    --load_from_cache_file True \
    --split_dataset_ratio 0.0 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 8 \
    --save_steps 10 \
    --save_total_limit 20 \
    --logging_steps 1 \
    --warmup_ratio 0.1 \
    --dataloader_num_workers 8 \
    --output_dir /XYFS01/HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/checkpoints/simple/stage3/our_dpo \
    --dataset_num_proc 128 \
    --deepspeed zero2 \
    --max_length 10240 \
    --resume_from_checkpoint /XYFS01/HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/checkpoints/simple/stage2/Lingshu_stage2_final/v7-20251111-140713/checkpoint-388 \
    --resume_only_model True \
    --ignore_data_skip True \
    --padding_free True \
    --packing true \
    --use_dora true \
    --loss_type sigmoid bco_pair sft \
    --loss_weights 0.8 0.2 1.0 \
    --rpo_alpha 0.0 \
#    --lora_rank 8 \
#    --lora_alpha 32 \
#    --target_modules all-linear \