#!/bin/bash
NNODES=1 \
NODE_RANK=0 \
NPROC_PER_NODE=4 \
MAX_PIXELS=112896 \
ROOT_IMAGE_DIR=/XYFS01/HOME/sysu_ldchen/sysu_ldchen_1/VLM/01DATA \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --custom_register_path /XYFS01/HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/swift_part/my_register_a800.py \
    --model  /HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/checkpoints/simple/stage1_pubmed_only_us_stage1_5_all_4k/Lingshu \
    --model_type qwen2_5_vl \
    --template qwen2_5_vl \
    --train_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit false \
    --freeze_llm false \
    --freeze_aligner false \
    --dataset cot_data_caption_report \
    --max_pixels 112896 \
    --attn_impl flash_attention_2 \
    --torch_dtype bfloat16 \
    --load_from_cache_file True \
    --split_dataset_ratio 0.0 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 8 \
    --save_steps 10 \
    --save_total_limit 10 \
    --logging_steps 1 \
    --warmup_ratio 0.1 \
    --dataloader_num_workers 8 \
    --output_dir /HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/checkpoints/simple/stage3/Lingshu_stage3_cot_complex_caption_report_dpo \
    --dataset_num_proc 128 \
    --deepspeed zero3 \
    --use_dora true \
    --max_length 10240 \
    --resume_from_checkpoint /HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/checkpoints/simple/stage3/our_dpo/v3-20251117-152845/checkpoint-123 \
    --resume_only_model True \
    --ignore_data_skip True \
    --packing true \
    --padding_free True \

#    --lora_rank 8 \
#    --lora_alpha 32 \
#    --target_modules all-linear \