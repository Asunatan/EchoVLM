from swift.llm import sft_main, TrainArguments
import my_register
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["ROOT_IMAGE_DIR"]="/data/scy/SCY/my_vlm/dataset"
if __name__ == '__main__':
    os.environ['MAX_PIXELS'] = '112896'
    sft_main(TrainArguments(
        model='/data/scy/SCY/Model_weights/Qwen2.5-VL-3B-Instruct',
        dataset=['Ultrasound_train'],
        val_dataset= ['Ultrasound_val'],
        # model_type='lingshu_with_donov3_and_usfm',
        # template='lingshu_with_donov3_and_usfm_template',
        model_type='qwen2_5_vl',
        template='qwen2_5_vl',
        load_from_cache_file=True,
        # split_dataset_ratio=0.0,
        train_type='lora',
        freeze_vit=False,
        freeze_llm=True,
        freeze_aligner=False,
        torch_dtype='bfloat16',
        attn_impl='flash_attention_2',
        # padding_free=True,
        # packing=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=1e-4,
        lora_rank=8,
        lora_alpha=32,
        target_modules='all-linear',
        gradient_accumulation_steps=5,
        eval_steps=100,
        save_steps=100,
        save_total_limit=None,
        logging_steps=5,
        max_length=None,
        output_dir='output',
        warmup_ratio=0.05,
        dataloader_num_workers=8,
        dataset_num_proc=8,
    ))