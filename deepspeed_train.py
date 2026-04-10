import json
import os
import sys
from PIL import Image

from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, \
    LlavaOnevisionForConditionalGeneration, LlavaForConditionalGeneration, MllamaForConditionalGeneration, \
    LlavaNextForConditionalGeneration, Trainer, Qwen2_5_VLForConditionalGeneration, Gemma3ForConditionalGeneration, \
    InternVLForConditionalGeneration, AutoModelForCausalLM, DINOv3ViTModel
from pathlib import Path
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import transformers

from models.lingshu_with_dinov3 import Lingshu_with_Dinov3_ForConditionalGeneration
from models.lingshu_with_usfm import Lingshu_with_USFM_ForConditionalGeneration
from models.lingshu_multi_vision_qformer import Multi_Vision_Qformer_ForConditionalGeneration
from models.internvl_3_5_moe import InternVL_MOE_ForConditionalGeneration
from models.lingshu_multi_vision import Multi_Vision_ForConditionalGeneration
from models.internvl_3_5_moe_usfm_512 import InternVL_MOE_USFM_512_ForConditionalGeneration
from collators.datacollator_apply_chat_template import CustomDataCollatorApplyChatTemplate, InternVLProcessor_Custom
from arguments import ModelArguments, DataArguments, TrainingArguments, LoraArguments
from collators.datasetV2 import LazySupervisedDataset
from models.ultrasound_moe import UltrasoundMOEForConditionalGeneration
from models.qwen2_5_vl_moe import Qwen2_5_VL_MOE_ForConditionalGeneration
from models.ovis2_5.modeling_ovis2_5 import Ovis2_5
from models.qwen2_vl_continued_moe import Qwen2VLMOEForConditionalGeneration
from utils import (
    rank0_print, find_all_linear_names, safe_save_model_for_hf_trainer,
)

#之前的transformers的版本4.56.2
def check_images():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments)
    )
    model_args, data_args = \
        parser.parse_args_into_dataclasses()

    error_log_file = "image_load_errors.txt"

    with open(error_log_file, "a", encoding="utf-8") as log_file:
        # 如果是文件夹路径，遍历所有JSON文件
        all_data_files = []

        for path in data_args.train_data_path:
            if os.path.isdir(path):  # 如果是文件夹
                # 遍历文件夹下所有JSON文件
                for file_name in os.listdir(path):
                    if file_name.lower().endswith('.json'):
                        file_path = os.path.join(path, file_name)
                        all_data_files.append(file_path)
            else:  # 如果是文件
                all_data_files.append(path)

        print(f"总共找到 {len(all_data_files)} 个JSON文件")

        total_original = 0
        total_kept = 0
        # 新增：统计图片数量不匹配的样本数
        total_image_mismatch = 0

        for data_file in all_data_files:
            print(f"正在处理: {data_file}")
            with open(data_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            original_count = len(data)  # 记录原始样本数
            total_original += original_count

            removed_cnt = 0
            kept = []  # 过滤后保留的样本
            # 新增：记录当前文件中图片数量不匹配的样本数
            file_image_mismatch = 0

            for item in tqdm(data, desc=f"Filtering {os.path.basename(data_file)}", file=sys.stdout):
                # 1. 决定用哪组对话
                conversations = (
                        item.get("conversations") or
                        (item.get("Alignment_VQA_conversations")
                         if model_args.training_stage == 'stage1' else None) or
                        (item.get("Instruction-Tuning_VQA_conversations")
                         if model_args.training_stage == 'stage2' else None)
                )
                if conversations is None:
                    log_file.write(f"skip_no_conv: {data_file} - {item}\n")
                    removed_cnt += 1
                    continue

                # 2. 检查是否有图片，没有图片的纯文本数据直接保留
                images = item.get('image')
                if images is None or images == []:
                    # 没有图片，纯文本数据，直接保留
                    kept.append(item)
                    continue

                # 3. 有图片的情况下，统一成 list
                if isinstance(images, str):
                    images = [images]
                actual_image_count = len(images)  # 实际图片数量

                # 4. 统计对话中<image>标签的总数
                total_image_tags = 0
                for conv in conversations:
                    if isinstance(conv, dict) and "from" in conv and "value" in conv and conv["from"] == "human":
                        total_image_tags += conv["value"].count("<image>")

                # 5. 验证图片数量与标签数量是否匹配
                if total_image_tags != actual_image_count:
                    log_file.write(
                        f"image_count_mismatch: {data_file} - 对话中<image>标签数为{total_image_tags}, "
                        f"实际图片数为{actual_image_count}\n"
                    )
                    removed_cnt += 1
                    total_image_mismatch += 1
                    file_image_mismatch += 1
                    continue

                # 6. 逐张图片试读
                valid = True
                for img_path in images:
                    # 尝试多种可能的图片路径组合
                    possible_paths = [
                        os.path.join(data_args.image_folder, img_path),  # 原始路径
                        os.path.join(data_args.image_folder, os.path.basename(img_path)),  # 只取文件名
                        img_path,  # 完整路径（如果已经是完整路径）
                        os.path.join(os.path.dirname(data_file), img_path),  # 相对于当前JSON文件的路径
                    ]

                    img_found = False
                    full_path = None

                    for path in possible_paths:
                        if os.path.exists(path):
                            full_path = path
                            img_found = True
                            break

                    if not img_found:
                        log_file.write(f"image_not_found: {img_path} - tried: {possible_paths}\n")
                        valid = False
                        break

                    try:
                        with Image.open(full_path) as im:
                            im = im.convert("RGB")
                    except Exception as e:
                        log_file.write(f"bad_image: {full_path} - {e}\n")
                        valid = False
                        break

                if valid:
                    kept.append(item)
                else:
                    removed_cnt += 1

            # 5. 覆盖写回原始 JSON（默认注释掉，需要时取消注释）
            # with open(data_file, "w", encoding="utf-8") as f:
            #     json.dump(kept, f, ensure_ascii=False, indent=4)

            kept_count = len(kept)
            total_kept += kept_count

            print(f"{data_file} 过滤完成：")
            print(f"  原始样本数: {original_count}")
            print(f"  删除样本数: {removed_cnt}")
            print(f"  其中图片数量不匹配: {file_image_mismatch}")  # 新增统计
            print(f"  保留样本数: {kept_count}")
            print(f"  保留比例: {kept_count / original_count * 100:.2f}%")

        print(f"\n=== 总体统计 ===")
        print(f"总原始样本数: {total_original}")
        print(f"总保留样本数: {total_kept}")
        print(f"总删除样本数: {total_original - total_kept}")
        print(f"其中图片数量不匹配的样本数: {total_image_mismatch}")  # 新增统计
        print(f"总体保留比例: {total_kept / total_original * 100:.2f}%")
        print("所有文件处理完成！")

def should_enable_grad_for_stage(param_name,training_stage=None):
    """
    判断在 Stage 1 训练中，给定参数名是否应该开启梯度。

    Args:
        param_name (str): 模型参数的名称。

    Returns:
        bool: 如果参数名包含指定关键词，则返回 True，否则返回 False。
    """
    keywords=None
    if training_stage == 'stage1':
        keywords = [
            'moe', 'coefficient', 'visual.vision_selector', 'multi_modal_projector',
            'fusion_projector', 'dinov3_projector',
            'usfm_projector','fusion_projector','qformer','merger.mlp',

        ]
    elif training_stage == 'stage1_5':
        keywords = [
            'moe', 'coefficient', 'visual.vision_selector', 'multi_modal_projector',
            'fusion_projector',   'dinov3_weight','usfm_weight','image_weight','dinov3_projector',
            'usfm_projector','fusion_projector','qformer','merger.mlp',
            'usfm', 'dinov3','visual'
        ]
    # 检查参数名是否包含任何一个关键词
    for keyword in keywords:
        if keyword in param_name:
            return True
    return False

def train():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, LoraArguments,TrainingArguments)
    )
    #TrainingArguments
    model_args, data_args, lora_args,training_args = parser.parse_args_into_dataclasses()
    # model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()
    # training_args = TrainingArguments
    output_dir = getattr(training_args, 'output_dir', None)
    assert output_dir is not None, "output_dir is required"
    training_args.output_dir = os.path.join(output_dir, model_args.model_id)
    os.makedirs(training_args.output_dir, exist_ok=True)
    rank0_print(f'save_output_dir----------------------->{training_args.output_dir}')

    default_assistant_start_tokens = None
    default_assistant_end_tokens = None



    rank0_print("Loading model, tokenizer, processor...")
    if model_args.model_id == 'lingshu_with_dinov3':
        model = Lingshu_with_Dinov3_ForConditionalGeneration.from_pretrained(model_args.model_local_path,
                                                                      torch_dtype=torch.bfloat16,
                                                                      attn_implementation="flash_attention_2",
                                                                      # device_map='cuda:0',
                                                                      )
        if model_args.training_stage=='stage1':
            model.init_checkpoint_model(dinov3_weights_path=model_args.dinov3_weight,training_stage=model_args.training_stage)
            model.dinov3.to(model.device)
        default_assistant_start_tokens="<|im_start|>assistant\n"
        default_assistant_end_tokens="<|im_end|>\n"
    elif model_args.model_id == 'lingshu_with_usfm':
        model = Lingshu_with_USFM_ForConditionalGeneration.from_pretrained(model_args.model_local_path,
                                                                      torch_dtype=torch.bfloat16,
                                                                      attn_implementation="flash_attention_2",
                                                                      # device_map='cuda:0',
                                                                      )
        if model_args.training_stage=='stage1':
            model.init_checkpoint_model(usfm_weights_path=model_args.usfm_weight,training_stage=model_args.training_stage)
            model.usfm.to(model.device)
        default_assistant_start_tokens="<|im_start|>assistant\n"
        default_assistant_end_tokens="<|im_end|>\n"
    elif model_args.model_id == 'multi_vision' or model_args.model_id == 'multi_vision_qwen':
        model = Multi_Vision_ForConditionalGeneration.from_pretrained(model_args.model_local_path,
                                                                       torch_dtype=torch.bfloat16,
                                                                       attn_implementation="flash_attention_2",
                                                                       # device_map='cuda:0',
                                                                       )
        if model_args.training_stage == 'stage1':
            model.init_checkpoint_model(dinov3_weights_path=model_args.dinov3_weight,usfm_weights_path=model_args.usfm_weight,
                                        training_stage=model_args.training_stage)
            # model.usfm.to(model.device)
            # model.dinov3.to(model.device)
        default_assistant_start_tokens="<|im_start|>assistant\n"
        default_assistant_end_tokens="<|im_end|>\n"
    elif model_args.model_id == 'multi_vision_qformer':
        model = Multi_Vision_Qformer_ForConditionalGeneration.from_pretrained(model_args.model_local_path,
                                                                       torch_dtype=torch.bfloat16,
                                                                       attn_implementation="flash_attention_2",
                                                                       # device_map='cuda:0',
                                                                       )
        model.init_checkpoint_model(dinov3_weights_path=model_args.dinov3_weight,usfm_weights_path=model_args.usfm_weight)
        # usfm_checkpoint = torch.load('/data/scy/SCY/Model_weights/USFM_latest.pth', map_location="cpu")
        # model.model.usfm.load_state_dict(usfm_checkpoint,strict=False)
        default_assistant_start_tokens="<|im_start|>assistant\n"
        default_assistant_end_tokens="<|im_end|>\n"
    elif model_args.model_id == 'InternVL3_5_MOE':
        model = InternVL_MOE_ForConditionalGeneration.from_pretrained(model_args.model_local_path,
                                                                       torch_dtype=torch.bfloat16,
                                                                       attn_implementation="flash_attention_2",
                                                                       device_map='cuda',
                                                                       )
        model.model.dinov3 = DINOv3ViTModel.from_pretrained('/data/scy/SCY/Model_weights/dinov3-vitl16-pretrain-lvd1689m')
        # usfm_checkpoint = torch.load('/data/scy/SCY/Model_weights/USFM_latest.pth', map_location="cpu")
        # model.model.usfm.load_state_dict(usfm_checkpoint,strict=False)
        default_assistant_start_tokens="<|im_start|>assistant\n"
        default_assistant_end_tokens="<|im_end|>\n"
    elif model_args.model_id == 'InternVL3_5_MOE_USFM_512':
        model = InternVL_MOE_USFM_512_ForConditionalGeneration.from_pretrained(model_args.model_local_path,
                                                                       torch_dtype=torch.bfloat16,
                                                                       attn_implementation="flash_attention_2",
                                                                       # device_map='cuda',
                                                                       )
        model.model.dinov3=model.model.dinov3.from_pretrained('/data/scy/SCY/Model_weights/dinov3-vitl16-pretrain-lvd1689m')
        model.model.usfm = model.model.usfm.from_pretrained(
            '/data/scy/SCY/SonoVLM_V2/data_generation/Miscellaneous_data/dinov3_usfm_final')
        # usfm_checkpoint = torch.load('/data/scy/SCY/Model_weights/USFM_latest.pth', map_location="cpu")
        # model.model.usfm.load_state_dict(usfm_checkpoint,strict=False)
        default_assistant_start_tokens="<|im_start|>assistant\n"
        default_assistant_end_tokens="<|im_end|>\n"
    elif model_args.model_id == 'InternVL3_5':
        model = InternVLForConditionalGeneration.from_pretrained(model_args.model_local_path,
                                                                       torch_dtype=torch.bfloat16,
                                                                       attn_implementation="flash_attention_2",
                                                                       # device_map='cuda',
                                                                       )
        default_assistant_start_tokens="<|im_start|>assistant\n"
        default_assistant_end_tokens="<|im_end|>\n"
    elif model_args.model_id == 'Ovis2_5':
        model = Ovis2_5.from_pretrained(model_args.model_local_path,
                                                     torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True,
                                                     attn_implementation="flash_attention_2",
                                                     device_map='cuda',
                                                     )
        default_assistant_start_tokens="<|im_start|>assistant\n"
        default_assistant_end_tokens="<|im_end|>\n"
    elif model_args.model_id == 'Qwen2_5_VL_MOE':
        model = Qwen2_5_VL_MOE_ForConditionalGeneration.from_pretrained(model_args.model_local_path,
                                                                       torch_dtype=torch.bfloat16,
                                                                       attn_implementation="flash_attention_2",
                                                                       device_map='cuda',
                                                                       )
    elif model_args.model_id == 'Qwen2_5_VL' or model_args.model_id =='Lingshu' or model_args.model_id =='HuatuoGPT-Vision':
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_args.model_local_path,
                                                                        torch_dtype=torch.bfloat16,
                                                                        attn_implementation="flash_attention_2",
                                                                        # device_map='cuda',
                                                                        )
        default_assistant_start_tokens="<|im_start|>assistant\n"
        default_assistant_end_tokens="<|im_end|>\n"

    elif model_args.model_id == 'LLaVA-OneVision':
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_args.model_local_path,
                                                                       torch_dtype=torch.bfloat16,
                                                                       attn_implementation="flash_attention_2",
                                                                       # device_map='cuda',
                                                                       )
    elif model_args.model_id == 'LLaVA-Med':
        model = LlavaForConditionalGeneration.from_pretrained(model_args.model_local_path,
                                                              torch_dtype=torch.bfloat16,
                                                              attn_implementation="flash_attention_2",
                                                              # device_map='cuda',
                                                              )
        default_assistant_start_tokens="[/INST]"
        default_assistant_end_tokens="</s>"
    elif model_args.model_id == 'LLaVA':
        model = LlavaNextForConditionalGeneration.from_pretrained(model_args.model_local_path,
                                                                  torch_dtype=torch.bfloat16,
                                                                  attn_implementation="flash_attention_2",
                                                                  # device_map='cuda',
                                                                  )
        default_assistant_start_tokens="[/INST]"
        default_assistant_end_tokens="</s>"
    elif model_args.model_id == 'Qwen2-VL':  # 2B and 7B
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_args.model_local_path,
                                                                torch_dtype=torch.bfloat16,
                                                                attn_implementation="flash_attention_2",
                                                                # device_map='cuda:3',
                                                                )
        default_assistant_start_tokens="<|im_start|>assistant\n"
        default_assistant_end_tokens="<|im_end|>\n"
    elif model_args.model_id == 'Llama-3.2-Vision':
        model = MllamaForConditionalGeneration.from_pretrained(model_args.model_local_path,
                                                               torch_dtype=torch.bfloat16,
                                                               attn_implementation="flash_attention_2",
                                                               # device_map='cuda',
                                                               )
    elif model_args.model_id == 'Qwen2-VL-MOE':
        model = Qwen2VLMOEForConditionalGeneration.from_pretrained(model_args.model_local_path,
                                                                 torch_dtype=torch.bfloat16,
                                                                 attn_implementation="flash_attention_2",
                                                                 # device_map='cuda',
                                                                 )
    elif model_args.model_id == 'Ultrasound-MOE':
        model = UltrasoundMOEForConditionalGeneration.from_pretrained(model_args.model_local_path,
                                                                   torch_dtype=torch.bfloat16,
                                                                   attn_implementation="flash_attention_2",
                                                                   # device_map='cuda',
                                                                   )
        model.initialize_vision_selector(model_args.model_local_path)
    elif model_args.model_id == 'Medgemma':
        model = Gemma3ForConditionalGeneration.from_pretrained(model_args.model_local_path,
                                                                   torch_dtype=torch.bfloat16,
                                                                   attn_implementation="flash_attention_2",
                                                                   # device_map='cuda:3',
                                                                   )
        default_assistant_start_tokens="<start_of_turn>model\n"
        default_assistant_end_tokens="<end_of_turn>\n"
    elif model_args.model_id == 'InternVL3':
        model = InternVLForConditionalGeneration.from_pretrained(model_args.model_local_path,
                                                                   torch_dtype=torch.bfloat16,
                                                                   attn_implementation="flash_attention_2",
                                                                   # device_map='cuda',
                                                                   )


    if model_args.training_stage == 'stage1':
        model.requires_grad_(False)
        rank0_print("Training in Stage 1 ...")
        for name, param in model.named_parameters():
            if should_enable_grad_for_stage(name,model_args.training_stage):
                param.requires_grad = True
    elif model_args.training_stage == 'stage1_5':
        model.requires_grad_(False)
        rank0_print("Training in Stage 1.5 ...")
        for name, param in model.named_parameters():
            if should_enable_grad_for_stage(name,model_args.training_stage):
                param.requires_grad = True
    elif model_args.training_stage == 'stage2':
        rank0_print("Training in Stage 2 ...")

        # 假设你有一个 lora_args.use_lora 的配置项来控制是否启用 LoRA
        if lora_args.use_lora: # 只有在启用 LoRA 时才进行 LoRA 配置
            rank0_print("Training lora parameters.")
            model.requires_grad_(False)
            named_modules = {n: m for n, m in model.named_modules()}
            lora_modules = []
            if lora_args.train_vision_encoder_lora:
                rank0_print("LoRA for vision encoder enabled...")
                lora_modules.extend(find_all_linear_names(named_modules, lora_args.lora_vision_encoder_keys))
            if lora_args.train_llm_lora:
                rank0_print("LoRA for LLM enabled...")
                lora_modules.extend(find_all_linear_names(named_modules, lora_args.lora_llm_keys))
            if lora_args.train_vision_projector_lora:
                rank0_print("LoRA for Projector enabled...")
                lora_modules.extend(find_all_linear_names(named_modules, lora_args.lora_vision_projector_keys))

            lora_config = LoraConfig(
                r=lora_args.lora_r,
                lora_alpha=lora_args.lora_alpha,
                target_modules=lora_modules,
                lora_dropout=lora_args.lora_dropout,
                bias=lora_args.lora_bias,
                task_type="CAUSAL_LM",
            )

            if lora_args.q_lora:
                model = prepare_model_for_kbit_training(
                    model, use_gradient_checkpointing=training_args.gradient_checkpointing
                )

            model = get_peft_model(model, lora_config)
            rank0_print(f"LoRA applied. Training {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters.")

            # 处理特定模块（如 MoE 相关）的梯度设置
            for name, param in model.named_parameters():
                if 'moe' in name or 'coefficient' in name or 'visual.vision_selector' in name:
                    param.requires_grad = True
                    rank0_print(f"Enabled gradient for specific module: {name}")

        else: # 不启用 LoRA，训练全参数
            rank0_print("Training full parameters.")
            model.requires_grad_(True)
    elif model_args.training_stage == 'stage3':
        rank0_print("Training in Stage 3...")
        for name, param in model.named_parameters():
            if 'moe' in name or 'coefficient' in name:
                param.requires_grad = True

    rank0_print(f'***********************{model_args.model_id}*************************')
    rank0_print(model)
    rank0_print("Trainable parameters:")
    training_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            rank0_print(f"\t{name}")
            # print(name)
            training_params += param.numel()
        total_params += param.numel()
    rank0_print(f"training parameters:{training_params / 1e9:.4f}B")
    rank0_print(f"total_params parameters:{total_params / 1e9:.4f}B")

    model.enable_input_require_grads()
    # load data
    processor=None
    if model_args.model_id == 'Ovis2_5':
        processor = (model.visual_tokenizer,model.text_tokenizer)
    elif model_args.model_id == 'InternVL3_5':
        processor = InternVLProcessor_Custom.from_pretrained(model_args.auto_processor_local_path)
    else:
        processor = AutoProcessor.from_pretrained(model_args.auto_processor_local_path)
    # 这里是做短上下文的
    # tokenizer.model_max_length = training_args.model_max_length
    # tokenizer.padding_side = 'right'

    rank0_print("Loading data...")
    train_dataset = LazySupervisedDataset(
        data_path=data_args.train_data_path,
        image_folder=data_args.image_folder,
        video_folder=data_args.video_folder,
        num_frames=data_args.num_frames,
        user_key=data_args.user_key,
        assistant_key=data_args.assistant_key,
        training_stage=model_args.training_stage,
        # processor=processor,
    )
    if data_args.eval_data_path:
        eval_dataset = LazySupervisedDataset(
        data_path=data_args.eval_data_path,
        image_folder=data_args.image_folder,
        video_folder=data_args.video_folder,
        num_frames=data_args.num_frames,
        user_key=data_args.user_key,
        assistant_key=data_args.assistant_key,
        # processor=processor,
        )
    else:
        eval_dataset = None
        training_args.eval_strategy = "no"

    # data collator
    data_collator = CustomDataCollatorApplyChatTemplate(model_args=model_args,
                                                        default_assistant_start_tokens=default_assistant_start_tokens,
                                                        default_assistant_end_tokens=default_assistant_end_tokens,
                                                        auto_processor_local_path=model_args.auto_processor_local_path)
    # trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train(resume_from_checkpoint=False)
    torch.cuda.synchronize()
    trainer.save_state()
    trainer.save_model(training_args.output_dir)

    # model.save_pretrained(training_args.output_dir)
    if lora_args.use_lora:
        base_model = trainer.model.merge_and_unload()
        base_model.save_pretrained(os.path.join(training_args.output_dir, 'merged_full_parameters'))



if __name__ == "__main__":
    # check_images()
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ["NCCL_P2P_LEVEL"] = "NVL"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    train()
