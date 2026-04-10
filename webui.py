import argparse
import base64
import os
import sys
import subprocess
from pathlib import Path
import gradio as gr
from PIL import Image
from peft import PeftModel
from torchvision.transforms import v2 as V2
from models.qwen2_vl_continued_moe import Qwen2VLMOEForConditionalGeneration
from supported_models import MODEL_HF_PATH, MODEL_FAMILIES
import torch
from PIL import Image
import requests
from vllm import LLM, SamplingParams
from io import BytesIO
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, DataCollatorForSeq2Seq, \
    LlavaOnevisionForConditionalGeneration, MllamaForConditionalGeneration
from datasets import LazySupervisedDataset
from torch.utils.data import DataLoader
def launch_training(
    model_id, model_local_path, run_id, data_path, eval_data_path, image_folder, video_folder, num_frames,
    train_vision_encoder, use_vision_lora, train_vision_projector,
    use_lora, q_lora, lora_r, lora_alpha,
    ds_stage, per_device_batch_size, grad_accum, num_epochs,
    lr, model_max_len, num_gpus, use_tf32, num_workers, prefetch_factor
):
    # Construct the distributed args
    distributed_args = f"--nnodes=1 --nproc_per_node {num_gpus} --rdzv_backend c10d --rdzv_endpoint localhost:0"

    # Construct the command pip install --upgrade gradio
    cmd = [
        "torchrun",
        *distributed_args.split(),
        "train.py",
        f"--model_id={model_id}",
        f"--model_local_path={model_local_path}",
        f"--data_path={data_path}",
        f"--eval_data_path={eval_data_path}",
        f"--image_folder={image_folder}",
        f"--video_folder={video_folder}",
        f"--num_frames={num_frames}",
        f"--output_dir=./checkpoints/{run_id}",
        "--report_to=wandb",
        f"--run_name={run_id}",
        f"--deepspeed=./ds_configs/{ds_stage}.json",
        "--bf16=True",
        f"--num_train_epochs={num_epochs}",
        f"--per_device_train_batch_size={per_device_batch_size}",
        f"--per_device_eval_batch_size={per_device_batch_size}",
        f"--gradient_accumulation_steps={grad_accum}",
        "--eval_strategy=epoch",
        "--save_strategy=epoch",
        "--save_total_limit=1",
        f"--learning_rate={lr}",
        "--weight_decay=0.",
        "--warmup_ratio=0.03",
        "--lr_scheduler_type=cosine",
        "--logging_steps=1",
        f"--tf32={use_tf32}",
        f"--model_max_length={model_max_len}",
        "--gradient_checkpointing=True",
        f"--dataloader_num_workers={num_workers}",
        f"--dataloader_prefetch_factor={prefetch_factor}",
        f"--train_vision_encoder={train_vision_encoder}",
        f"--use_vision_lora={use_vision_lora}",
        f"--train_vision_projector={train_vision_projector}",
        f"--use_lora={use_lora}",
        f"--q_lora={q_lora}",
        f"--lora_r={lora_r}",
        f"--lora_alpha={lora_alpha}",
    ]

    # Run the command
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    
    # Stream the output
    output = ""
    for line in process.stdout:
        output += line
        yield output

    # Wait for the process to complete
    process.wait()

    if process.returncode == 0:
        yield output + "\nTraining completed successfully!"
    else:
        yield output + f"\nTraining failed with return code {process.returncode}"

textbox = gr.Textbox(
    show_label=False, placeholder="Enter text and press ENTER", container=False
)


class ModelInference:
    def __init__(self,fintune_model_dir,original_model_dir,
                 save_conversation_path,use_cache=True,checkpoint=None,use_vllm=True):
        # 初始化 vLLM 的 LLM 对象
        self.use_vllm = use_vllm
        self.use_cache = use_cache
        if use_vllm:
            self.model = LLM(
                model=fintune_model_dir,
                trust_remote_code=True,
                limit_mm_per_prompt={"image": 30},
                # max_model_len=32768,
                # tensor_parallel_size=4,  # 根据你的硬件调整
                dtype="bfloat16",
                seed=42,
            )
            self.sampling_params = SamplingParams(
                temperature=0.8,
                max_tokens=2048,
                repetition_penalty=1.1
                # stop=["<|im_sep|>", "<|im_end|>"]
            )
        else:
            if 'moe' in fintune_model_dir.lower():
                self.model = Qwen2VLMOEForConditionalGeneration.from_pretrained(original_model_dir,
                                                                                attn_implementation="flash_attention_2",
                                                                                device_map=None,
                                                                                torch_dtype=torch.bfloat16).eval()
                self.model.config.output_router_logits = False
                # self.model = PeftModel.from_pretrained(self.model, fintune_model_dir)
                self.model_name = 'moe'
            elif 'qwen' in fintune_model_dir.lower():
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(fintune_model_dir,
                                                                             attn_implementation="flash_attention_2",
                                                                             device_map="cuda:3",
                                                                             torch_dtype=torch.bfloat16).eval()
                self.model_name = 'qwen'
            elif 'LLaVA-Onevision-7B' in fintune_model_dir:
                self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(fintune_model_dir,
                                                                                    device_map="cuda:3",
                                                                                    attn_implementation="flash_attention_2",
                                                                                    torch_dtype=torch.bfloat16).eval()
                self.model_name = 'LLaVA-Onevision-7B'
            elif 'Llama-3.2-11B-Vision-Instruct' in fintune_model_dir:
                self.model = MllamaForConditionalGeneration.from_pretrained(fintune_model_dir,
                                                                            device_map="cuda:3",
                                                                            # attn_implementation="flash_attention_2",
                                                                            torch_dtype=torch.bfloat16).eval()
                self.model_name = 'Llama-3.2-11B-Vision-Instruct'
        #
        self.processor = AutoProcessor.from_pretrained(original_model_dir)
        # if 'Llama-3.2-11B-Vision-Instruct' in fintune_model_dir:
        #     self.processor.tokenizer.padding_side = 'left'
        self.chat_history = []  # 聊天历史
        self.image_transformer = V2.Compose([
            V2.ToPILImage(),
            # V2.RandomCrop(384,)
            V2.Resize((336,336)),
            # V2.AutoAugment(interpolation=InterpolationMode.BICUBIC),
            # V2.RandomHorizontalFlip(p=0.5)
        ])
        self.conversation_history=[]
        self.images_list = []


    def run_inference(self, text, image_files):
        # 这里是模型推理的逻辑
        # 示例代码，实际使用时替换为你的模型推理逻辑

        user_content = []
        user_content.append({"type": "text", "text": text})
        if image_files is not None:
            num_image = len(image_files)
            if num_image > 0:
                for image_path in image_files:
                    img = Image.open(image_path[0]).convert("RGB")
                    img = self.image_transformer(img)
                    self.images_list.append(img)
                    user_content.append({"type": "image"})
        self.conversation_history.append({
            "role": "user",
            "content": user_content
        })
        chat_text = self.processor.apply_chat_template(self.conversation_history, tokenize=False, add_generation_prompt=True)
        if self.use_vllm:
            outputs = self.model.generate({
                "prompt": chat_text,
                "multi_modal_data": {'image': self.images_list}
            }, self.sampling_params)
            output_text = outputs[0].outputs[0].text
        else:
            inputs = self.processor(text=[chat_text], images=self.images_list, padding=True,return_tensors="pt")
            inputs = inputs.to(self.model.device)
            if self.model_name == 'LLaVA-Onevision-7B':
                generated_ids = self.model.generate(**inputs, max_new_tokens=4096, use_cache=self.use_cache,
                                                    pad_token_id=self.processor.tokenizer.eos_token_id)
            else:
                generated_ids = self.model.generate(**inputs, use_cache=self.use_cache, max_new_tokens=4096)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True,
                                                      clean_up_tokenization_spaces=False)[0]
        self.conversation_history.append({
                            "role": "assistant",
                            "content": [{"type": "text", "text": output_text}]
                        })
        # 综合文本和图片的回复
        return output_text
    def process_input(self, image_files, message, chat_history):
        image_components = []
        if image_files:
            for img in image_files:
                with open(img[0], "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                # 添加可点击放大的图片组件
                img_html = f'''
                <div class="img-container" style="display: inline-block; margin: 5px;">
                    <img 
                        src="data:image/png;base64,{encoded_string}" 
                        style="
                            max-width: 300px; 
                            max-height: 300px;
                            cursor: zoom-in;
                            transition: transform 0.3s;
                        "
                        onclick="
                            this.style.transform = this.style.transform === 'scale(2)' ? 'scale(1)' : 'scale(2)';
                            this.style.zIndex = this.style.zIndex === '999' ? '1' : '999';
                        "
                    >
                </div>
                '''
                image_components.append(img_html)
        # 组合消息内容
        if image_components:
            full_message = f"{message}\n{''.join(image_components)}"
        else:
            full_message = message
        chat_history.append({"role": "user","content": full_message})

        model_response = self.run_inference(message, image_files)
        # 添加模型回复到聊天历史
        chat_history.append({"role": "assistant", "content":model_response})

        return "", [], chat_history  # 清空文本框和图片选择，返回更新后的聊天记录


    def upvote(self, chat_history):
        # 处理点赞逻辑
        return chat_history

    def downvote(self, chat_history):
        # 处理点踩逻辑
        return chat_history

    def flag(self, chat_history):
        # 处理标记逻辑
        return chat_history

    def regenerate(self, chat_history):
        # 处理重新生成回复逻辑
        if len(chat_history) > 0 and chat_history[-1][0] == "MedAI":
            # 重新生成最后一条回复
            user_message = chat_history[-2][1] if len(chat_history) > 1 else ""
            user_images = []  # 需要保存用户上传的图片路径
            model_response = self.run_inference(user_message, user_images)
            chat_history[-1] = ("MedAI", model_response)
        return chat_history

    def clear_history(self):
        # 清空聊天历史
        self.conversation_history = []
        self.images_list=[]
        return []






def create_ui():
    inference = ModelInference(fintune_model_dir=args.model_path,original_model_dir=args.original_model_path,
                               save_conversation_path=args.save_conversation_path)# 创建推理类实例
    with gr.Blocks(css="#container {max-width: 3200px; margin: auto;}") as ui:
        gr.Markdown("# Training GUI of Northwestern Polytechnical University and MedAI🚀", elem_id="title")#设置标题
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Model")
                model_id = gr.Dropdown(
                    choices=list(MODEL_HF_PATH.keys()),
                    value=list(MODEL_HF_PATH.keys())[0],
                    label="Model ID",
                    info="Select the model to be fine-tuned"
                )
                model_hf_path = gr.Textbox(
                    label="Model HuggingFace Path",
                    value=MODEL_HF_PATH.get(list(MODEL_HF_PATH.keys())[0], ""),
                    interactive=False,
                    info="Corresponding HuggingFace path"
                )
                model_local_path = gr.Textbox(
                    label="Model Local Path",
                    value="",
                    info="Local path to the model (optional; in case you want to do multiple rounds of finetuning)",
                )

            with gr.Column(scale=1):
                gr.Markdown("## LLM")
                with gr.Column():
                    use_lora = gr.Checkbox(
                        value=True,
                        label="Use LoRA",
                        info="Whether to use LoRA for LLM"
                    )
                    q_lora = gr.Checkbox(
                        value=False,
                        label="Use Q-LoRA",
                        info="Whether to use Q-LoRA for LLM; only effective when 'Use LoRA' is True"
                    )
                    lora_r = gr.Number(
                        value=8,
                        label="LoRA R",
                        info="The LoRA rank (both LLM and vision encoder)"
                    )
                    lora_alpha = gr.Number(
                        value=8,
                        label="LoRA Alpha",
                        info="The LoRA alpha (both LLM and vision encoder)"
                    )

            with gr.Column(scale=1):
                gr.Markdown("## Vision")
                train_vision_encoder = gr.Checkbox(
                    value=False,
                    label="Train Vision Encoder",
                    info="Whether to train the vision encoder"
                )
                use_vision_lora = gr.Checkbox(
                    value=False,
                    label="Use Vision LoRA",
                    info="Whether to use LoRA for vision encoder (only effective when 'Train Vision Encoder' is True)"
                )
                train_vision_projector = gr.Checkbox(
                    value=False,
                    label="Train Vision Projector",
                    info="Whether to train the vision projector (only full finetuning is supported)"
                )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Data")
                data_path = gr.Textbox(
                    value="./example_data/celeba_image_train.json",
                    label="Training Data Path",
                    info="Path to the training data json file"
                )
                eval_data_path = gr.Textbox(
                    value="./example_data/celeba_image_eval.json",
                    label="Evaluation Data Path",
                    info="Path to the evaluation data json file (optional)"
                )
                image_folder = gr.Textbox(
                    value="./example_data/images",
                    label="Image Folder",
                    info="Path to the image root folder"
                )
                video_folder = gr.Textbox(
                    value="./example_data/videos",
                    label="Video Folder",
                    info="Path to the video root folder"
                )
                num_frames = gr.Number(
                    value=8,
                    label="Number of Frames",
                    info="Frames sampled from each video"
                )
                        
            with gr.Column(scale=1):
                gr.Markdown("## Training")
                run_id = gr.Textbox(
                    value=f"{list(MODEL_HF_PATH.keys())[0]}_lora-True_qlora-False",
                    label="Run ID",
                    info="Unique identifier for this training run"
                )
                num_gpus = gr.Number(
                    value=1,
                    label="Number of GPUs",
                    info="Number of GPUs to use for distributed training"
                )
                per_device_batch_size = gr.Number(
                    value=2,
                    label="Per Device Batch Size",
                    info="Batch size per GPU"
                )
                grad_accum = gr.Number(
                    value=1,
                    label="Gradient Accumulation Steps",
                    info="Number of steps to accumulate gradients (effective batch size = per_device_batch_size * grad_accum)"
                )
                lr = gr.Number(
                    value=2e-5,
                    label="Learning Rate",
                    info="Learning rate for training"
                )
                num_epochs = gr.Number(
                    value=5,
                    label="Number of Epochs",
                    info="Number of training epochs"
                )

            with gr.Column(scale=1):
                gr.Markdown("## Training")
                num_workers = gr.Number(
                    value=4,
                    label="DataLoader Num Workers",
                    info="Number of workers for dataLoader"
                )
                prefetch_factor = gr.Number(
                    value=2,
                    label="DataLoader Prefetch Factor",
                    info="Number of batches prefetched by dataLoader"
                )
                model_max_len = gr.Number(
                    value=512,
                    label="Model Max Length",
                    info="Maximum input length of the model"
                )
                ds_stage = gr.Dropdown(
                    ["zero2", "zero3"],
                    value="zero3",
                    label="DeepSpeed Stage",
                    info="DeepSpeed stage; choose between zero2 and zero3"
                )
                use_tf32 = gr.Checkbox(
                    value=True,
                    label="Use TF32",
                    info="Whether to use TF32 precision (for Ampere+ GPUs)"
                )

        train_button = gr.Button("Start Training", variant="primary")
        output = gr.Textbox(label="Training Output", interactive=False)


        def update_hf_path(selected_model):
            return MODEL_HF_PATH.get(selected_model, "")

        model_id.change(update_hf_path, inputs=[model_id], outputs=[model_hf_path])
        
        def update_default_run_id(model_id, use_lora, q_lora):
            return f"{model_id}_lora-{use_lora}_qlora-{q_lora}"
        
        model_id.change(update_default_run_id, inputs=[model_id, use_lora, q_lora], outputs=[run_id])
        use_lora.change(update_default_run_id, inputs=[model_id, use_lora, q_lora], outputs=[run_id])
        q_lora.change(update_default_run_id, inputs=[model_id, use_lora, q_lora], outputs=[run_id])

        train_button.click(
            launch_training,
            inputs=[
                model_id, model_local_path, run_id, data_path, eval_data_path, image_folder, video_folder, num_frames,
                train_vision_encoder, use_vision_lora, train_vision_projector,
                use_lora, q_lora, lora_r, lora_alpha,
                ds_stage, per_device_batch_size, grad_accum, num_epochs,
                lr, model_max_len, num_gpus, use_tf32, num_workers, prefetch_factor
            ],
            outputs=output
        )
        # 创建Gradio界面
        # with gr.Blocks() as demo:
        gr.Markdown("# Inference GUI of Northwestern Polytechnical University and MedAI🚀", elem_id="title")
        # 合并后的窗口
        with gr.Column():  # 主列容器
            # 聊天历史区域（保持原有结构）
            with gr.Row():
                gr.Markdown("### Chat History")

            chatbot = gr.Chatbot(
                label="对话历史",
                elem_id="chatbot",
                show_copy_button=True,
                type="messages",
                avatar_images=("/data/scy/SCY/my_vlm/user.png", "/data/scy/SCY/my_vlm/assistant.png"),
                render_markdown=True,  # 启用Markdown渲染
                height=800

            )
            with gr.Row():
                image_list = gr.Gallery(
                    label="Upload Multiple Images",
                    elem_id="image_gallery",
                    columns=100,  # 设置为较大的值，允许显示更多图片
                    height=150,
                )
                textbox = gr.Textbox(
                    show_label=False,
                    placeholder="Type a message...",
                    container=False,
                    lines=4
                )


            # 交互按钮区域
            with gr.Row(elem_id="buttons"):
                upvote_btn = gr.Button(value="👍  Upvote", interactive=True)
                downvote_btn = gr.Button(value="👎  Downvote", interactive=True)
                flag_btn = gr.Button(value="⚠️  Flag", interactive=True)
                regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=True)
                clear_btn = gr.Button(value="🗑️  Clear history", interactive=True)
                submit_btn = gr.Button(value="Submit",variant="primary")
        # 设置事件处理
        submit_btn.click(
            inference.process_input,
            inputs=[image_list, textbox, chatbot],
            outputs=[textbox, image_list, chatbot],
            queue=False
        )

        upvote_btn.click(
            inference.upvote,
            inputs=[chatbot],
            outputs=[chatbot]
        )

        downvote_btn.click(
            inference.downvote,
            inputs=[chatbot],
            outputs=[chatbot]
        )

        flag_btn.click(
            inference.flag,
            inputs=[chatbot],
            outputs=[chatbot]
        )

        regenerate_btn.click(
            inference.regenerate,
            inputs=[chatbot],
            outputs=[chatbot]
        )

        clear_btn.click(
            inference.clear_history,
            outputs=[chatbot]
        )
    return ui


# Launch the Gradio interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Evaluation for LLaVA Generated Outputs', add_help=False)
    parser.add_argument("--model_path", type=str, default='/data/scy/SCY/LLava/LLaVA-Onevision-7B_fine_tuning',
                        choices=['/data/scy/SCY/LLava/LLaVA-Onevision-7B','/data/scy/SCY/LLava/Llama-3.2-11B-Vision-Instruct'])
    parser.add_argument("--original_model_path", type=str, default='/data/scy/SCY/LLava/LLaVA-Onevision-7B_fine_tuning',
                        choices=['/data/scy/SCY/LLava/LLaVA-Onevision-7B',
                                 '/data/scy/SCY/LLava/Llama-3.2-11B-Vision-Instruct'])
    parser.add_argument("--json_file_path", type=str, default=[
        '/data/scy/SCY/my_vlm/补llava/qwen-trainED-qa3.json'
    ],choices=['breast', 'gynaecology', 'heart', 'kidney', 'liver', 'neck', 'thyroid'])
    parser.add_argument('--save_conversation_path', type=str,
                        default="save_conversation.xlsx", help='path to prediction file',)
    args, unparsed = parser.parse_known_args()
    ui = create_ui()
    ui.launch(server_name="10.168.191.173", server_port=7800,share=False,debug=True, allowed_paths=['/tmp/gradio'])