import json
import os
import re
from typing import Dict, List, Sequence, Union, Optional, Callable, Tuple, Any
from PIL.Image import Image
from abc import ABC, abstractmethod
import numpy as np
import torch
from transformers import PreTrainedTokenizer, AutoProcessor, AutoConfig, ViTImageProcessorFast
import torch.distributed as dist
from transformers.image_utils import make_flat_list_of_images, ImageInput
from transformers.models.internvl.processing_internvl import InternVLProcessorKwargs
from transformers.tokenization_utils_base import TextInput, PreTokenizedInput
from transformers.video_utils import make_batched_videos, VideoInput
from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import ImageInput, concatenate_list, make_flat_list_of_images
from transformers.processing_utils import ImagesKwargs, MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.video_utils import VideoInput, make_batched_videos
# from models.dinov3.image_processing_dinov3_vit_fast import DINOv3ViTImageProcessorFast
from transformers import DINOv3ViTImageProcessorFast
from transformers import InternVLProcessor
import time
from concurrent.futures import ThreadPoolExecutor, as_completed  # 替换 ProcessPoolExecutor
from dataclasses import dataclass, field
def rank0_print(*args):
    if dist.is_initialized() and dist.get_rank() == 0:
        print(*args)


SYSTEM_MESSAGE = "You are a helpful assistant."
IGNORE_INDEX = -100


class InternVLProcessor_Custom(InternVLProcessor):

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Optional[Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]] = None,
        audio=None,
        videos: Optional[VideoInput] = None,
        **kwargs: Unpack[InternVLProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to PreTrainedTokenizerFast's [`~PreTrainedTokenizerFast.__call__`] to encode the text if `text`
        is not `None`, otherwise encode default OCR queries which depends on the `format`, `box`, `color`, `multi_page` and
        `crop_to_patches` arguments. To prepare the vision inputs, this method forwards the `images` and `kwrags` arguments to
        GotOcr2ImageProcessor's [`~GotOcr2ImageProcessor.__call__`] if `images` is not `None`.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `list[str]`, `list[list[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            videos (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The image or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        if text is None:
            raise ValueError("You have to specify text.")

        output_kwargs = self._merge_kwargs(
            InternVLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if not isinstance(text, (list, tuple)):
            text = [text]

        # Process images and videos separately, as videos don't support crop_to_patches
        image_num_patches = []
        video_num_patches = []
        image_videos_inputs = {}
        image_pixel_values = None
        video_pixel_values = None
        image_num_patches_indices = np.array([0])
        video_patch_indices = np.array([0])
        video_num_patches_indices = np.array([0])
        if images is not None:
            images = make_flat_list_of_images(images)
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_num_patches = image_inputs.pop("num_patches")
            image_pixel_values = image_inputs.pop("pixel_values")
            image_num_patches_indices = np.cumsum(image_num_patches)
        if videos is not None:
            videos = make_batched_videos(videos)
            video_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            video_pixel_values = video_inputs.pop("pixel_values_videos")

            # Obtain per frame information first and then flatten to (BS * T, ...)
            num_frames_per_video = [len(video) for video in video_pixel_values]
            video_num_patches = [1 for frames in num_frames_per_video for _ in range(frames)]
            video_patch_indices = np.cumsum(num_frames_per_video)
            video_num_patches_indices = np.cumsum(video_num_patches)
            video_pixel_values = video_pixel_values.flatten(0, 1)

        if images is not None or videos is not None:
            text, image_video_patches, image_index, video_index = self._insert_media_placeholders(
                text,
                image_pixel_values,
                video_pixel_values,
                image_num_patches,
                video_num_patches,
                image_num_patches_indices,
                video_num_patches_indices,
                video_patch_indices,
            )
            if images is not None and image_index != len(images):
                raise ValueError("Number of image placeholders in the prompt does not match the number of images.")
            if videos is not None and video_index != len(videos):
                raise ValueError("Number of video placeholders in the prompt does not match the number of videos.")

            # Concatenate the interleaved image and video patches (function agnostic to the patches type (list, numpy array, torch tensor))
            image_videos_inputs = {"pixel_values": concatenate_list(image_video_patches)}

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[np.isin(array_ids, self.image_ids)] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data={**text_inputs, **image_videos_inputs,**{'image_num_patches_indices':image_num_patches_indices}}, tensor_type=return_tensors)


def process_single_sample(
    input_ids: List[int],  # 单个样本的 input_ids（list 格式，方便进程传递）
    sample_text: str,
    start_tokens: List[int],      # 辅助参数：start token list
    end_tokens: List[int],        # 辅助参数：end token list
) -> List[int]:
    """
    单个样本的 labels 计算（独立函数，可被多进程调用）
    返回：(labels_list, is_all_ignore)
    """

    sequence_length = len(input_ids)
    labels = [IGNORE_INDEX] * sequence_length
    start_len = len(start_tokens)
    end_len = len(end_tokens)
    start_index_list, end_index_list = [], []
    current_position = 0
    while current_position < sequence_length:
        # 找 start token
        start_index = None
        for i in range(current_position, sequence_length - start_len + 1):
            if input_ids[i:i + start_len] == start_tokens:
                start_index = i + start_len
                break
        if start_index is None:
            break

        # 找 end token
        end_index = None
        for j in range(start_index, sequence_length - end_len + 1):
            if input_ids[j:j + end_len] == end_tokens:
                end_index = j + end_len
                break
        if end_index is None:
            break

        # 填充 labels
        labels[start_index:end_index] = input_ids[start_index:end_index]
        start_index_list.append(start_index)
        end_index_list.append(end_index)
        # 4. 下一轮从 end token 之后继续
        current_position = end_index
    if len(start_index_list) != len(end_index_list):
        rank0_print("Number of start_indexes does not match the number of end_indexes")
        labels = [IGNORE_INDEX] * sequence_length
    # 检查是否全为 IGNORE_INDEX
    elif all(label == IGNORE_INDEX for label in labels):
        rank0_print(
            f"[ERROR] All labels are IGNORE_INDEX (-100)!\n"
            f"input_text={sample_text}\n"
            f"input_ids ={input_ids}\n"
            f"labels    ={labels}"
        )
    return labels


@dataclass
class CustomDataCollatorApplyChatTemplateV2():
    def __init__(
        self,
        model_args = None,
        training_args=None,
        default_assistant_start_tokens=None,
        default_assistant_end_tokens=None,
        checkpoints_path: str = None,
        auto_processor_local_path: str = None,
    ) -> None:
        self.max_length = training_args.max_length
        self.dinov3_processor=None
        self.usfm_processor=None
        if model_args.model_id=='lingshu_with_usfm' or model_args.model_id=='lingshu_with_dinov3' or model_args.model_id=='multi_vision':
            if model_args.dinov3_processor is not None:
                self.dinov3_processor = DINOv3ViTImageProcessorFast.from_pretrained(model_args.dinov3_processor)
            if model_args.usfm_processor is not None:
                self.usfm_processor = ViTImageProcessorFast.from_pretrained(model_args.usfm_processor)

        if isinstance(auto_processor_local_path, str):
            if 'InternVL3_5' in auto_processor_local_path:
                self.processor = InternVLProcessor_Custom.from_pretrained(auto_processor_local_path)
                self.dinov3_processor = DINOv3ViTImageProcessorFast.from_pretrained(model_args.dinov3_processor)
            else:
                self.processor = AutoProcessor.from_pretrained(auto_processor_local_path)
        else:
            raise 'Please provide the original preprocess path.'
        #151644,77091,198
        # 动态获取开始和结束 token
        rank0_print(
            f'default_assistant_start_tokens：{default_assistant_start_tokens}------->default_assistant_end_tokens:{default_assistant_end_tokens}')
        self.assistant_start_tokens = torch.tensor(self.processor.tokenizer.encode(default_assistant_start_tokens, add_special_tokens=False),
                                                   dtype=torch.long)
        self.assistant_end_tokens = torch.tensor(self.processor.tokenizer.encode(default_assistant_end_tokens, add_special_tokens=False),
                                                 dtype=torch.long)
        rank0_print(
            f'default_assistant_start_tokens：{self.assistant_start_tokens}------->default_assistant_end_tokens:{self.assistant_end_tokens}')
        #151645,198
        self.assistant_start_tokens_len = len(self.assistant_start_tokens)
        self.assistant_end_tokens_len = len(self.assistant_end_tokens)
        self.checkpoints_path = checkpoints_path
        if self.checkpoints_path:
            os.makedirs(self.checkpoints_path, exist_ok=True)
            self.checkpoints_path = os.path.join(self.checkpoints_path, 'training_dataset_checkpoints.json')


    def save_trained_data(self, id):
        if self.checkpoints_path:
            with open(self.checkpoints_path, 'a') as f:
                json.dump(id, f)
                f.write('\n')

    @torch.no_grad()
    def get_labels(self, input_ids: torch.Tensor,input_text) -> torch.Tensor:
        sequence_length = input_ids.size(0)
        device = input_ids.device
        # 先把结果全部设为 IGNORE_INDEX
        labels = torch.full((sequence_length,), IGNORE_INDEX,
                            dtype=torch.long, device=device)
        start_index_list,end_index_list =[],[]
        current_position = 0
        while current_position < sequence_length:
            # 1. 从 current_position 开始找下一个 start token 的起点
            start_index = None
            for i in range(current_position,sequence_length - self.assistant_start_tokens_len + 1):
                if torch.equal(
                        input_ids[i: i + self.assistant_start_tokens_len],
                        self.assistant_start_tokens
                ):
                    start_index = i + self.assistant_start_tokens_len
                    break

            if start_index is None:
                break
            # 2. 从 start_index 之后找第一个 end token
            end_index = None
            for j in range(start_index,
                           sequence_length - self.assistant_end_tokens_len + 1):
                if torch.equal(
                        input_ids[j: j + self.assistant_end_tokens_len],
                        self.assistant_end_tokens
                ):
                    end_index = j + self.assistant_end_tokens_len
                    break
            if end_index is None:
                break  # 有 start 没 end，直接结束,被truncation不参与计算

            # 3. 把区间内的内容写回 labels
            labels[start_index: end_index] = input_ids[start_index: end_index]
            start_index_list.append(start_index)
            end_index_list.append(end_index)
            # 4. 下一轮从 end token 之后继续
            current_position = end_index
        if len(start_index_list) != len(end_index_list):
            rank0_print("Number of start_indexes does not match the number of end_indexes")
            labels = torch.full((sequence_length,), IGNORE_INDEX,dtype=torch.long, device=device)
        elif (labels == IGNORE_INDEX).all():
            rank0_print(
                f"[ERROR] All labels are IGNORE_INDEX (-100)!\n"
                f"input_text={input_text}\n"
                f"input_ids ={input_ids.tolist()}\n"
                f"labels    ={labels.tolist()}"
            )
        return labels

    @torch.no_grad()
    def get_labels_parallel(self, input_ids: torch.Tensor, batch_text: List[str]) -> torch.Tensor:
        """
        多进程并行计算 labels：为每个样本启动一个进程
        input_ids: [B, L]
        batch_text: [B]
        return: [B, L]
        """
        B, L = input_ids.shape
        input_ids_list = input_ids.cpu().tolist()  # 转为 list，方便进程传递

        # 构建任务列表：每个任务包含单个样本的参数
        tasks = []
        for i in range(B):
            tasks.append((
                input_ids_list[i],  # 单个样本的 input_ids（list）
                batch_text[i],      # 单个样本的 text
                self.assistant_start_tokens.tolist(),  # start tokens（list）
                self.assistant_end_tokens.tolist(),    # end tokens（list）
            ))
        # 多进程执行
        batch_labels_list = [None] * B
        if  B > 1:
            # 用 ProcessPoolExecutor 管理进程池
            with ThreadPoolExecutor(max_workers=min(B,8)) as executor:
                # 提交所有任务，返回 Future 对象
                future_to_idx = {}
                for original_idx, task in enumerate(tasks):
                    # 提交任务时，记录“任务结果 → 原始输入位置”的映射
                    future = executor.submit(process_single_sample, *task)
                    future_to_idx[future] = original_idx
                # 收集结果：不管哪个进程先完成，都放回原始位置
                for future in as_completed(future_to_idx):
                    original_idx = future_to_idx[future]  # 拿到该结果对应的输入位置
                    try:
                        labels_list = future.result()
                        # 核心：将结果放入与输入对应的索引位置（确保 inputs[original_idx] → labels[original_idx]）
                        batch_labels_list[original_idx] = labels_list
                    except Exception as e:
                        rank0_print(f"样本 {original_idx}（原始输入位置）计算 labels 失败：{e}")
                        # 失败时，该位置填充全 IGNORE_INDEX，不影响其他位置顺序
                        batch_labels_list[original_idx] = [IGNORE_INDEX] * L
        else:
            # 单进程模式（fallback，用于 batchsize=1 或禁用多进程）
            for i, task in enumerate(tasks):
                labels_list = process_single_sample(*task)
                batch_labels_list.append(labels_list)
        # 转为 tensor 返回
        return torch.tensor(batch_labels_list, dtype=torch.long, device=input_ids.device)
    def __call__(self, batch_instances: list[dict[str, Any]]) -> dict[str, Any]:
        batch_images=[]
        batch_text=[]
        for instance in batch_instances:
            if instance.get('images'):
                batch_images.append(instance['images'])
            template_text = instance.get('messages')
            apply_chat_template_text = self.processor.apply_chat_template(template_text, tokenize=False,
                                                                          add_generation_prompt=False,
                                                                          add_vision_id=False)
            batch_text.append(apply_chat_template_text)
            #废案
            # chat_template_text_list = apply_chat_template_text.split('<|im_start|>assistant\n')
            # system_user_prompt = self.processor(text=chat_template_text_list[0],images=instance['images'],return_tensors="pt")
            # assistant_response = self.processor(text=chat_template_text_list[1],return_tensors="pt")

        # Tokenize and process in batch
        batch_encoding = self.processor(
            images=batch_images,
            text=batch_text,
            padding=True,
            padding_side="right",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=False,  # to avoid adding the BOS, twice see https://huggingface.co/blog/qgallouedec/gotchas-in-tokenizer-behavior#7-chat-template-and-tokenization-dont-compose-due-to-special-tokens
        )

        input_ids = batch_encoding["input_ids"]  # (B, L)

        batch_labels = self.get_labels_parallel(input_ids, batch_text)
        batch_encoding["labels"] = batch_labels
        assert input_ids.shape == batch_labels.shape, f"input_ids shape {input_ids.shape} != labels shape {batch_labels.shape}"
        if self.dinov3_processor is not None :
            dinov3_pixel_values = self.dinov3_processor(images=batch_images,return_tensors="pt",size={"height": 448, "width": 448})
            batch_encoding["dinov3_pixel_values"] = dinov3_pixel_values['pixel_values']
        if self.usfm_processor is not None :
            usfm_pixel_values = self.usfm_processor(images=batch_images,return_tensors="pt",size={"height": 224, "width": 224})
            batch_encoding["usfm_pixel_values"] = usfm_pixel_values['pixel_values']
        return batch_encoding

@dataclass
class CustomDataCollatorApplyChatTemplate():
    def __init__(
        self,
        model_args = None,
        default_assistant_start_tokens=None,
        default_assistant_end_tokens=None,
        checkpoints_path: str = None,
        auto_processor_local_path: str = None,
    ) -> None:
        self.dinov3_processor=None
        self.usfm_processor=None
        if model_args.dinov3_processor is not None:
            self.dinov3_processor = DINOv3ViTImageProcessorFast.from_pretrained(model_args.dinov3_processor)
        if model_args.usfm_processor is not None:
            self.usfm_processor = ViTImageProcessorFast.from_pretrained(model_args.usfm_processor)

        if isinstance(auto_processor_local_path, str):
            if 'InternVL3_5' in auto_processor_local_path:
                self.processor = InternVLProcessor_Custom.from_pretrained(auto_processor_local_path)
                self.dinov3_processor = DINOv3ViTImageProcessorFast.from_pretrained(model_args.dinov3_processor)
            else:
                self.processor = AutoProcessor.from_pretrained(auto_processor_local_path)
                from transformers import Qwen2_5_VLProcessor,Qwen2TokenizerFast
        #151644,77091,198
        # 动态获取开始和结束 token
        rank0_print(
            f'default_assistant_start_tokens：{default_assistant_start_tokens}------->default_assistant_end_tokens:{default_assistant_end_tokens}')
        self.assistant_start_tokens = torch.tensor(self.processor.tokenizer.encode(default_assistant_start_tokens, add_special_tokens=False),
                                                   dtype=torch.long)
        self.assistant_end_tokens = torch.tensor(self.processor.tokenizer.encode(default_assistant_end_tokens, add_special_tokens=False),
                                                 dtype=torch.long)
        rank0_print(
            f'default_assistant_start_tokens：{self.assistant_start_tokens}------->default_assistant_end_tokens:{self.assistant_end_tokens}')
        #151645,198
        self.assistant_start_tokens_len = len(self.assistant_start_tokens)
        self.assistant_end_tokens_len = len(self.assistant_end_tokens)
        self.checkpoints_path = checkpoints_path
        if self.checkpoints_path:
            os.makedirs(self.checkpoints_path, exist_ok=True)
            self.checkpoints_path = os.path.join(self.checkpoints_path, 'training_dataset_checkpoints.json')


    def save_trained_data(self, id):
        if self.checkpoints_path:
            with open(self.checkpoints_path, 'a') as f:
                json.dump(id, f)
                f.write('\n')

    @torch.no_grad()
    def get_labels(self, input_ids: torch.Tensor,input_text) -> torch.Tensor:
        # rank0_print(f'起始标记:{self.assistant_start_tokens}')

        sequence_length = input_ids.size(0)
        device = input_ids.device
        # 先把结果全部设为 IGNORE_INDEX
        labels = torch.full((sequence_length,), IGNORE_INDEX,
                            dtype=torch.long, device=device)
        start_index_list,end_index_list =[],[]
        current_position = 0
        while current_position < sequence_length:
            # 1. 从 current_position 开始找下一个 start token 的起点
            start_index = None
            for i in range(current_position,sequence_length - self.assistant_start_tokens_len + 1):
                if torch.equal(
                        input_ids[i: i + self.assistant_start_tokens_len],
                        self.assistant_start_tokens
                ):
                    start_index = i + self.assistant_start_tokens_len
                    break

            if start_index is None:
                break
            # 2. 从 start_index 之后找第一个 end token
            end_index = None
            for j in range(start_index,
                           sequence_length - self.assistant_end_tokens_len + 1):
                if torch.equal(
                        input_ids[j: j + self.assistant_end_tokens_len],
                        self.assistant_end_tokens
                ):
                    end_index = j + self.assistant_end_tokens_len
                    break
            if end_index is None:
                break  # 有 start 没 end，直接结束

            # 3. 把区间内的内容写回 labels
            labels[start_index: end_index] = input_ids[start_index: end_index]
            start_index_list.append(start_index)
            end_index_list.append(end_index)
            # 4. 下一轮从 end token 之后继续
            current_position = end_index
        assert len(start_index_list) == len(end_index_list), "Number of start_indexes does not match the number of end_indexes"
        if (labels == IGNORE_INDEX).all():
            rank0_print(
                f"[ERROR] All labels are IGNORE_INDEX (-100)!\n"
                f"input_text={input_text}\n"
                f"input_ids ={input_ids.tolist()}\n"
                f"labels    ={labels.tolist()}"
            )
        return labels

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch_ids = [instance['id'] for instance in instances]
        batch_images = [instance['images'] for instance in instances]
        batch_videos = [instance["videos"] for instance in instances]
        batch_system_prompts: List[Union[str, None]] = [instance["system_prompt"] for instance in instances]
        batch_conversations: List[List] = [instance["conversations"] for instance in instances]
        assert len(batch_images) == len(batch_videos) == len(batch_system_prompts) == len(
            batch_conversations), "Batch lists length mismatch"
        has_images = any(len(imgs) > 0 for imgs in batch_images)  # 有任一图像则为True
        has_videos = any(len(vids) > 0 for vids in batch_videos)  # 纯文本时为False
        batch_apply_chat_template_text = []
        # for system_prompt, cur_images, cur_videos, cur_convs in zip(batch_system_prompts, batch_images, batch_videos,
        #                                                                 batch_conversations):
        for batch_idex, (system_prompt, cur_images, cur_videos, cur_convs) in enumerate(zip(
                batch_system_prompts, batch_images, batch_videos, batch_conversations)):
            # self.save_trained_data(id)
            # print(cur_images)
            # cur_images = [img for img in cur_images]
            cur_text = []
            cur_num_images = cur_num_videos = 0
            # if system_prompt is not None:
            #     cur_text.append({
            #         "role": "system",
            #         "content": [{"type": "text", "text": system_prompt}]
            #     })
            # 处理btch中的每一条
            for i, text in enumerate(cur_convs):
                text_key = text["from"]
                text_value = str(text['value'])
                if i % 2 == 0:
                    assert text_key == "human", f"Invalid conversation user_key:{text_key}"
                    num_images = text_value.count("<image>")
                    cur_num_images += num_images
                    num_videos = text_value.count("<video>")
                    cur_num_videos += num_videos
                    text_value = text_value.replace("<image>\n", "").replace("<video>\n", "").strip()
                    text_value = text_value.replace("<image> ", "").replace("<video>", "").strip()
                    text_value = text_value.replace("<image>", "").replace("<video>", "").strip()
                    # if num_images != len(cur_images) and num_images!=0:
                    #     num_images = len(cur_images)
                    cur_text.append({
                        "role": "user",
                        "content": [{"type": "image"}] * num_images + \
                                   [{"type": "video"}] * num_videos + \
                                   [{"type": "text", "text": text_value}]

                    })
                else:
                    assert text_key == "gpt", f"Invalid conversation assistant_key:{text_key}"
                    cur_text.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": text_value}]
                    })
            assert len(cur_images) == cur_num_images, \
                f"Image count mismatch: len(cur_images)={len(cur_images)}, cur_num_images={cur_num_images}, " \
                f"sample id={batch_ids[batch_idex]}, " \
                f"cur_images content: {cur_images!r}"
            assert len(cur_videos) == cur_num_videos, f"Video count mismatch"
            # heavily borrowed from https://github.com/2U1/Qwen2-VL-Finetune
            # 关于添加图像的ID号：https://moon-ci-docs.huggingface.co/docs/transformers/pr_35264/en/model_doc/qwen2_vl
            apply_chat_template_text = self.processor.apply_chat_template(cur_text, tokenize=False,
                                                                          add_generation_prompt=False,
                                                                          add_vision_id=False)
            batch_apply_chat_template_text.append(apply_chat_template_text)
        # Tokenize and process in batch
        processor_kwargs = {
            "text": batch_apply_chat_template_text,
            "padding": True,
            "return_tensors": "pt"
        }
        batch_images = [imgs for imgs in batch_images if imgs]   # [[img1], [], [img2,img3]] -> [img1, img2, img3]
        if has_images:  # 只要有一张图就传
            processor_kwargs["images"] = batch_images
        batch_encoding = self.processor(**processor_kwargs)

        input_ids = batch_encoding["input_ids"]  # (B, L)

        batch_labels = torch.stack(
            [self.get_labels(ids, txt) for ids, txt in zip(input_ids, batch_apply_chat_template_text)]
        )
        batch_encoding["labels"] = batch_labels
        assert input_ids.shape == batch_labels.shape, f"input_ids shape {input_ids.shape} != labels shape {batch_labels.shape}"
        if self.dinov3_processor is not None and has_images:
            dinov3_pixel_values = self.dinov3_processor(images=batch_images,return_tensors="pt",size={"height": 448, "width": 448})
            batch_encoding["dinov3_pixel_values"] = dinov3_pixel_values['pixel_values']
        if self.usfm_processor is not None and has_images:
            usfm_pixel_values = self.usfm_processor(images=batch_images,return_tensors="pt",size={"height": 224, "width": 224})
            batch_encoding["usfm_pixel_values"] = usfm_pixel_values['pixel_values']
        return batch_encoding
