import random
import re

import PIL.PngImagePlugin

import av
import os
import json
import copy
import pandas as pd
from PIL import Image
from typing import Dict, List, Optional
import math
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import v2 as V2, InterpolationMode
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
# 将限制提高到 1000MB（或按需调整）
# PIL.PngImagePlugin.MAX_TEXT_CHUNK = 1000 * 1024 * 1024  # 10MB
import torch.distributed as dist
def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args)

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
# MAX_PIXELS = 16384 * 28 * 28
MAX_PIXELS = 196 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning
    which is generalized enough to handle both images and videos.
    """

    def __init__(
            self,
            data_path: str,
            image_folder: Optional[str] = None,
            video_folder: Optional[str] = None,
            num_frames: int = 8,
            user_key: str = "human",
            assistant_key: str = "gpt",
            training_stage=None,
            checkpoint=None,
            test_excel_file=None,
            processor=None,
    ) -> None:
        super(LazySupervisedDataset, self).__init__()
        self.list_data_dict = []
        # breast:13000,gynaecology:66492,heart:47061,kidney:16388,liver:18043,thyriod:21969,vessel:4320
        # random.seed(42)
        self.training_stage = training_stage
        def get_json_files(paths):
            json_files = []
            for path in paths:
                if os.path.isdir(path):
                    # 如果是文件夹，遍历所有文件
                    for root, _, files in os.walk(path):
                        for file in files:
                            if file.endswith('.json'):
                                json_files.append(os.path.join(root, file))
                elif os.path.isfile(path) and path.endswith('.json'):
                    # 如果是JSON文件，直接添加
                    json_files.append(path)
                else:
                    print(f"警告: 跳过非JSON文件或不存在的路径: {path}")
            return json_files

        json_files = get_json_files(data_path)
        with ProcessPoolExecutor(max_workers=7) as exe:
            future_map = {exe.submit(_process_one_json, f, training_stage): f for f in json_files}
            for fut in as_completed(future_map):
                file = future_map[fut]
                try:
                    samples = fut.result()
                    self.list_data_dict.extend(samples)
                except Exception as e:
                    print(f"[WARN] 处理 {file} 异常: {e}", file=sys.stderr)


        if checkpoint and test_excel_file is not None:
            test_df = pd.read_excel(test_excel_file, dtype=str)
            excel_ids = set(test_df['id'])
            self.list_data_dict = [item for item in self.list_data_dict if item['id'] not in excel_ids]
        self.image_folder = image_folder
        self.video_folder = video_folder
        self.num_frames = num_frames
        self.user_key = user_key
        self.assistant_key = assistant_key


        self.image_transformer = V2.Compose([
            V2.ToPILImage(),
            # V2.RandomCrop(384,)
            V2.Resize((392,392)),
            # V2.AutoAugment(interpolation=InterpolationMode.BICUBIC),
            # V2.RandomHorizontalFlip(p=0.5)
        ])

    def __len__(self) -> int:
        return len(self.list_data_dict)

    def round_by_factor(self, number: int, factor: int) -> int:
        """Returns the closest integer to 'number' that is divisible by 'factor'."""
        return round(number / factor) * factor

    def ceil_by_factor(self, number: int, factor: int) -> int:
        """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
        return math.ceil(number / factor) * factor

    def floor_by_factor(self, number: int, factor: int) -> int:
        """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
        return math.floor(number / factor) * factor

    def smart_resize(self,
                     height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS,
                     max_pixels: int = MAX_PIXELS
                     ) -> tuple[int, int]:
        """
        Rescales the image so that the following conditions are met:

        1. Both dimensions (height and width) are divisible by 'factor'.

        2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

        3. The aspect ratio of the image is maintained as closely as possible.
        """
        if max(height, width) / min(height, width) > MAX_RATIO:
            raise ValueError(
                f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
            )
        h_bar = max(factor, self.round_by_factor(height, factor))
        w_bar = max(factor, self.round_by_factor(width, factor))
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = self.floor_by_factor(height / beta, factor)
            w_bar = self.floor_by_factor(width / beta, factor)
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = self.ceil_by_factor(height * beta, factor)
            w_bar = self.ceil_by_factor(width * beta, factor)
        return h_bar, w_bar

    # 同步裁剪 conversations 中的 <image> 标记
    def filter_images_in_conversations(self,conversations, keep_indices):
        new_conversations = []
        image_counter = 0
        for turn in conversations:
            text = turn.get("value", "")
            new_text = text
            if turn.get("from") == "human":
                # 替换 <image> 标记
                parts = []
                for seg in text.split("<image>"):
                    if seg:
                        parts.append(seg)
                    if "<image>" in text and image_counter not in keep_indices:
                        # 跳过这个 <image> 标记
                        pass
                    else:
                        parts.append("<image>")
                    image_counter += 1
                # 去掉最后一个多余的 <image>
                if parts and parts[-1] == "<image>" and image_counter - 1 not in keep_indices:
                    parts.pop()
                new_text = "".join(parts)
                print(new_text)
                print(f"这里有{keep_indices}张图")
            new_conversations.append({"from": turn["from"], "value": new_text})

        return new_conversations
    def __getitem__(self, i) -> Dict[str, List]:
        source = self.list_data_dict[i]
        images = []
        image_sources = source.get("image")
        if image_sources is not None:
            if isinstance(image_sources, str):
                image_sources = [image_sources]
            elif not isinstance(image_sources, list):
                raise ValueError(f"Invalid image source type: {type(image_sources)}")

            if source.get("conversations")[0].get("from") == "system":
                source["conversations"][1]["value"] = "<image>" * len(image_sources) + re.sub(r"<image>\s*", "",
                                                                                  source["conversations"][1]["value"]).strip()
            elif source.get("conversations")[0].get("from") == "human":
                source["conversations"][0]["value"] = "<image>" * len(image_sources) + re.sub(r"<image>\s*", "",
                                                                                  source["conversations"][0]["value"]).strip()
            else:
                raise ValueError(f"<image>错误)")
                # source["conversations"]=self.filter_images_in_conversations(source["conversations"], image_sources)
                # conversations_list = source.get("conversations")
            for image_path in image_sources:
                image_path = os.path.join(self.image_folder, image_path)
                try:
                    img = Image.open(image_path).convert("RGB")
                    width, height = img.size
                    resized_height, resized_width = self.smart_resize(
                        height,
                        width,
                        factor=IMAGE_FACTOR,
                        min_pixels=MIN_PIXELS,
                        max_pixels=MAX_PIXELS,
                    )
                    img = img.resize((resized_width, resized_height))
                    # img = self.image_transformer(img)
                except Exception as e:
                    raise ValueError(f"图片加载错误:{image_path}, 错误信息: {e}") from e
                # width, height = img.size
                # resized_height, resized_width = self.smart_resize(
                #     height,
                #     width,
                #     factor=IMAGE_FACTOR,
                #     min_pixels=MIN_PIXELS,
                #     max_pixels=MAX_PIXELS,
                # )
                # img = img.resize((resized_width, resized_height))
                images.append(img)
            assert len(images) == len(image_sources), f"图像加载数量不匹配: 期望 {len(image_sources)}, 实际加载 {len(images)}"
        videos = []
        video_sources = source.get("video")  # 无字段时返回None
        if video_sources is not None:  # 仅当video字段存在且非空时处理
            if isinstance(video_sources, str):
                video_sources = [video_sources]
            elif not isinstance(video_sources, list):
                raise ValueError(f"Invalid video source type: {type(video_sources)} (value: {video_sources})")

            num_frames = [self.num_frames] * len(video_sources)
            for video_path, cur_num_frames in zip(video_sources, num_frames):
                if self.video_folder is not None:
                    video_path = os.path.join(self.video_folder, video_path)
                try:
                    container = av.open(video_path)
                    total_frames = container.streams.video[0].frames
                    indices = np.arange(0, total_frames, total_frames / cur_num_frames).astype(int)
                    clip = read_video_pyav(container, indices)
                    videos.append(clip)
                except Exception as e:
                    raise ValueError(f"视频加载错误:{video_path}, 错误信息: {e}") from e

        conversations_list = source.get("conversations")
        assert conversations_list is not None and len(conversations_list) > 0, \
            f"No conversations found in {source}"

        # 处理系统提示：查找from为"system"的条目
        system_prompt = None
        conversation_start_idx = 0  # 实际对话内容的起始索引

        # 检查第一个元素是否为系统提示
        first_turn = conversations_list[0]
        if isinstance(first_turn, dict) and first_turn.get("from") == "system":
            system_prompt = first_turn.get("value")
            conversation_start_idx = 1  # 跳过系统提示

        # 计算有效对话轮次（需为偶数）
        effective_turns = len(conversations_list) - conversation_start_idx
        assert effective_turns % 2 == 0, \
            f"Effective conversation turns must be even, got {effective_turns} " \
            f"(total turns={len(conversations_list)}, has_system={bool(system_prompt)})"

        return dict(
            id = source.get('id', None),
            image_id=source.get("image", None),
            images=images,
            videos=videos,
            conversations=conversations_list[conversation_start_idx:],  # 从实际对话开始处截取
            system_prompt=system_prompt
        )





# ---------- 纯函数：单文件处理 ----------
def _process_one_json(
    data_file: str,
    training_stage: Optional[str],
) -> List[Dict]:
    """
    读一个 JSON 文件并按 training_stage 清洗数据。
    返回样本列表；出错时返回空列表并把异常信息打到 stderr。
    """
    rank0_print(f'加载数据集----------------------------->{data_file}')
    try:
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] 跳过损坏文件 {data_file}: {e}", file=sys.stderr)
        return []

    # 下面这段逻辑和你原来的一模一样
    if training_stage == "stage1":
        if "PubMedVision_Chinese" in data_file:
            for item in data:
                item["conversations"] = copy.deepcopy(item["Alignment_VQA_conversations"])
                del item["Instruction-Tuning_VQA_conversations"]
                del item["Alignment_VQA_conversations"]
    elif training_stage == "stage1_5":
        if "PubMedVision_Chinese" in data_file:
            for item in data:
                item["conversations"] = copy.deepcopy(item["Alignment_VQA_conversations"])
                del item["Instruction-Tuning_VQA_conversations"]
                del item["Alignment_VQA_conversations"]
    elif training_stage == "stage2":
        if "PubMedVision_Chinese" in data_file:
            for item in data:
                item["conversations"] = copy.deepcopy(item["Instruction-Tuning_VQA_conversations"])
                del item["Alignment_VQA_conversations"]
                del item["Instruction-Tuning_VQA_conversations"]
    return data


class LazySupervisedDatasetV2(Dataset):
    def __init__(
        self,
        data_path: str,
        image_folder: Optional[str] = None,
        video_folder: Optional[str] = None,
        num_frames: int = 8,
        user_key: str = "user",
        assistant_key: str = "assistant",
        training_stage=None,
        test_file=None,
        adapt_2_trl=False,
        processor=None,
        num_workers: int = min(32, os.cpu_count() or 8),  # 并行度
    ) -> None:
        super().__init__()

        # 1. 收集所有 JSON 文件
        def gather_json_files(paths):
            files = []
            for p in paths:
                if os.path.isdir(p):
                    for root, _, fs in os.walk(p):
                        files += [os.path.join(root, f) for f in fs if f.endswith(".json")]
                elif os.path.isfile(p) and p.endswith(".json"):
                    files.append(p)
                else:
                    print(f"[WARN] 跳过非 JSON 路径: {p}", file=sys.stderr)
            return files

        json_files = gather_json_files([data_path] if isinstance(data_path, str) else data_path)

        # 2. 多进程并行处理
        all_data = []
        with ProcessPoolExecutor(max_workers=num_workers) as exe:
            future_map = {exe.submit(_process_one_json, f, training_stage): f for f in json_files}
            for fut in as_completed(future_map):
                file = future_map[fut]
                try:
                    samples = fut.result()
                    all_data.extend(samples)
                except Exception as e:
                    print(f"[WARN] 处理 {file} 异常: {e}", file=sys.stderr)

        all_exclude_ids = set()
        if test_file is not None:
            file_ext = os.path.splitext(test_file)[1].lower()
            if file_ext == '.json':
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                try:
                    checkpoint_data = json.loads(content)
                    if isinstance(checkpoint_data, list):
                        all_exclude_ids.update(
                            item.get("id") for item in checkpoint_data
                            if item.get("id") is not None
                        )
                    elif isinstance(checkpoint_data, dict):
                        all_exclude_ids.add(checkpoint_data["id"])  # 修复：add 不是 update
                except json.JSONDecodeError:
                    for line in content.splitlines():
                        data = json.loads(line)
                        if isinstance(data, dict):
                            all_exclude_ids.add(data["id"])  # 修复：add 不是 update
            elif file_ext in ['.xlsx', '.xls']:
                checkpoint_data = pd.read_excel(test_file, dtype=str)
                if 'id' in checkpoint_data.columns:
                    all_exclude_ids.update(checkpoint_data['id'].dropna())
            if all_exclude_ids:
                initial_count = len(all_data)
                all_data = [it for it in all_data if it.get("id") not in all_exclude_ids]
                final_count = len(all_data)
                print(f"[INFO] 过滤前数据量: {initial_count}, 过滤后数据量: {final_count}, 共排除 {initial_count - final_count} 条数据。")

        # 4. 保存到成员变量
        self.list_data_dict = all_data
        self.image_folder = image_folder
        self.video_folder = video_folder
        self.num_frames = num_frames
        self.user_key = user_key
        self.assistant_key = assistant_key
        self.training_stage = training_stage
        self.adapt_2_trl=adapt_2_trl #refer https://github.com/huggingface/trl/blob/v0.24.0/trl/trainer/sft_trainer.py#L252
        #todo这里保留，决定是否需要使用数据增强
        # self.image_transformer = V2.Compose([
        #     V2.ToPILImage(),
        #     # V2.RandomCrop(384,)
        #     V2.Resize((392,392)),
        #     # V2.AutoAugment(interpolation=InterpolationMode.BICUBIC),
        #     # V2.RandomHorizontalFlip(p=0.5)
        # ])

    def __len__(self) -> int:
        return len(self.list_data_dict)

    def round_by_factor(self, number: int, factor: int) -> int:
        """Returns the closest integer to 'number' that is divisible by 'factor'."""
        return round(number / factor) * factor

    def ceil_by_factor(self, number: int, factor: int) -> int:
        """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
        return math.ceil(number / factor) * factor

    def floor_by_factor(self, number: int, factor: int) -> int:
        """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
        return math.floor(number / factor) * factor

    def smart_resize(self,
                     height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS,
                     max_pixels: int = MAX_PIXELS
                     ) -> tuple[int, int]:
        """
        Rescales the image so that the following conditions are met:

        1. Both dimensions (height and width) are divisible by 'factor'.

        2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

        3. The aspect ratio of the image is maintained as closely as possible.
        """
        if max(height, width) / min(height, width) > MAX_RATIO:
            raise ValueError(
                f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
            )
        h_bar = max(factor, self.round_by_factor(height, factor))
        w_bar = max(factor, self.round_by_factor(width, factor))
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = self.floor_by_factor(height / beta, factor)
            w_bar = self.floor_by_factor(width / beta, factor)
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = self.ceil_by_factor(height * beta, factor)
            w_bar = self.ceil_by_factor(width * beta, factor)
        return h_bar, w_bar

    def __getitem__(self, i) -> Dict[str, List]:
        # 深拷贝避免修改原始数据
        source = copy.deepcopy(self.list_data_dict[i])

        # 处理图片逻辑
        images_list = []
        image_sources = source.get("images")
        if image_sources is not None:
            if isinstance(image_sources, str):
                image_sources = [image_sources]
            # 加载和处理图片
            for image_path in image_sources:
                image_path = os.path.join(self.image_folder, image_path)
                try:
                    img = Image.open(image_path).convert("RGB")
                    width, height = img.size
                    resized_height, resized_width = self.smart_resize(
                        height, width,
                        factor=IMAGE_FACTOR,
                        min_pixels=MIN_PIXELS,
                        max_pixels=MAX_PIXELS,
                    )
                    img = img.resize((resized_width, resized_height))
                    # img = self.image_transformer(img)
                except Exception as e:
                    raise ValueError(f"图片加载错误:{image_path}, 错误信息: {e}") from e
                images_list.append(img)

        # 处理视频逻辑
        videos_list = []
        video_sources = source.get("video")
        if video_sources is not None:
            if isinstance(video_sources, str):
                video_sources = [video_sources]

            num_frames = [self.num_frames] * len(video_sources)
            for video_path, cur_num_frames in zip(video_sources, num_frames):
                if self.video_folder is not None:
                    video_path = os.path.join(self.video_folder, video_path)
                try:
                    container = av.open(video_path)
                    total_frames = container.streams.video[0].frames
                    indices = np.arange(0, total_frames, total_frames / cur_num_frames).astype(int)
                    clip = read_video_pyav(container, indices)
                    videos_list.append(clip)
                except Exception as e:
                    raise ValueError(f"视频加载错误:{video_path}, 错误信息: {e}") from e

        # 统一处理消息格式
        if self.adapt_2_trl:
            # TRL模式：简单移除<image>标签
            for msg in source.get("messages", []):
                if msg.get("role") == "user":
                    msg["content"] = re.sub(r"<image>\s*", "", msg.get("content", ""))
                    break
        else:
            # 标准模式：结构化内容
            total_image_tags = 0
            messages = source.get("messages")

            for msg in messages:
                role = msg.get("role")
                content = msg.get("content", "")

                if role == "system":
                    msg["content"] = [{"type": "text", "text": content}]
                elif role == "user":
                    # 计算图片数量
                    images_length = content.count("<image>")
                    total_image_tags += images_length

                    # 生成图片条目
                    image_entries = [{"type": "image"} for _ in range(images_length)]

                    # 清理内容并构建新结构
                    cleaned_content = re.sub(r"<image>\s*", "", content).strip()
                    msg["content"] = [*image_entries, {"type": "text", "text": cleaned_content}]
                elif role == "assistant":
                    msg["content"] = [{"type": "text", "text": content}]
                else:
                    raise ValueError(
                        f"Invalid role in message: {role}. Expected 'user', 'assistant', or 'system'.")

            # 验证图片数量一致性
            assert len(images_list) == total_image_tags, \
                f"Instance {source.get('id', '<unknown>')}: " \
                f"images length ({len(images_list)}) != " \
                f"<image> tag count ({total_image_tags})"

        # 返回处理后的数据（不修改原始source）
        return {
            "images": images_list,
            "videos": videos_list,
            "messages": source.get("messages"),
            "id": source.get("id")
        }


if __name__ == "__main__":
    from transformers import AutoProcessor
    from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling
    ds = LazySupervisedDatasetV2(
        data_path="/data/scy/SCY/SonoVLM_V2/data_generation/Miscellaneous_data/output_json/stage1/balance/25000-500/train",
        image_folder="/data/scy/SCY/my_vlm/dataset",
        video_folder="/data/scy/SCY/my_vlm/dataset",
        num_frames=8,
        user_key="user",
        assistant_key="gpt",
        training_stage="stage1",
        num_workers=32
    )
    processor = AutoProcessor.from_pretrained('/data/scy/SCY/Model_weights/Qwen2.5-VL-7B-Instruct')
    collator = DataCollatorForVisionLanguageModeling(
        processor=processor,
        max_length=2048,
        pad_to_multiple_of=1,
        return_tensors="pt"
    )
    examples = [ds[0], ds[1]]   # 若数据集只有 1 条，就改成 [ds[0], ds[0]]
    # 5. 调用 collator
    batch = collator(examples)
    print(batch)


