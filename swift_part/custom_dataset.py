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



# ---------- 纯函数：单文件处理 ----------
def _process_one_json(
    data_file: str,
) -> List[Dict]:
    """
    读一个 JSON 文件并按 training_stage 清洗数据。
    返回样本列表；出错时返回空列表并把异常信息打到 stderr。
    """
    try:
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] 跳过损坏文件 {data_file}: {e}", file=sys.stderr)
        return []
    return data


class EvalDataset(Dataset):
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
            future_map = {exe.submit(_process_one_json, f): f for f in json_files}
            for fut in as_completed(future_map):
                file = future_map[fut]
                try:
                    samples = fut.result()
                    all_data.extend(samples)
                except Exception as e:
                    print(f"[WARN] 处理 {file} 异常: {e}", file=sys.stderr)

        # 3. 过滤测试集 ID（如果有）
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
        source = self.list_data_dict[i]
        return source





