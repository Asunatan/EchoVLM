import os.path
import random
import re
from functools import partial
from typing import Any, Dict, List, Literal, Optional
# from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, Qwen2_5_VLConfig, Qwen2_5_VLModel, \
#     DINOv3ViTImageProcessorFast, ViTImageProcessorFast
import torch
import sys
from swift.llm.model.register import get_model_tokenizer_from_local
from swift.llm.template.constant import MLLMTemplateType
from transformers.integrations import is_deepspeed_zero3_enabled
from torchvision.transforms import v2 as V2, InterpolationMode
from swift.llm import (Model, ModelGroup, ModelMeta, MultiModelKeys, Template, TemplateMeta, get_model_tokenizer,
                       get_model_tokenizer_with_flash_attn, get_packed_seq_params, get_template, register_model,
                       register_model_arch, register_template, to_float_dtype, get_model_tokenizer_multimodal,
                       )
from swift.llm.model.model.qwen import patch_qwen_vl_utils, get_model_tokenizer_qwen2_vl
from swift.llm.model.patcher import patch_get_input_embeddings
from swift.llm.model.utils import use_submodel_func, AttnImpl
from swift.llm.template.template_inputs import StdTemplateInputs
from swift.llm.template.utils import Context, findall
from swift.llm.template.vision_utils import load_audio
from swift.utils import get_env_args, get_logger, is_deepspeed_enabled

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, Qwen2_5OmniConfig, AutoConfig
from swift.llm import MODEL_MAPPING, MODEL_ARCH_MAPPING
from transformers.utils.versions import require_version


from typing import Any, Dict, Optional

from swift.llm import DatasetMeta, ResponsePreprocessor, load_dataset, register_dataset,MessagesPreprocessor,RowPreprocessor,AutoPreprocessor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from SonoVLM_V2.models.lingshu_multi_vision import Qwen2_5_VL_MOE_Config
class CustomPreprocessor(MessagesPreprocessor):


    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:

        return super().preprocess(row)


register_dataset(
    DatasetMeta(
        ms_dataset_id='/XYFS01/HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/dataset/stage2_data/train',
        dataset_name='Ultrasound_train',
        # dataset_path='/data/scy/SCY/SonoVLM_V2/dataset/train/stage1',
        preprocess_func=CustomPreprocessor(),
    ))
register_dataset(
    DatasetMeta(
        ms_dataset_id='/XYFS01/HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/dataset/stage2_data/val',
        dataset_name='Ultrasound_val',
        split=['validation'],
        # dataset_path='/data/scy/SCY/SonoVLM_V2/dataset/train/stage1',
        preprocess_func=CustomPreprocessor(),
    ))
register_dataset(
    DatasetMeta(
        ms_dataset_id='/data/scy/SCY/SonoVLM_V2/data_generation/Miscellaneous_data/output_json/dpo',
        dataset_name='Ultrasound_dpo',
        # split=['validation'],
        # dataset_path='/data/scy/SCY/SonoVLM_V2/dataset/train/stage1',
        preprocess_func=CustomPreprocessor(),
    ))
register_dataset(
    DatasetMeta(
        ms_dataset_id='/XYFS01/HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/dataset/compare_dataset',
        dataset_name='compare',
        # split=['validation'],
        # dataset_path='/data/scy/SCY/SonoVLM_V2/dataset/train/stage1',
        preprocess_func=CustomPreprocessor(),
    ))
register_dataset(
    DatasetMeta(
        ms_dataset_id='/XYFS01/HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/dataset/stage_data_new',
        dataset_name='stage_data_new',
        # split=['validation'],
        # dataset_path='/data/scy/SCY/SonoVLM_V2/dataset/train/stage1',
        preprocess_func=CustomPreprocessor(),
    ))
register_dataset(
    DatasetMeta(
        ms_dataset_id='/XYFS01/HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/dataset/stage2_data_repeat_system',
        dataset_name='stage2_data_repeat_system',
        # split=['validation'],
        # dataset_path='/data/scy/SCY/SonoVLM_V2/dataset/train/stage1',
        preprocess_func=CustomPreprocessor(),
    ))
register_dataset(
    DatasetMeta(
        ms_dataset_id='/HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/dataset/stage_data_new_fix',
        dataset_name='stage_data_new_fix',
        # split=['validation'],
        # dataset_path='/data/scy/SCY/SonoVLM_V2/dataset/train/stage1',
        preprocess_func=CustomPreprocessor(),
    ))
register_dataset(
    DatasetMeta(
        ms_dataset_id='/HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/dataset/stage_data_new_no_system_caption',
        dataset_name='stage_data_new_no_system_caption',
        # split=['validation'],
        # dataset_path='/data/scy/SCY/SonoVLM_V2/dataset/train/stage1',
        preprocess_func=CustomPreprocessor(),
    ))
register_dataset(
    DatasetMeta(
        ms_dataset_id='/HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/dataset/stage_data_new_no_system_caption_2_1',
        dataset_name='stage_data_new_no_system_caption_2_1',
        # split=['validation'],
        # dataset_path='/data/scy/SCY/SonoVLM_V2/dataset/train/stage1',
        preprocess_func=CustomPreprocessor(),
    ))
register_dataset(
    DatasetMeta(
        ms_dataset_id='/HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/dataset/stage_data_new_no_system_caption_2_2',
        dataset_name='stage_data_new_no_system_caption_2_2',
        # split=['validation'],
        # dataset_path='/data/scy/SCY/SonoVLM_V2/dataset/train/stage1',
        preprocess_func=CustomPreprocessor(),
    ))
register_dataset(
    DatasetMeta(
        ms_dataset_id='/HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/dataset/stage2_data_final_no_fix_no_sys',
        dataset_name='stage2_data_final_no_fix_no_sys',
        # split=['validation'],
        # dataset_path='/data/scy/SCY/SonoVLM_V2/dataset/train/stage1',
        preprocess_func=CustomPreprocessor(),
    ))
register_dataset(
    DatasetMeta(
        ms_dataset_id='/HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/dataset/stage2_data_final_no_fix_no_sys_finding',
        dataset_name='stage2_data_final_no_fix_no_sys_finding',
        # split=['validation'],
        # dataset_path='/data/scy/SCY/SonoVLM_V2/dataset/train/stage1',
        preprocess_func=CustomPreprocessor(),
    ))
register_dataset(
    DatasetMeta(
        ms_dataset_id='/HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/dataset/7000_report_cot_json',
        dataset_name='cot_data_simple',
        # split=['validation'],
        # dataset_path='/data/scy/SCY/SonoVLM_V2/dataset/train/stage1',
        preprocess_func=CustomPreprocessor(),
    ))
register_dataset(
    DatasetMeta(
        ms_dataset_id='/HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/dataset/cot_complex_all',
        dataset_name='cot_data_complex',
        # split=['validation'],
        # dataset_path='/data/scy/SCY/SonoVLM_V2/dataset/train/stage1',
        preprocess_func=CustomPreprocessor(),
    ))
register_dataset(
    DatasetMeta(
        ms_dataset_id='/HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/dataset/cot_data_caption_report',
        dataset_name='cot_data_caption_report',
        # split=['validation'],
        # dataset_path='/data/scy/SCY/SonoVLM_V2/dataset/train/stage1',
        preprocess_func=CustomPreprocessor(),
    ))
register_dataset(
    DatasetMeta(
        ms_dataset_id='/HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/dataset/cot_data_caption_report1',
        dataset_name='cot_data_caption_report1',
        # split=['validation'],
        # dataset_path='/data/scy/SCY/SonoVLM_V2/dataset/train/stage1',
        preprocess_func=CustomPreprocessor(),
    ))
register_dataset(
    DatasetMeta(
        ms_dataset_id='/HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/dataset/cot_data_caption_report_sample',
        dataset_name='cot_data_caption_report_sample',
        # split=['validation'],
        # dataset_path='/data/scy/SCY/SonoVLM_V2/dataset/train/stage1',
        preprocess_func=CustomPreprocessor(),
    ))
register_dataset(
    DatasetMeta(
        ms_dataset_id='/HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/dataset/cot_data_caption_report_i2f',
        dataset_name='cot_data_caption_report_i2f',
        # split=['validation'],
        # dataset_path='/data/scy/SCY/SonoVLM_V2/dataset/train/stage1',
        preprocess_func=CustomPreprocessor(),
    ))

register_dataset(
    DatasetMeta(
        ms_dataset_id='/HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/dataset/finding2impression_onlytxt',
        dataset_name='finding2impression_onlytxt',
        # split=['validation'],
        # dataset_path='/data/scy/SCY/SonoVLM_V2/dataset/train/stage1',
        preprocess_func=CustomPreprocessor(),
    ))
register_dataset(
    DatasetMeta(
        ms_dataset_id='/HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/dataset/cot_data_caption_report_i2f/finding2impression',
        dataset_name='finding2impression',
        # split=['validation'],
        # dataset_path='/data/scy/SCY/SonoVLM_V2/dataset/train/stage1',
        preprocess_func=CustomPreprocessor(),
    ))
register_dataset(
    DatasetMeta(
        ms_dataset_id='/HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/dataset/finding2impression_onlytxt_finding_img',
        dataset_name='finding2impression_onlytxt_finding_img',
        # split=['validation'],
        # dataset_path='/data/scy/SCY/SonoVLM_V2/dataset/train/stage1',
        preprocess_func=CustomPreprocessor(),
    ))
register_dataset(
    DatasetMeta(
        ms_dataset_id='/HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/dataset/finding2impression_onlytxt_new',
        dataset_name='finding2impression_onlytxt_new',
        # split=['validation'],
        # dataset_path='/data/scy/SCY/SonoVLM_V2/dataset/train/stage1',
        preprocess_func=CustomPreprocessor(),
    ))
if __name__ == '__main__':
    os.environ["ROOT_IMAGE_DIR"] = "/data/scy/SCY/my_vlm/dataset"
    dataset = load_dataset('Ultrasound_val')[0]
    print(f'dataset: {dataset}')
    for data in dataset:
        print(f'dataset[0]: {dataset[0]}')


def get_model_tokenizer_processor(model_dir, *args, **kwargs):
    from SonoVLM_V2.models.lingshu_multi_vision import Multi_Vision_ForConditionalGeneration
    from transformers import Qwen2_5_VLProcessor,Qwen2_5_VLConfig,DINOv3ViTImageProcessorFast,ViTImageProcessorFast
    kwargs['automodel_class'] = kwargs['automodel_class'] or Multi_Vision_ForConditionalGeneration
    processor = Qwen2_5_VLProcessor.from_pretrained('/data/scy/SCY/Model_weights/Qwen2.5-VL-7B-Instruct',trust_remote_code=True)
    processor.dinov3_processor= DINOv3ViTImageProcessorFast.from_pretrained('/data/scy/SCY/Model_weights/dinov3-vitl16-pretrain-lvd1689m')
    processor.usfm_processor = ViTImageProcessorFast.from_pretrained('/data/scy/SCY/SonoVLM_V2/models/usfm')
    kwargs['tokenizer'] = processor.tokenizer

    kwargs['model_config'] = Qwen2_5_VL_MOE_Config.from_pretrained(model_dir, trust_remote_code=True)
    model, _ = get_model_tokenizer_with_flash_attn(model_dir, *args, **kwargs)
    model.init_checkpoint_model(dinov3_weights_path='/data/scy/SCY/Model_weights/dinov3-vitl16-pretrain-lvd1689m',
                                usfm_weights_path='/data/scy/SCY/SonoVLM_V2/models/usfm/USFM_latest.pth',
                                training_stage='stage1')
    # model.usfm.to(model.device)
    # model.dinov3.to(model.device)
    if model is not None:
        base_model = model.model if 'AWQ' in model.__class__.__name__ else model
        patch_get_input_embeddings(base_model.visual, 'patch_embed')

    from qwen_vl_utils import vision_process
    check_qwen_vl_utils = kwargs.get('_check_qwen_vl_utils', True)
    if check_qwen_vl_utils:
        require_version('qwen_vl_utils<0.0.12')
    global_vars = patch_qwen_vl_utils(vision_process)
    processor.global_vars = global_vars  # In order to have different hashes for the template.
    return model, processor



register_model_arch(
    MultiModelKeys(
        'lingshu_with_donov3_and_usfm',
        # `freeze_llm`, `freeze_vit`, `freeze_aligner` behavior is determined by the values below.
        # For example: full parameter training, if `freeze_vit=True`, it will freeze parameters of
        # model layers prefixed with `thinker.audio_tower` and `thinker.visual`.
        # LoRA training, if `freeze_vit=False`, it will additionally add LoRA to Linear layers
        # prefixed with `thinker.audio_tower` and `thinker.visual`.
        language_model='model',
        vision_tower=['visual', 'usfm','dinov3'],
        aligner=['dinov3_projector','usfm_projector','fusion_projector','fusion_weights'],#'visual.merger'
        # Generator parts will never be trained or remain frozen.
        generator=['lm_head'],
    ))

register_model(
    ModelMeta(
        model_type='lingshu_with_donov3_and_usfm',
        model_groups=
            [# 注册你的模型（name 为模型名，path 为权重路径）
            ModelGroup([
                Model('Qwen/lingshu_with_donov3_and_usfm-7B', model_path='/data/scy/SCY/Model_weights/Qwen2.5-VL-7B-Instruct'),
                # Model('Qwen/Qwen2.5-Omni-7B', 'Qwen/Qwen2.5-Omni-7B'),
            ]),
        ],
        template='lingshu_with_donov3_and_usfm_template',
        # Function to get model and processor.
        get_function=get_model_tokenizer_processor,
        model_arch='lingshu_with_donov3_and_usfm',  # Usually set only for multimodal models
        # Used for automatic model_type matching
        architectures=['Multi_Vision_VLModel', 'Multi_Vision_ForConditionalGeneration'],
        # Used to prompt users about dependency versions (can be removed)
        # requires=['transformers>=4.50', 'soundfile', 'qwen_omni_utils', 'decord'],
        is_multimodal=True,  # Whether it's a multimodal model
        # Used to prompt users (can be removed)
        tags=['vision', 'text'],
        # Additional files to save during full parameter training/merge-lora
        # additional_saved_files=['spk_dict.pt'],
    ))

if __name__ == '__main__':
    # Test and debug
    model, processor = get_model_tokenizer('/data/scy/SCY/Model_weights/Qwen2.5-VL-3B-Instruct', model_type='lingshu_with_donov3_and_usfm')
    # model, processor = get_model_tokenizer('/data/scy/SCY/Model_weights/Qwen2.5-VL-7B-Instruct',
    #                                        model_type='qwen2_5_vl')
    a=0
    # 查看所有已注册的模型类型（model_type）
    # print("Registered model types:")
    # for model_type in MODEL_MAPPING.keys():
    #     print(f" - {model_type}")
    # # 查看你注册的模型结构（model_arch）
    # print("\nRegistered model architectures:")
    # for arch_name in MODEL_ARCH_MAPPING.keys():
    #     print(f" - {arch_name}")
    # a=0

logger = get_logger()

from swift.llm.template.template.qwen import Qwen2_5VLTemplate, QwenTemplateMeta

image_transformer = V2.Compose([
            V2.ToPILImage(),
            # V2.RandomCrop(384,)
            V2.Resize((392, 392)),
            # V2.AutoAugment(interpolation=InterpolationMode.BICUBIC),
            # V2.RandomHorizontalFlip(p=0.5)
        ])
class LingShuMultiVisionTemplate_V2(Qwen2_5VLTemplate):
    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        inputs.images = [image_transformer(im) for im in inputs.images]
        encoded = super()._encode(inputs)
        dinov3_pixel_values = self.processor.dinov3_processor(images=inputs.images, return_tensors="pt",
                                                    size={"height": 448, "width": 448})
        encoded["dinov3_pixel_values"] = dinov3_pixel_values['pixel_values']
        usfm_pixel_values = self.processor.usfm_processor(images=inputs.images, return_tensors="pt",
                                                size={"height": 224, "width": 224})
        encoded["usfm_pixel_values"] = usfm_pixel_values['pixel_values']
        return encoded
    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        dinov3_pixel_values = [b['dinov3_pixel_values'] for b in batch if b.get('dinov3_pixel_values') is not None]
        if len(dinov3_pixel_values) > 0:
            res['dinov3_pixel_values'] = torch.concat(dinov3_pixel_values)
        usfm_pixel_values = [b['usfm_pixel_values'] for b in batch if b.get('usfm_pixel_values') is not None]
        if len(usfm_pixel_values) > 0:
            res['usfm_pixel_values'] = torch.concat(usfm_pixel_values)
        return res
    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs




register_template(
    QwenTemplateMeta(
        'lingshu_with_donov3_and_usfm_template',
        template_cls=LingShuMultiVisionTemplate_V2,
        # default_system='You are a helpful ultrasound assistant.'
        )
)


if __name__ == '__main__':
    # Test and debug
    model, processor = get_model_tokenizer('/data/scy/SCY/SonoVLM_V2/checkpoints/multi_vision/checkpoint-4000', model_type='lingshu_with_donov3_and_usfm')
    template = get_template('lingshu_with_donov3_and_usfm_template', processor)
    # model, processor = get_model_tokenizer('/data/scy/SCY/Model_weights/Qwen2.5-VL-7B-Instruct', model_type='qwen2_5_vl')
    # template = get_template('qwen2_5_vl', processor)
    data = {
        'messages': [
            {
                'role': 'user',
                'content': '<image>Describe the image content.'
            },
            {
                'role': 'assistant',
                'content': 'A child and a cat.'
            },
        ],
        # 'videos': ['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4'],
        'images': ['/data/scy/SCY/2024testnew/2022010713270739/3218093300062.jpg'],
    }
    # data = {
    #     'messages': [
    #         {
    #             'role': 'user',
    #             'content': [{"type": "image"},
    #                         {"type": "text", "text": 'Describe the image <image> content.'}
    #                         ]
    #         },
    #         {
    #             'role': 'assistant',
    #             'content': [{"type": "text", "text": 'A child and a cat.'}]
    #         },
    #     ],
    #     # 'videos': ['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4'],
    #     'images': ['/data/scy/SCY/2024testnew/2022010713270739/3218093300062.jpg'],
    # }
    template.set_mode('train')
    encoded = template.encode(data)
    print('input_ids: ' + template.safe_decode(encoded['input_ids']))
    print('labels: ' + template.safe_decode(encoded['labels']))
    print('keys: ' + str(encoded.keys()))
