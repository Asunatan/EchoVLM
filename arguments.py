from typing import Dict, Optional, List, Union
from dataclasses import dataclass, field

import transformers
from transformers import SchedulerType, IntervalStrategy
from transformers.trainer_utils import SaveStrategy
from transformers.training_args import OptimizerNames


@dataclass
class ModelArguments:
    model_id: str = field(default="Qwen2_5_VL")
    #InternVL3_5_MOE,InternVL3_5_MOE_USFM_512,lingshu_with_usfm,lingshu_with_dinov3
    #["Qwen2-VL-MOE","Qwen2-VL",'LLaVA-OneVision'，Qwen2_5_VL,Lingshu,HuatuoGPT-Vision,multi_vision,multi_vision_qformer
    # 'LLaVA-Med','LLaVA','Ultrasound-MOE','Qwen2_5_VL_MOE','Medgemma','Ovis2_5','InternVL3_5']
    model_local_path: Optional[str] = field(default='/data/scy/SCY/Model_weights/Qwen2.5-VL-3B-Instruct')
    auto_processor_local_path: Optional[str] = field(default='/data/scy/SCY/Model_weights/Qwen2.5-VL-3B-Instruct')

    # dinov3_processor: Optional[str] = field(default='/data/scy/SCY/Model_weights/dinov3-vitl16-pretrain-lvd1689m')#default=None
    # usfm_processor: Optional[str] = field(default='/data/scy/SCY/SonoVLM_V2/models/usfm')

    dinov3_processor: Optional[str] = field(default=None)#default=None
    usfm_processor: Optional[str] = field(default=None)

    dinov3_weight: Optional[str] = field(default=None)#default=None
    usfm_weight: Optional[str] = field(default=None)

    training_stage: Optional[str] = field(default='stage2')
    #choices = ['stage1','stage1_5' 'stage2', 'stage3','Comparative_experiment','ablation']
@dataclass
class DataArguments:
    train_data_path: Optional[List[str]] = field(
        default_factory=lambda: [
            '/data/scy/SCY/my_vlm/dataset/breast_train_clear_2025.json',
            '/data/scy/SCY/my_vlm/dataset/gynaecology_train_clear_2025.json',
            '/data/scy/SCY/my_vlm/dataset/heart_train_clear_2025.json',
            '/data/scy/SCY/my_vlm/dataset/kidney_train_clear_2025.json',
            '/data/scy/SCY/my_vlm/dataset/liver_train_clear_2025.json',
            '/data/scy/SCY/my_vlm/dataset/thyroid_train_clear_2025.json',
            '/data/scy/SCY/my_vlm/dataset/vessel_train_clear_2025.json'
                                 ],
        metadata={"help": "Path to the training json."})
    eval_data_path: Optional[List[str]] = field(
        default=None,
        # default_factory=lambda: ['/data/scy/SCY/SonoVLM_V2/dataset/val/breast_val.json'
        #                          ],
        metadata={"help": "Path to the evaluation data json file."})
    #/XYFS01/HOME/sysu_ldchen/sysu_ldchen_1/VLM/01DATA
    #/data/scy/SCY/my_vlm/dataset
    image_folder: Optional[str] = field(default='/data/scy/SCY/my_vlm/dataset')
    video_folder: Optional[str] = field(default=None)
    num_frames: Optional[int] = field(default=8)
    user_key: Optional[str] = field(default="human")
    assistant_key: Optional[str] = field(default="gpt")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    dataloader_num_workers: int = field(
        default=10,
        metadata={
            "help": (
                "Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded"
                " in the main process."
            )
        },
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
    #/XYFS01/HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/checkpoints
    #/data/scy/SCY/SonoVLM_V2/checkpoints
    output_dir: str = field(default='/data/scy/SCY/SonoVLM_V2/checkpoints/echovlm',
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )

    report_to: Union[None, str, List[str]] = field(
        default='tensorboard', metadata={"help": "The list of integrations to report the results and logs to."}
    )
    dataloader_drop_last: bool = field(
        default=True, metadata={"help": "Drop the last incomplete batch if it is not divisible by the batch size."}
    )
    tf32: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to enable tf32 mode, available in Ampere and newer GPU architectures. This is an experimental"
                " API and it may change."
            )
        },
    )
    logging_steps: float = field(
        default=1,
        metadata={
            "help": (
                "Log every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    lr_scheduler_type: Union[SchedulerType, str] = field(
        default="cosine",
        metadata={"help": "The scheduler type to use."},
    )
    warmup_ratio: float = field(
        default=0.05, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )

    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    learning_rate: float = field(default=1e-4, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    data_seed: Optional[int] = field(default=42, metadata={"help": "Random seed to be used with data samplers."})
    save_total_limit: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in"
                " `output_dir`. When `load_best_model_at_end` is enabled, the 'best' checkpoint according to"
                " `metric_for_best_model` will always be retained in addition to the most recent ones. For example,"
                " for `save_total_limit=5` and `load_best_model_at_end=True`, the four last checkpoints will always be"
                " retained alongside the best model. When `save_total_limit=1` and `load_best_model_at_end=True`,"
                " it is possible that two checkpoints are saved: the last one and the best one (if they are different)."
                " Default is unlimited checkpoints"
            )
        },
    )
    save_only_model: bool = field(
        default=False,
        metadata={
            "help": (
                "When checkpointing, whether to only save the model, or also the optimizer, scheduler & rng state."
                "Note that when this is true, you won't be able to resume training from checkpoint."
                "This enables you to save storage by not storing the optimizer, scheduler & rng state."
                "You can only load the model using from_pretrained with this option set to True."
            )
        },
    )
    save_steps: float = field(
        default=1000,
        metadata={
            "help": (
                "Save checkpoint every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    save_strategy: Union[SaveStrategy, str] = field(
        default="epoch",
        metadata={"help": "The checkpoint save strategy to use."},
    )
    eval_strategy: Union[IntervalStrategy, str] = field(
        default="no",
        metadata={"help": "The evaluation strategy to use."},
    )
    eval_steps: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Run an evaluation every X steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    per_device_train_batch_size: int = field(
        default=1, metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=1, metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for evaluation."}
    )
    num_train_epochs: float = field(default=1, metadata={"help": "Total number of training epochs to perform."})
    #/XYFS01/HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/ds_configs/zero2.json
    #/data/scy/SCY/SonoVLM_V2/ds_configs/zero2.json

    deepspeed: Optional[Union[dict, str]] = field(
        default='/data/scy/SCY/SonoVLM_V2/ds_configs/zero2.json',
        # default=None,
        metadata={
            "help": (
                "Enable deepspeed and pass the path to deepspeed json config file (e.g. `ds_config.json`) or an already"
                " loaded json file as a dict"
            )
        },
    )
    bf16: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
                " architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental API and it may change."
            )
        },
    )
    optim: Union[OptimizerNames, str] = field(
        default='adamw_torch',
        metadata={"help": "The optimizer to use."},
    )
    do_train: bool = field(default=True, metadata={"help": "Whether to run training."})
    dataloader_persistent_workers: bool = field(
        default=True,
        metadata={
            "help": "If True, the data loader will not shut down the worker processes after a dataset has been consumed once. This allows to maintain the workers Dataset instances alive. Can potentially speed up training, but will increase RAM usage."
        },
    )
    ddp_timeout: Optional[int] = field(
        default=500,
        metadata={
            "help": "Overrides the default timeout for distributed training (value should be given in seconds)."
        },
    )

    remove_unused_columns: Optional[bool] = field(
        default=False, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )


@dataclass
class LoraArguments:
    use_lora: bool = field(default=True)
    train_vision_encoder_lora: bool = field(default=True)
    train_vision_projector_lora: bool = field(default=True)
    train_llm_lora: bool = field(default=True)
    q_lora: bool = field(default=False)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_weight_path: str = ""
    lora_bias: str = "none"
    lora_all_keys: Optional[List[str]] = field(default_factory=
    lambda:  ["visual", "model"])
    lora_vision_encoder_keys: Optional[List[str]] = field(default_factory=
    lambda:  ["visual.patch_embed", "visual.rotary_pos_emb", "visual.blocks","vision_tower"])
    lora_vision_projector_keys: Optional[List[str]] = field(default_factory=
    lambda: ["visual.merger",'multi_modal_projector'])
    lora_llm_keys: Optional[List[str]] = field(default_factory=
    lambda:   ["language_model",'multi_modal_projector'])