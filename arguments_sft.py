from typing import Dict, Optional, List, Union, Literal
from dataclasses import dataclass, field

import transformers
from transformers import SchedulerType, IntervalStrategy
from transformers.trainer_utils import SaveStrategy
from transformers.training_args import OptimizerNames
from trl import ModelConfig,ScriptArguments,SFTConfig

@dataclass
class ModelArguments(ModelConfig):
    model_id: Optional[str] = field(default="Medgemma")
    #InternVL3_5_MOE,InternVL3_5_MOE_USFM_512,lingshu_with_usfm,lingshu_with_dinov3
    #["Qwen2-VL-MOE","Qwen2-VL",'LLaVA-OneVision'，Qwen2_5_VL,Lingshu,HuatuoGPT-Vision,multi_vision,multi_vision_qformer
    # 'LLaVA-Med','LLaVA','Ultrasound-MOE','Qwen2_5_VL_MOE','Medgemma','Ovis2_5','InternVL3_5']
    auto_processor_local_path: Optional[str] = field(default='/data/scy/SCY/Model_weights/medgemma-4b-it')
    model_local_path: Optional[str] = field(
        default='/data/scy/SCY/Model_weights/medgemma-4b-it',
        metadata={"help": "Model checkpoint for weights initialization."},
    )
    dinov3_processor: Optional[str] = field(default=None)#default=None
    usfm_processor: Optional[str] = field(default=None)
    dinov3_weight: Optional[str] = field(default=None)#default=None
    usfm_weight: Optional[str] = field(default=None)

    training_stage: Optional[str] = field(default='stage2')
    #choices = ['stage1','stage1_5' 'stage2', 'stage3','Comparative_experiment','ablation']

    dtype: Optional[str] = field(
        default="bfloat16",
        metadata={
            "help": "Override the default `torch.dtype` and load the model under this dtype.",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={
            "help": "Whether to allow for custom models defined on the Hub in their own modeling files. This option "
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        },
    )
    attn_implementation: Optional[str] = field(
        default='flash_attention_2',
        metadata={
            "help": "Which attention implementation to use. You can run `--attn_implementation=flash_attention_2`, in "
            "which case you must install this manually by running `pip install flash-attn --no-build-isolation`."
        },
    )
    use_peft: bool = field(
        default=True,
        metadata={"help": "Whether to use PEFT for training."},
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA R value."},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha."},
    )
    lora_bias: Literal["none", "all", "lora_only"] = field(
        default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout."},
    )
    lora_target_modules: Optional[list[str]] = field(
        default=None,
        metadata={"help": "LoRA target modules."},
    )
    train_vision_encoder_lora: bool = field(default=True)
    train_vision_projector_lora: bool = field(default=True)
    train_llm_lora: bool = field(default=True)
    lora_all_keys: Optional[List[str]] = field(default_factory=
    lambda:  ["visual", "model"])
    lora_vision_encoder_keys: Optional[List[str]] = field(default_factory=
    lambda:  ["visual.patch_embed", "visual.rotary_pos_emb", "visual.blocks","vision_tower"])
    lora_vision_projector_keys: Optional[List[str]] = field(default_factory=
    lambda: ["visual.merger",'multi_modal_projector'])
    lora_llm_keys: Optional[List[str]] = field(default_factory=
    lambda:   ["language_model"])

    lora_target_parameters: Optional[list[str]] = field(
        default=None,
        metadata={"help": "List of target parameters for LoRA."},
    )
    lora_modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={"help": "Model layers to unfreeze & train."},
    )
    lora_task_type: str = field(
        default="CAUSAL_LM",
        metadata={"help": "Task type to pass for LoRA (use 'SEQ_CLS' for reward modeling)."},
    )
    use_rslora: bool = field(
        default=False,
        metadata={
            "help": "Whether to use Rank-Stabilized LoRA, which sets the adapter scaling factor to `lora_alpha/√r`, "
            "instead of the original default value of `lora_alpha/r`."
        },
    )
    use_dora: bool = field(
        default=False,
        metadata={
            "help": "Enable Weight-Decomposed Low-Rank Adaptation (DoRA). This technique decomposes the updates of "
            "the weights into two parts, magnitude and direction. Direction is handled by normal LoRA, whereas the "
            "magnitude is handled by a separate learnable parameter. This can improve the performance of LoRA, "
            "especially at low ranks. Right now, DoRA only supports linear and Conv2D layers. DoRA introduces a "
            "bigger overhead than pure LoRA, so it is recommended to merge weights for inference."
        },
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Whether to use 8 bit precision for the base model. Works only with LoRA."},
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Whether to use 4 bit precision for the base model. Works only with LoRA."},
    )
    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization type.", "choices": ["fp4", "nf4"]},
    )
    use_bnb_nested_quant: bool = field(
        default=False,
        metadata={"help": "Whether to use nested quantization."},
    )
    # Deprecated params
    torch_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={
            "help": "Override the default `torch.dtype` and load the model under this dtype.",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )





@dataclass
class DataArguments(ScriptArguments):
    train_data_path: Optional[List[str]] = field(
        default_factory=lambda: [
            '/XYFS01/HOME/sysu_ldchen/sysu_ldchen_1/VLM/my_vlm/datasets/echovlm'
                                 ],
        metadata={"help": "Path to the training json."})
    eval_data_path: Optional[List[str]] = field(
        default=None,
        # default_factory=lambda: ['/data/scy/SCY/SonoVLM_V2/dataset/val/breast_val.json'
        #                          ],
        metadata={"help": "Path to the evaluation data json file."})
    #/XYFS01/HOME/sysu_ldchen/sysu_ldchen_1/VLM/01DATA
    #/data/scy/SCY/my_vlm/dataset
    image_folder: Optional[str] = field(default='/XYFS01/HOME/sysu_ldchen/sysu_ldchen_1/VLM/01DATA')
    video_folder: Optional[str] = field(default=None)
    num_frames: Optional[int] = field(default=8)
    user_key: Optional[str] = field(default="human")
    assistant_key: Optional[str] = field(default="gpt")


@dataclass
class TrainingArguments(SFTConfig):
    output_dir: Optional[str] = field(
        default='/data/scy/SCY/SonoVLM_V2/dataset/echovlm',
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written. Defaults to 'trainer_output' if not provided."
        },
    )
    do_train: bool = field(default=True, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=True, metadata={"help": "Whether to run eval on the dev set."})

    max_length: Optional[int] = field(
        default=10240,
        metadata={
            "help": "Maximum length of the tokenized sequence. Sequences longer than `max_length` are truncated from"
            "the right. If `None`, no truncation is applied. When packing is enabled, this value sets the "
            "sequence length."
        },
    )
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
    weight_decay: float = field(default=0.1, metadata={"help": "Weight decay for AdamW if we apply some."})

    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
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
        default=100,
        metadata={
            "help": (
                "Save checkpoint every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    save_strategy: Union[SaveStrategy, str] = field(
        default="steps",
        metadata={"help": "The checkpoint save strategy to use."},
    )
    eval_strategy: Union[IntervalStrategy, str] = field(
        default="steps",
        metadata={"help": "The evaluation strategy to use."},
    )
    eval_steps: Optional[float] = field(
        default=1000,
        metadata={
            "help": (
                "Run an evaluation every X steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    torch_empty_cache_steps: Optional[int] = field(
        default=100,
        metadata={
            "help": "Number of steps to wait before calling `torch.<device>.empty_cache()`."
            "This can help avoid CUDA out-of-memory errors by lowering peak VRAM usage at a cost of about [10% slower performance](https://github.com/huggingface/transformers/issues/31372)."
            "If left unset or set to None, cache will not be emptied."
        },
    )
    gradient_accumulation_steps: int = field(
        default=6,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    per_device_train_batch_size: int = field(
        default=2, metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=2, metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for evaluation."}
    )
    num_train_epochs: float = field(default=1, metadata={"help": "Total number of training epochs to perform."})

    #/XYFS01/HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/ds_configs/zero2.json
    #/data/scy/SCY/SonoVLM_V2/ds_configs/zero2.json

    deepspeed: Optional[Union[dict, str]] = field(
        default='/XYFS01/HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/ds_configs/zero3.json',
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
    label_smoothing_factor: float = field(
        default=0.0, metadata={"help": "The label smoothing epsilon to apply (zero means no label smoothing)."}
    )
    use_liger_kernel: bool = field(
        default=False,
        metadata={"help": "Whether or not to enable the Liger Kernel for model training."},
    )
    # Parameters that control the training
    completion_only_loss: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to compute loss only on the completion part of the sequence. If set to `True`, loss is "
                "computed only on the completion, which is supported only for prompt-completion datasets. If `False`, "
                "loss is computed on the entire sequence. If `None` (default), the behavior depends on the dataset: "
                "loss is computed on the completion for prompt-completion datasets, and on the full sequence for "
                "language modeling datasets."
            )
        },
    )
    assistant_only_loss: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to compute loss only on the assistant part of the sequence. If set to `True`, loss is "
                "computed only on the assistant responses, which is supported only for conversational datasets. If `False`, "
                "loss is computed on the entire sequence."
            )
        },
    )
    loss_type: str = field(
        default="nll",
        metadata={
            "help": (
                'Type of loss to use. Possible values are `"nll"` (negative log-likelihood, default) and `"dft"` '
                "(Dynamic Fine-Tuning, as described in https://huggingface.co/papers/2508.05629)."
            )
        },
    )
    neftune_noise_alpha: Optional[float] = field(
        default=5.0,
        metadata={
            "help": "Activates neftune noise embeddings into the model. NEFTune has been proven to drastically improve model performances for instruction fine-tuning. Check out the original paper here: https://huggingface.co/papers/2310.05914 and the original code here: https://github.com/neelsjain/NEFTune. Only supported for `PreTrainedModel` and `PeftModel` classes."
        },
    )
    packing: bool = field(
        default=False,
        metadata={
            "help": "Whether to group multiple sequences into fixed-length blocks to improve computational efficiency "
            "and reduce padding. Uses `max_length` to define sequence length."
        },
    )
    packing_strategy: str = field(
        default="bfd",
        metadata={
            "help": "Strategy for packing sequences. Can be either `'bfd'` (best-fit decreasing, default), or "
            "`'wrapped'`."
        },
    )
    padding_free: bool = field(
        default=False,
        metadata={
            "help": "Whether to perform forward passes without padding by flattening all sequences in the batch into "
            "a single continuous sequence. This reduces memory usage by eliminating padding overhead. Currently, this "
            "is only supported with the FlashAttention 2 or 3, which can efficiently handle the flattened batch "
            "structure. When packing is enabled with strategy `'bfd'`, padding-free is enabled, regardless of the "
            "value of this parameter."
        },
    )
    pad_to_multiple_of: Optional[int] = field(
        default=None,
        metadata={"help": "If set, the sequences will be padded to a multiple of this value."},
    )
    eval_packing: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to pack the eval dataset. If `None`, uses the same value as `packing`."},
    )
    auto_find_batch_size: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to automatically decrease the batch size in half and rerun the training loop again each time"
                " a CUDA Out-of-Memory was reached"
            )
        },
    )

# class LoraArguments:
#     use_lora: bool = field(default=False)
#     train_vision_encoder_lora: bool = field(default=True)
#     train_vision_projector_lora: bool = field(default=True)
#     train_llm_lora: bool = field(default=True)
#     q_lora: bool = field(default=False)
#     lora_r: int = field(default=8)
#     lora_alpha: int = field(default=16)
#     lora_dropout: float = field(default=0.05)
#     lora_weight_path: str = ""
#     lora_bias: str = "none"
#     lora_all_keys: Optional[List[str]] = field(default_factory=
#     lambda:  ["visual", "model"])
#     lora_vision_encoder_keys: Optional[List[str]] = field(default_factory=
#     lambda:  ["visual.patch_embed", "visual.rotary_pos_emb", "visual.blocks","vision_tower"])
#     lora_vision_projector_keys: Optional[List[str]] = field(default_factory=
#     lambda: ["visual.merger",'multi_modal_projector'])
#     lora_llm_keys: Optional[List[str]] = field(default_factory=
#     lambda:   ["language_model",'multi_modal_projector'])