import os.path

import math
from typing import List, Dict, Optional


import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import Sampler
import transformers
from transformers import Trainer
from transformers.trainer import has_length, logger
from transformers.utils import is_sagemaker_mp_enabled


class NoTextOnlyBatchSampler(Sampler):
    r"""
    Sampler that tries its best to sample batches such that no batch has only 
    text (unimodal) data. This is necessary for training with deepspeed. 
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        is_text_only: Optional[List[bool]] = None,
        generator=None,
    ):
        if is_text_only is None:
            raise ValueError("`is_text_only` must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.is_text_only = is_text_only
        self.generator = generator
        self.mega_batch_size = batch_size * world_size

    def __len__(self):
        return len(self.is_text_only)

    def __iter__(self):
        # mm: multimodal, entry that has both text and image/video
        # uni: unimodal, entry that has only text
        mm_indices = [i for i, is_text_only in enumerate(self.is_text_only) if not is_text_only]
        uni_indices = [i for i, is_text_only in enumerate(self.is_text_only) if is_text_only]

        num_batches = math.ceil((len(mm_indices) + len(uni_indices)) / self.mega_batch_size)
        if len(mm_indices) < num_batches:
            raise ValueError(
                f"{len(mm_indices)} multimodal entries, {len(num_batches)} batches. "
                "Not enough multimodal data in the dataset, or the batch size is too small. " 
                "There will be at least one batch that is text-only, which doesn't work with deepspeed. "
                "Try increasing the batch size first."
            )

        # shuffle indices
        mm_indices = [mm_indices[i] for i in torch.randperm(len(mm_indices), generator=None).tolist()]
        uni_indices = [uni_indices[i] for i in torch.randperm(len(uni_indices), generator=None).tolist()]

        # distribute indices into batches
        num_uni_indices_in_mega_batch = [len(uni_indices) // num_batches] * num_batches
        for i in range(len(uni_indices) % num_batches):
            num_uni_indices_in_mega_batch[i] += 1
        
        mega_batches = []
        cur_uni_index = 0
        cur_mm_index = 0
        for i, num_uni_indices in enumerate(num_uni_indices_in_mega_batch):
            mega_batch = []
            mega_batch.extend(uni_indices[cur_uni_index:cur_uni_index + num_uni_indices])
            cur_uni_index += num_uni_indices
            assert len(mega_batch) < self.mega_batch_size

            if i < num_batches - 1:
                increment = self.mega_batch_size - len(mega_batch)
                mega_batch.extend(
                    mm_indices[cur_mm_index:cur_mm_index + increment]
                )
                cur_mm_index += increment
            else: # last batch
                mega_batch.extend(mm_indices[cur_mm_index:])
                assert len(mega_batch) <= self.mega_batch_size, "Last batch is too big."
            
            mega_batches.append(mega_batch)
        
        mega_batch_indices = torch.randperm(len(mega_batches), generator=self.generator)
        mega_batches = [mega_batches[i] for i in mega_batch_indices]
        indices = [i for mega_batch in mega_batches for i in mega_batch]
        return iter(indices)


class TrainerWithCustomSampler(Trainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        is_text_only = self.train_dataset.is_text_only
        return NoTextOnlyBatchSampler(
            self.args.train_batch_size,
            world_size=self.args.world_size * self.args.gradient_accumulation_steps,
            is_text_only=is_text_only,
        )
    
    def _get_eval_sampler(self, eval_dataset: torch.utils.data.Dataset) -> Optional[torch.utils.data.Sampler]:
        is_text_only = eval_dataset.is_text_only
        return NoTextOnlyBatchSampler(
            self.args.eval_batch_size,
            world_size=self.args.world_size,
            is_text_only=is_text_only,
        )
    # def create_optimizer(self):
    #     """
    #     Setup the optimizer.
    #
    #     We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    #     Trainer's init through `optimizers`, or subclass and override this method in a subclass.
    #     """
    #     opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
    #     # from deepspeed.runtime.zero.stage_1_and_2 import
    #     if self.optimizer is None:
    #         decay_parameters = self.get_decay_parameter_names(opt_model)
    #         # params_name_list=[]
    #         # params_list=[]
    #         # for n, p in opt_model.named_parameters():
    #         #     if (n in decay_parameters and p.requires_grad and 'experts' in n):
    #         #         params_list.append(p)
    #         #         params_name_list.append(n)
    #         optimizer_grouped_parameters = [
    #             {
    #                 "params": [
    #                     p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
    #                     # weight组
    #                 ],
    #                 "weight_decay": self.args.weight_decay,
    #                 "name": "deca_weight",
    #                 # "moe": True
    #             },
    #             {
    #                 "params": [
    #                     p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad )
    #                     # bias组
    #                 ],
    #                 "weight_decay": 0.0,
    #                 "name": "no_decay_bias",
    #                 # "moe":True
    #             },
    #         #     {
    #         #         "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and 'experts' in n)#weight组
    #         #         ],
    #         #         "weight_decay": self.args.weight_decay,
    #         #         "name": "deca_weight",
    #         #         "moe":True
    #         #     },
    #         #     {
    #         #         "params": [
    #         #             p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and 'experts' not in n)
    #         #             # weight组
    #         #         ],
    #         #         "weight_decay": self.args.weight_decay,
    #         #         "name": "deca_weight",
    #         #     },
    #         #     # {
    #         #     #     "params": [
    #         #     #         p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and 'experts' in n)#bias组
    #         #     #     ],
    #         #     #     "weight_decay": 0.0,
    #         #     #     "name": "no_decay_bias",
    #         #     #     "moe":True
    #         #     # },
    #         #     {
    #         #         "params": [
    #         #             p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and 'experts' not in n)
    #         #             # bias组
    #         #         ],
    #         #         "weight_decay": 0.0,
    #         #         "name": "no_decay_bias"
    #         # },
    #         ]
    #         if self.args.train_moe:#跑非moe模型要关闭
    #             from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
    #             optimizer_grouped_parameters = split_params_into_different_moe_groups_for_optimizer(optimizer_grouped_parameters)
    #         # optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
    #
    #         optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)
    #
    #         # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
    #         # e.g. for GaLore optimizer.
    #         if "params" in optimizer_kwargs:
    #             optimizer_grouped_parameters = optimizer_kwargs.pop("params")
    #
    #         # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
    #         # e.g. for LOMO optimizer.
    #         if "model" in optimizer_kwargs:
    #             optimizer_grouped_parameters = optimizer_kwargs.pop("model")
    #
    #         # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
    #         # to avoid arguments conflicts.
    #         if "optimizer_dict" in optimizer_kwargs:
    #             optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")
    #
    #         self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    #
    #         if optimizer_cls.__name__ == "Adam8bit":
    #             import bitsandbytes
    #
    #             manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
    #
    #             skipped = 0
    #             for module in opt_model.modules():
    #                 if isinstance(module, nn.Embedding):
    #                     skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
    #                     logger.info(f"skipped {module}: {skipped/2**20}M params")
    #                     manager.register_module_override(module, "weight", {"optim_bits": 32})
    #                     logger.debug(f"bitsandbytes: will optimize {module} in fp32")
    #             logger.info(f"skipped: {skipped/2**20}M params")
    #
    #     if is_sagemaker_mp_enabled():
    #         self.optimizer = smp.DistributedOptimizer(self.optimizer)
    #
    #     return self.optimizer


def find_all_linear_names(named_modules: Dict, target_modules: List[str]):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in named_modules.items():
        if not any([module_name in name for module_name in target_modules]):
            continue

        if isinstance(module, cls) and ('moe' not in name) and ('coefficient' not in name):
            lora_module_names.add(name)

    for name in list(lora_module_names):
        if 'lm_head' in name: # needed for 16-bit
            lora_module_names.remove(name)

    return sorted(list(lora_module_names))


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args)


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,save_stage: str, use_lora:str,output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        if save_stage=='stage2' or use_lora:
            base_model = trainer.model.merge_and_unload()
            base_model.save_pretrained(os.path.join(output_dir,'merged_full_parameters'))
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)