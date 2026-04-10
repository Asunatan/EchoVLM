# --- 来自 [Webpage](https://huggingface.co/papers/2111.09886) 的内容： ---
# SimMIM: A Simple Framework for Masked Image Modeling
# ... 代码 ...
# SimMIM 可以应用于不同的骨干网络，如 ViT 或 Swin。
# ... 代码 ...
from functools import partial
from pathlib import Path

import torch
from torch import nn
from transformers import ViTModel

from modeling_usfm import USFM_VisionTransformer


def map_simmim_to_hf(simmim_state_dict: dict) -> dict:
    """
    将 SimMIM 风格的 state_dict 映射到 Hugging Face ViT 模型的 state_dict。
    假设 SimMIM 权重是基于 Swin Transformer 骨干网络预训练的。
    """
    ViTModel
    hf_state_dict = {}
    # 用于记录 SimMIM 权重中被使用的键
    used_keys = set()

    # --- 嵌入层 (Embeddings) ---
    for key in ['patch_embed.proj.weight', 'patch_embed.proj.bias']:
        if key in simmim_state_dict:
            hf_state_dict[f'vit.embeddings.patch_embeddings.projection.{key.split(".")[-1]}'] = simmim_state_dict[key]
            used_keys.add(key)
    if 'cls_token' in simmim_state_dict:
        hf_state_dict['vit.embeddings.cls_token'] = simmim_state_dict['cls_token']
        used_keys.add('cls_token')

    # --- Transformer Blocks ---
    for i in range(12):
        simmim_block_prefix = f'blocks.{i}.'
        hf_block_prefix = f'vit.encoder.layer.{i}.'

        # ... (映射 norm, attn, mlp 的代码保持不变) ...
        # Norm 1
        for norm_key in [f'{simmim_block_prefix}norm1.weight', f'{simmim_block_prefix}norm1.bias']:
            if norm_key in simmim_state_dict:
                hf_key = norm_key.replace(f'blocks.{i}.norm1', f'vit.encoder.layer.{i}.layernorm_before')
                hf_state_dict[hf_key] = simmim_state_dict[norm_key]
                used_keys.add(norm_key)
        # Norm 2
        for norm_key in [f'{simmim_block_prefix}norm2.weight', f'{simmim_block_prefix}norm2.bias']:
            if norm_key in simmim_state_dict:
                hf_key = norm_key.replace(f'blocks.{i}.norm2', f'vit.encoder.layer.{i}.layernorm_after')
                hf_state_dict[hf_key] = simmim_state_dict[norm_key]
                used_keys.add(norm_key)

        # Attention
        if f'{simmim_block_prefix}attn.qkv.weight' in simmim_state_dict:
            qkv_weight = simmim_state_dict[f'{simmim_block_prefix}attn.qkv.weight']
            q_weight, k_weight, v_weight = qkv_weight.chunk(3, dim=0)
            hf_state_dict[f'{hf_block_prefix}attention.attention.query.weight'] = q_weight
            hf_state_dict[f'{hf_block_prefix}attention.attention.key.weight'] = k_weight
            hf_state_dict[f'{hf_block_prefix}attention.attention.value.weight'] = v_weight
            used_keys.add(f'{simmim_block_prefix}attn.qkv.weight')

        if f'{simmim_block_prefix}attn.q_bias' in simmim_state_dict and f'{simmim_block_prefix}attn.v_bias' in simmim_state_dict:
            q_bias = simmim_state_dict[f'{simmim_block_prefix}attn.q_bias']
            v_bias = simmim_state_dict[f'{simmim_block_prefix}attn.v_bias']
            k_bias = torch.zeros_like(q_bias)
            hf_state_dict[f'{hf_block_prefix}attention.attention.query.bias'] = q_bias
            hf_state_dict[f'{hf_block_prefix}attention.attention.key.bias'] = k_bias
            hf_state_dict[f'{hf_block_prefix}attention.attention.value.bias'] = v_bias
            used_keys.add(f'{simmim_block_prefix}attn.q_bias')
            used_keys.add(f'{simmim_block_prefix}attn.v_bias')

        for attn_key in [f'{simmim_block_prefix}attn.proj.weight', f'{simmim_block_prefix}attn.proj.bias']:
            if attn_key in simmim_state_dict:
                hf_key = attn_key.replace(f'blocks.{i}.attn.proj', f'vit.encoder.layer.{i}.attention.output.dense')
                hf_state_dict[hf_key] = simmim_state_dict[attn_key]
                used_keys.add(attn_key)

        # MLP
        for mlp_key in [f'{simmim_block_prefix}mlp.fc1.weight', f'{simmim_block_prefix}mlp.fc1.bias',
                        f'{simmim_block_prefix}mlp.fc2.weight', f'{simmim_block_prefix}mlp.fc2.bias']:
            if mlp_key in simmim_state_dict:
                if 'fc1' in mlp_key:
                    hf_key = mlp_key.replace(f'blocks.{i}.mlp.fc1', f'vit.encoder.layer.{i}.intermediate.dense')
                else: # fc2
                    hf_key = mlp_key.replace(f'blocks.{i}.mlp.fc2', f'vit.encoder.layer.{i}.output.dense')
                hf_state_dict[hf_key] = simmim_state_dict[mlp_key]
                used_keys.add(mlp_key)

        # Layer Scale Gammas (未映射)
        for gamma_key in [f'{simmim_block_prefix}gamma_1', f'{simmim_block_prefix}gamma_2']:
            if gamma_key in simmim_state_dict:
                # 这些键存在，但没有对应的 HF ViT 键，因此它们不会被添加到 hf_state_dict
                # 但我们可以记录它们被发现但未使用
                pass # gamma_key will be in simmim_state_dict but not in used_keys or hf_state_dict

    # --- Final Layer Normalization ---
    for norm_key in ['norm.weight', 'norm.bias']:
        if norm_key in simmim_state_dict:
            hf_state_dict[f'vit.layernorm.{norm_key.split(".")[-1]}'] = simmim_state_dict[norm_key]
            used_keys.add(norm_key)

    # --- Report on mapped and unmapped keys ---
    all_simmim_keys = set(simmim_state_dict.keys())
    unmapped_keys = all_simmim_keys - used_keys
    print(f"--- Weight Mapping Report ---")
    print(f"Total SimMIM keys: {len(all_simmim_keys)}")
    print(f"Mapped (Used) keys: {len(used_keys)}")
    print(f"Unmapped keys: {len(unmapped_keys)}")
    if unmapped_keys:
        print(f"Unmapped keys (will be discarded): {sorted(list(unmapped_keys))}")
    print(f"Total HF ViT keys created: {len(hf_state_dict)}")
    print(f"--- End Report ---")

    return hf_state_dict

# --- 示例用法 ---
if __name__ == "__main__":
    # 加载 SimMIM 风格的权重
    weights_path='/data/scy/SCY/SonoVLM_V2/models/usfm/USFM_latest.pth'
    usfm = USFM_VisionTransformer(norm_layer=partial(nn.LayerNorm, eps=1e-6))
    state_dict = torch.load(Path(weights_path), map_location="cpu")
    usfm.load_state_dict(state_dict, strict=True)
    simmim_weights = torch.load("/XYFS01/HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/models/usfm/USFM_latest.pth", map_location='cpu')

    # 假设 simmim_weights 是包含您列表中键的字典
    hf_weights = map_simmim_to_hf(simmim_weights)

    # 保存转换后的权重
    # torch.save(hf_weights, "path_to_hf_vit_checkpoint.bin")
    # 例如，保存为 'converted_hf_weights.pth'
    torch.save(hf_weights, "/XYFS01/HOME/sysu_ldchen/sysu_ldchen_1/SonoVLM_V2/models/usfm/converted_hf_weights.pth")
    print("Converted weights saved.")
    pass