# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import glob

import torch
from huggingface_hub import file_exists, hf_hub_download, snapshot_download
from safetensors import safe_open

from transformers import (
    AddedToken,
    AutoConfig,
    AutoImageProcessor,
    AutoTokenizer,
    LlavaConfig,
    LlavaProcessor,
    SiglipVisionConfig,
)
from modeling_llavamed import LlavaForConditionalGeneration

EPILOG_TXT = """Example:
    python transformers/src/transformers/models/llava/convert_llava_weights_to_hf.py --text_model_id lmsys/vicuna-7b-v1.5 --vision_model_id openai/clip-vit-large-patch14-336 --output_hub_path org/llava-v1.5-7b-conv --old_state_dict_id liuhaotian/llava-v1.5-7b

Example for creating the old state dict file with Python:

    import torch
    from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

    # load model
    kwargs = {"device_map": "auto", "dtype": torch.float16}
    model = LlavaLlamaForCausalLM.from_pretrained("liuhaotian/llava-v1.5-7b", **kwargs)

    # load vision tower
    model.get_vision_tower().load_model()

    # Save state dict
    torch.save(model.state_dict(), "tmp/hf_models/llava-v1.5-7b/model_state_dict.bin")
"""

KEYS_TO_MODIFY_MAPPING = {
    "model.vision_tower.": "",
    ".vision_resampler": "",  # all lmms-lab models do avg pooling, so no vision_resampler
    "model.mm_projector": "multi_modal_projector",
    "model": "model.model",
    "vision_model.model": "vision_model",
    "lm_head": "language_model.lm_head",
    "model.model": "language_model.model",
    "multi_modal_projector.0": "multi_modal_projector.linear_1",
    "multi_modal_projector.2": "multi_modal_projector.linear_2",
}


def load_original_state_dict(directory_path):
    # directory_path = snapshot_download(repo_id=model_id, allow_patterns=["*.safetensors"])

    original_state_dict = {}
    for path in glob.glob(f"{directory_path}/*"):
        if path.endswith(".safetensors"):
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    original_state_dict[key] = f.get_tensor(key)

    # tied weights so lm.head is not saved. Let's clone to load state dict
    if "lm_head.weight" not in original_state_dict:
        original_state_dict["lm_head.weight"] = original_state_dict["model.embed_tokens.weight"].clone()

    if "model.image_newline" in original_state_dict:
        # not used in the original implementation because "merge_type=flat"
        del original_state_dict["model.image_newline"]
    return original_state_dict


# used only for llava-interlave
# for ex: Qwen/Qwen1.5-0.5B-Chat google/siglip-so400m-patch14-384 lmms-lab/llava-next-interleave-qwen-0.5b
# def convert_state_dict_to_hf(state_dict):
#     new_state_dict = {}
#     for key, value in state_dict.items():
#         if key.endswith(".inv_freq"):
#             continue
#         for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
#             if key_to_modify in key:
#                 key = key.replace(key_to_modify, new_key)
#
#         new_state_dict[key] = value
#     return new_state_dict

def convert_state_dict_to_hf(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.endswith(".inv_freq"):
            continue

        # ✅ 按顺序匹配，只替换一次
        if key.startswith("model.vision_tower."):
            new_key = key.replace("model.vision_tower.vision_tower", "model.vision_tower")#'model.vision_tower.vision_tower.vision_model.embeddings.class_embedding'
        elif key.startswith("model.mm_projector.0."):
            new_key = key.replace("model.mm_projector.0.", "model.multi_modal_projector.linear_1.")
        elif key.startswith("model.mm_projector.2."):
            new_key = key.replace("model.mm_projector.2.", "model.multi_modal_projector.linear_2.")
        elif key.startswith("model.embed_tokens."):
            new_key = key.replace("model.embed_tokens.", "model.language_model.embed_tokens.")
        elif key.startswith("model.layers."):
            new_key = key.replace("model.layers.", "model.language_model.layers.")
        elif key.startswith("model.norm."):
            new_key = key.replace("model.norm.", "model.language_model.norm.")
        # elif key.startswith("lm_head."):
        #     new_key = key.replace("lm_head.", "language_model.lm_head.")
        else:
            new_key = key  # 保留未匹配的 key（如 vision_tower 内部）

        new_state_dict[new_key] = value

    return new_state_dict

def convert_llava_llama_to_hf(text_model_id, vision_model_id, output_path, old_state_dict_id):
    torch.set_default_dtype(torch.float16)
    text_config = AutoConfig.from_pretrained(text_model_id)

    tokenizer = AutoTokenizer.from_pretrained(text_model_id)
    tokenizer.add_tokens(AddedToken("<image>", special=True, normalized=False), special_tokens=True)
    if "Qwen" not in text_model_id:  # qwen already has a pad token
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    image_processor = AutoImageProcessor.from_pretrained(vision_model_id)
    processor = LlavaProcessor(tokenizer=tokenizer, image_processor=image_processor)

    if "siglip" in vision_model_id:
        vision_config = SiglipVisionConfig(
            hidden_size=1152,
            image_size=384,
            intermediate_size=4304,
            num_attention_heads=16,
            num_hidden_layers=26,
            patch_size=14,
            vision_use_head=False,
        ).to_dict()
    else:
        vision_config = None

    config = LlavaConfig(
        text_config=text_config,
        vision_config=vision_config,
    )

    # llms-lab interleave models do not use any selection strategy except for last hidden state
    if "Qwen" in text_model_id:
        config.image_token_id = 151646
        if "siglip" in vision_model_id:
            config.vision_feature_select_strategy = "full"
            config.vision_feature_layer = -1
    else:
        config.pad_token_id = 32001
        config.image_token_id = 32000

    with torch.device("meta"):
        model = LlavaForConditionalGeneration(config)

    # Some llava variants like microsoft/llava-med-v1.5-mistral-7b use safetensors to store weights
    # if file_exists(old_state_dict_id, "model_state_dict.bin"):
    #     state_dict_path = hf_hub_download(old_state_dict_id, "model_state_dict.bin")
    #     state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
    # else:
    state_dict = load_original_state_dict(old_state_dict_id)

    expected_keys = set(model.state_dict().keys())
    provided_keys = set(state_dict.keys())

    missing_keys = expected_keys - provided_keys
    unexpected_keys = provided_keys - expected_keys

    print(f"✅ Total expected keys: {len(expected_keys)}")
    print(f"✅ Provided keys: {len(provided_keys)}")
    if missing_keys:
        print("⚠️ Missing keys (not in original weights):")
        for k in sorted(missing_keys):
            print(f"  - {k}")
    else:
        print("✅ No missing keys.")

    if unexpected_keys:
        print("❓ Unexpected keys (in original but not in HF model):")
        for k in sorted(unexpected_keys):
            print(f"  - {k}")

    state_dict = convert_state_dict_to_hf(state_dict)
    model.load_state_dict(state_dict, strict=True, assign=True)

    pre_expansion_embeddings = model.language_model.embed_tokens.weight.data
    mu = torch.mean(pre_expansion_embeddings, dim=0).float()
    n = pre_expansion_embeddings.size()[0]
    sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
    dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5 * sigma)

    # We add an image token so we resize the model and pad to 64 for performance reasons
    pad_shape = 64
    vocab_size = config.text_config.vocab_size
    model.resize_token_embeddings(config.text_config.vocab_size + 2, pad_shape)
    model.language_model.embed_tokens.weight.data[vocab_size:] = torch.stack(
        tuple(dist.sample() for _ in range(model.language_model.embed_tokens.weight.data[vocab_size:].shape[0])),
        dim=0,
    )
    model.lm_head.weight.data[vocab_size:] = torch.stack(
        tuple(dist.sample() for _ in range(model.lm_head.weight.data[vocab_size:].shape[0])),
        dim=0,
    )
    model.save_pretrained(output_path, safe_serialization=True)

    # 2. 保存 processor（tokenizer + image_processor）
    processor.save_pretrained(output_path)

def main():
    parser = argparse.ArgumentParser(
        epilog=EPILOG_TXT,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--text_model_id",default='/data/scy/SCY/Model_weights/Mistral-7B-Instruct-v0.2',
        help="Hub location of the text model",
    )
    parser.add_argument(
        "--vision_model_id",default='/data/scy/SCY/Model_weights/clip-vit-large-patch14-336',
        help="Hub location of the vision model",
    )
    parser.add_argument(
        "--output_path",default='/data/scy/SCY/SonoVLM_V2/llavamed/revised',
        help="Location on the hub of the converted model",
    )
    parser.add_argument(
        "--old_state_dict_id",default='/data/scy/SCY/Model_weights/llava-med-v1.5-mistral-7b-original',
        help="Location on the hub of the raw state dict of the original model. The filename needs to be `model_state_dict.bin`",
    )
    args = parser.parse_args()
    convert_llava_llama_to_hf(args.text_model_id, args.vision_model_id, args.output_path, args.old_state_dict_id)


if __name__ == "__main__":
    main()
