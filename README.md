# EchoVLM (paper implementation)

Official PyTorch implementation of the model described in  
**"[EchoVLM: Dynamic Mixture-of-Experts Vision-Language Model for Universal Ultrasound Intelligence](https://arxiv.org/abs/2509.14977)"**.

## 🤖 Model Details

| Item        | Value                                           |
|-------------|-------------------------------------------------|
| Paper       | [arXiv:2509.14977](https://arxiv.org/abs/2509.14977) |
| Authors     | Chaoyin She¹, Ruifang Lu²                        |
| Code        | [GitHub repo](https://github.com/Asunatan/EchoVLM) |
| Model Hub   | [Hugging Face](https://huggingface.co/chaoyinshe/EchoVLM) |

## 🔄 Updates
- **Coming soon**: V2 with Chain-of-Thought reasoning and reinforcement learning enhancements—full training & inference code plus benchmark test-set will be fully open-sourced.
- **Dec 26, 2025**: During inference, ensure that `output_router_logits` is in a `false` state.
- **Dec 21, 2025**:
  
  `TypeError: _compute_default_rope_parameters() got an unexpected keyword argument 'rope_type'`
  
The latest transformers library (v4.57.3) no longer supports this code. You can either downgrade the library version (v4.47.0), or simply copy the `Qwen2VLRotaryEmbedding` function from v4.57.3 and pass it to your config—this is quite straightforward. If you need help, consulting the chat teacher is a good option. Alternatively, you can use the v2 version based on lingshu's implementation, which uses higher-quality data and offers better performance.
- <img width="1011" height="738" alt="image" src="https://github.com/user-attachments/assets/ebe826f4-6057-4069-87e4-b3abc6f5d2e4" />

- **Dec 1, 2025**: To better promote development in this field, we've open-sourced our latest instruction fine-tuned model based on Lingshu-7B. Essentially built on Qwen2.5VL, it enjoys a better ecosystem—for example, it can seamlessly leverage vLLM for accelerated inference. Released model weights on [Hugging Face](https://huggingface.co/chaoyinshe/EchoVLM_V2_lingshu_base_7b_instruct_preview).  
- **Sep 21, 2025**: The full, uncleaned model codebase is now open-sourced on GitHub!
- **Sep 19, 2025**: Released model weights on [Hugging Face](https://huggingface.co/chaoyinshe/EchoVLM).  
- **Sep 17, 2025**: Paper published on [arXiv](https://arxiv.org/abs/2509.14977).


## 🚀 Quick Start
### Using 🤗  Transformers to Chat

Here we show a code snippet to show you how to use the chat model with `transformers` and `qwen_vl_utils`:

```python
from EchoVLM import Qwen2VLMOEForConditionalGeneration
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# ===== 1. Load model & processor =====
model = Qwen2VLMOEForConditionalGeneration.from_pretrained(
    "chaoyinshe/EchoVLM",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",   # faster & memory-efficient
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("chaoyinshe/EchoVLM")
# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "An ultrasound image",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```
<details>
<summary>Multi image inference</summary>

```python
# Messages containing multiple images and a text query
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "ultrasound image 1"},
            {"type": "image", "image": "ultrasound image 2"},
            {"type": "text", "text": "帮我给出超声报告"},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```
</details>
<details>
<summary>Batch inference</summary>

```python
# Sample messages for batch inference
messages1 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "file:///path/to/image1.jpg"},
            {"type": "image", "image": "file:///path/to/image2.jpg"},
            {"type": "text", "text": "This patient has a hypoechoic nodule in the left breast. What is the next step in treatment?"},
        ],
    }
]
messages2 = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who are you?"},
]
# Combine messages for batch processing
messages = [messages1, messages2]

# Preparation for batch inference
texts = [
    processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    for msg in messages
]
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=texts,
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Batch Inference
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_texts = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_texts)
```
</details>

## 📌 Citation

If you use this model or code in your research, please cite:

```bibtex
@misc{she2025echovlmdynamicmixtureofexpertsvisionlanguage,
      title={EchoVLM: Dynamic Mixture-of-Experts Vision-Language Model for Universal Ultrasound Intelligence}, 
      author={Chaoyin She and Ruifang Lu and Lida Chen and Wei Wang and Qinghua Huang},
      year={2025},
      eprint={2509.14977},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.14977}, 
}
