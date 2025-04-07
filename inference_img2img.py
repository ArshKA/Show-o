# coding=utf-8
# Copyright 2024 NUS Show Lab.
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

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import wandb
from models import Showo, MAGVITv2, get_mask_chedule
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next
from training.utils import get_config, flatten_omega_conf, image_transform
from transformers import AutoTokenizer
import torch.nn.functional as F

def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")

if __name__ == '__main__':
    config = get_config()

    resume_wandb_run = config.wandb.resume
    run_id = config.wandb.get("run_id", None)
    if run_id is None:
        resume_wandb_run = False
        run_id = wandb.util.generate_id()
        config.wandb.run_id = run_id

    wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}

    wandb.init(
        project="demo",
        name=config.experiment.name + '_img2img',
        config=wandb_config,
    )

    # load from users passed arguments
    if config.get("image_path", None) is not None:
        config.image_path = config.image_path
    config.training.batch_size = config.batch_size if config.get("batch_size", None) is not None else config.training.batch_size
    config.training.guidance_scale = config.guidance_scale if config.get("guidance_scale", None) is not None else config.training.guidance_scale
    config.training.generation_timesteps = config.generation_timesteps if config.get("generation_timesteps", None) is not None else config.training.generation_timesteps
    config.noise_level = config.get("noise_level", 0.5)  # Default noise level if not specified
    # load from users passed arguments

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side="left")

    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
                                       ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)

    # Load VQ model for encoding/decoding images
    vq_model = get_vq_model_class(config.model.vq_model.type)
    vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()

    # Load Showo model for generation
    model = Showo.from_pretrained(config.model.showo.pretrained_model_path).to(device)
    model.eval()

    mask_token_id = model.config.mask_token_id

    # Load and preprocess input image
    print(f"Loading image from {config.image_path}")
    input_image = Image.open(config.image_path).convert("RGB")
    input_image_tensor = image_transform(input_image, resolution=config.dataset.params.resolution).to(device)
    input_image_tensor = input_image_tensor.unsqueeze(0).repeat(config.training.batch_size, 1, 1, 1)
    
    # Log original image
    original_image = torch.clamp((input_image_tensor[0] + 1.0) / 2.0, min=0.0, max=1.0)
    original_image *= 255.0
    original_image = original_image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    original_pil = Image.fromarray(original_image)
    wandb_images = [wandb.Image(original_pil, caption="Original Image")]

    # Get image tokens from VAE
    print("Encoding image to tokens...")
    image_tokens = vq_model.get_code(input_image_tensor) + len(uni_prompting.text_tokenizer)
    
    # Apply noise to image tokens based on noise_level
    print(f"Applying noise with level {config.noise_level}...")
    num_tokens_to_mask = int(config.model.showo.num_vq_tokens * config.noise_level)
    
    # Create batch of masks for random tokens
    masked_image_tokens = []
    for i in range(config.training.batch_size):
        # Create a mask for random tokens
        random_indices = torch.randperm(config.model.showo.num_vq_tokens)[:num_tokens_to_mask]
        mask = torch.zeros(config.model.showo.num_vq_tokens, dtype=torch.bool, device=device)
        mask[random_indices] = True
        
        # Apply mask to image tokens
        batch_tokens = image_tokens[i:i+1].clone()
        batch_tokens[0, mask] = mask_token_id
        masked_image_tokens.append(batch_tokens)
    
    masked_image_tokens = torch.cat(masked_image_tokens, dim=0)
    
    # Split the prompt if multiple prompts provided
    if isinstance(config.prompt, str):
        prompts = [config.prompt] * config.training.batch_size
    else:
        prompts = config.prompt[:config.training.batch_size]
        
    # Prepare input for the model
    input_ids, _ = uni_prompting((prompts, masked_image_tokens), 't2i_gen')
    
    # Prepare for conditional guidance
    if config.training.guidance_scale > 0:
        uncond_input_ids, _ = uni_prompting(([''] * len(prompts), masked_image_tokens), 't2i_gen')
        attention_mask = create_attention_mask_predict_next(
            torch.cat([input_ids, uncond_input_ids], dim=0),
            pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
            soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
            eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
            rm_pad_in_image=True
        )
    else:
        attention_mask = create_attention_mask_predict_next(
            input_ids,
            pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
            soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
            eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
            rm_pad_in_image=True
        )
        uncond_input_ids = None
    
    # Get mask schedule
    if config.get("mask_schedule", None) is not None:
        schedule = config.mask_schedule.schedule
        args = config.mask_schedule.get("params", {})
        mask_schedule = get_mask_chedule(schedule, **args)
    else:
        mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))
    
    # Generate new image tokens
    print("Generating new image...")
    with torch.no_grad():
        gen_token_ids = model.t2i_generate(
            input_ids=input_ids,
            uncond_input_ids=uncond_input_ids,
            attention_mask=attention_mask,
            guidance_scale=config.training.guidance_scale,
            temperature=config.training.get("generation_temperature", 1.0),
            timesteps=config.training.generation_timesteps,
            noise_schedule=mask_schedule,
            noise_type=config.training.get("noise_type", "mask"),
            seq_len=config.model.showo.num_vq_tokens,
            uni_prompting=uni_prompting,
            config=config,
        )
    
    # Clamp token ids to valid range
    gen_token_ids = torch.clamp(gen_token_ids, max=config.model.showo.codebook_size - 1, min=0)
    
    # Decode tokens to image
    generated_images = vq_model.decode_code(gen_token_ids)
    
    # Process images for display
    generated_images = torch.clamp((generated_images + 1.0) / 2.0, min=0.0, max=1.0)
    generated_images *= 255.0
    generated_images = generated_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    
    # Convert to PIL images
    pil_images = [Image.fromarray(image) for image in generated_images]
    
    # Log images to wandb
    wandb_images.extend([wandb.Image(image, caption=f"Generated: {prompts[i]}") 
                        for i, image in enumerate(pil_images)])
    wandb.log({"generated_images": wandb_images}, step=0)
    
    print("Done!") 