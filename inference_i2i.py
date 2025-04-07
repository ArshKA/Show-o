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
from models import Showo, MAGVITv2, CLIPVisionTower, get_mask_chedule
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next, create_attention_mask_for_mmu_vit
from training.utils import get_config, flatten_omega_conf, image_transform
from transformers import AutoTokenizer, CLIPImageProcessor
import torch.nn.functional as F

def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")

def get_image_description(model, image, vision_tower, clip_image_processor, device, max_length=100):
    """
    Get a text description of the input image using the model's multimodal understanding capabilities.
    """
    # Process the image through CLIP
    pixel_values = clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0].to(device)
    
    # Get image embeddings
    image_embeddings = vision_tower(pixel_values[None])
    image_embeddings = model.mm_projector(image_embeddings)
    
    # Create a prompt to ask for a description
    describe_prompt = "Describe this image in detail."
    
    # Use the MMU capabilities to generate a description
    # This implementation depends on the model's specific MMU interface
    with torch.no_grad():
        # Simplified example - you may need to adjust this based on your model's API
        input_ids = torch.tensor([model.tokenizer.encode(describe_prompt)]).to(device)
        
        # Create combined embeddings with image first
        text_embeddings = model.showo.model.embed_tokens(input_ids)
        combined_embeddings = torch.cat([image_embeddings, text_embeddings], dim=1)
        
        # Generate a description
        output_ids = model.generate(
            inputs_embeds=combined_embeddings,
            max_length=max_length,
            do_sample=False
        )
        
        description = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return description

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
        name=config.experiment.name + '_i2i',
        config=wandb_config,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side="left")

    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                    special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
                                    ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)

    vq_model = get_vq_model_class(config.model.vq_model.type)
    vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()

    # Load vision tower for image understanding
    vision_tower_name = "openai/clip-vit-large-patch14-336"
    vision_tower = CLIPVisionTower(vision_tower_name).to(device)
    clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)
    vision_tower.eval()

    model = Showo.from_pretrained(config.model.showo.pretrained_model_path).to(device)
    model.eval()

    mask_token_id = model.config.mask_token_id

    # Load from users passed arguments (similar to t2i)
    if config.get("validation_prompts_file", None) is not None:
        config.dataset.params.validation_prompts_file = config.validation_prompts_file
    config.training.batch_size = config.batch_size
    config.training.guidance_scale = config.guidance_scale
    config.training.generation_timesteps = config.generation_timesteps

    # Image-to-image generation mode
    if config.mode == 'i2i':
        # Load input image
        input_image_path = config.input_image_path
        input_image_ori = Image.open(input_image_path).convert("RGB")
        
        # Process input image for VQ model (for visualization)
        input_image = image_transform(input_image_ori, resolution=config.dataset.params.resolution).to(device)
        input_image = input_image.unsqueeze(0)
        
        # Get image tokens from the input image
        input_tokens = vq_model.get_code(input_image)[0] + len(uni_prompting.text_tokenizer)
        
        description = "This is an image of " + config.get("description", "a photograph")
        
        # Get prompt from config and enhance it with the description
        user_prompt = config.prompt
        prompts = [f"{description}. {user_prompt}"] * config.training.batch_size
        print(f"Enhanced prompt: {prompts[0]}")
        
        # Get mask schedule
        if config.get("mask_schedule", None) is not None:
            schedule = config.mask_schedule.schedule
            args = config.mask_schedule.get("params", {})
            mask_schedule = get_mask_chedule(schedule, **args)
        else:
            mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))
        
        # Approach: Use some tokens from original image as initialization
        # Determine how much of the original image to preserve
        preserve_ratio = config.get("preserve_ratio", 0.5)  # default to preserving 50%
        
        # Reshape input tokens to 2D grid format (assuming square tokens grid)
        tokens_side = int(input_tokens.shape[0] ** 0.5)
        tokens_2d = input_tokens.reshape(tokens_side, tokens_side)
        
        # Create a mask prioritizing central content (commonly the subject)
        h, w = tokens_side, tokens_side
        center_h, center_w = h // 2, w // 2
        mask = torch.ones_like(tokens_2d, dtype=torch.float)
        
        # Create a distance map from center (for central content preservation)
        for i in range(h):
            for j in range(w):
                # Calculate distance from center
                dist = ((i - center_h) ** 2 + (j - center_w) ** 2) ** 0.5
                # Normalize by max possible distance
                max_dist = ((0 - center_h) ** 2 + (0 - center_w) ** 2) ** 0.5
                # More likely to preserve central content
                mask[i, j] = dist / max_dist
        
        # Flatten and sort mask to determine which tokens to preserve
        flat_mask = mask.reshape(-1)
        num_preserved = int(preserve_ratio * len(flat_mask))
        _, preserve_indices = torch.topk(flat_mask, num_preserved, largest=False)
        
        # Create masked image tokens
        batch_size = len(prompts)
        image_tokens = torch.ones((batch_size, config.model.showo.num_vq_tokens),
                                 dtype=torch.long, device=device) * mask_token_id
        
        # Add preserved tokens from original image
        for i in range(batch_size):
            image_tokens[i, preserve_indices] = input_tokens.reshape(-1)[preserve_indices]
        
        # Prepare inputs with enhanced prompts
        input_ids, _ = uni_prompting((prompts, image_tokens), 't2i_gen')
        
        if config.training.guidance_scale > 0:
            uncond_input_ids, _ = uni_prompting(([''] * len(prompts), image_tokens), 't2i_gen')
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
        
        # Generate image tokens
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
        
        # Decode generated tokens
        gen_token_ids = torch.clamp(gen_token_ids, max=config.model.showo.codebook_size - 1, min=0)
        generated_images = vq_model.decode_code(gen_token_ids)
        
        # Create images for visualization (input image and generated image)
        images_to_display = torch.cat([input_image, generated_images], dim=0)
        images_to_display = torch.clamp((images_to_display + 1.0) / 2.0, min=0.0, max=1.0)
        images_to_display *= 255.0
        images_to_display = images_to_display.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        pil_images = [Image.fromarray(image) for image in images_to_display]
        
        # Create captions for the images
        captions = ["Input Image"] + [f"Generated: {prompts[i]}" for i in range(len(prompts))]
        
        # Log to wandb
        wandb_images = [wandb.Image(image, caption=captions[i]) for i, image in enumerate(pil_images)]
        wandb.log({"image_to_image_generation": wandb_images}, step=0)
        
        # Save generated images if output directory is specified
        if hasattr(config, 'output_dir') and config.output_dir:
            os.makedirs(config.output_dir, exist_ok=True)
            for i, image in enumerate(pil_images[1:]):  # Skip input image
                output_path = os.path.join(config.output_dir, f"generated_{i}.png")
                image.save(output_path)
                print(f"Saved generated image to {output_path}")
    
    else:
        print(f"Mode {config.mode} not supported in inference_i2i.py") 