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
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next, create_attention_mask_lvg
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
        name=config.experiment.name + '_seq2img',
        config=wandb_config,
    )

    # load from users passed arguments
    if config.get("image_path", None) is not None:
        config.image_path = config.image_path
    config.training.batch_size = config.batch_size if config.get("batch_size", None) is not None else config.training.batch_size
    config.training.guidance_scale = config.guidance_scale if config.get("guidance_scale", None) is not None else config.training.guidance_scale
    config.training.generation_timesteps = config.generation_timesteps if config.get("generation_timesteps", None) is not None else config.training.generation_timesteps
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

    # Load and preprocess input image (source image)
    print(f"Loading source image from {config.image_path}")
    input_image = Image.open(config.image_path).convert("RGB")
    input_image_tensor = image_transform(input_image, resolution=config.dataset.params.resolution).to(device)
    input_image_tensor = input_image_tensor.unsqueeze(0).repeat(config.training.batch_size, 1, 1, 1)
    
    # Log original image
    original_image = torch.clamp((input_image_tensor[0] + 1.0) / 2.0, min=0.0, max=1.0)
    original_image *= 255.0
    original_image = original_image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    original_pil = Image.fromarray(original_image)
    wandb_images = [wandb.Image(original_pil, caption="Source Image")]

    # Get image tokens from VAE for the source image
    print("Encoding source image to tokens...")
    source_image_tokens = vq_model.get_code(input_image_tensor) + len(uni_prompting.text_tokenizer)
    
    # Create a new set of masked tokens for the target image
    target_image_tokens = torch.ones((config.training.batch_size, config.model.showo.num_vq_tokens), 
                                    dtype=torch.long, device=device) * mask_token_id
    
    # Split the prompt if multiple prompts provided
    if isinstance(config.prompt, str):
        prompts = [config.prompt] * config.training.batch_size
    else:
        prompts = config.prompt[:config.training.batch_size]
    
    # Create a combined prompt that includes the source image context
    combined_prompts = []
    for i, prompt in enumerate(prompts):
        # Create a prompt that references the source image and describes the desired output
        combined_prompts.append(f"{prompt}")
    
    # Use the lvg_gen (language visual generation) task which supports text + image conditioning
    print("Preparing inputs for generation...")
    input_ids, _ = uni_prompting((combined_prompts, source_image_tokens), 'lvg_gen')
    
    # Get special token IDs and move them to the correct device
    def to_tensor(val, dtype=torch.long, device=device):
        return val.clone().detach().to(dtype=dtype, device=device)
         
    sov_token_id = to_tensor(uni_prompting.sptids_dict['<|sov|>'])
    eov_token_id = to_tensor(uni_prompting.sptids_dict['<|eov|>'])
    pad_token_id = to_tensor(uni_prompting.sptids_dict['<|pad|>'])
    soi_token_id = to_tensor(uni_prompting.sptids_dict['<|soi|>'])
    eoi_token_id = to_tensor(uni_prompting.sptids_dict['<|eoi|>'])
    
    # Prepare for conditional guidance
    if config.training.guidance_scale > 0:
        # Empty prompts for unconditional guidance
        uncond_prompts = [""] * len(combined_prompts)
        uncond_input_ids, _ = uni_prompting((uncond_prompts, source_image_tokens), 'lvg_gen')
        
        # Now create empty tokens for the target image (these will be fully generated)
        new_input_ids = []
        new_uncond_input_ids = []
        for i in range(config.training.batch_size):
            # Append target image tokens to be generated
            new_input_ids.append(torch.cat([
                input_ids[i:i+1], 
                # Add a special token (e.g., <|sov|>) to separate source from target
                torch.ones(1, 1, dtype=torch.long, device=device) * sov_token_id,
                target_image_tokens[i:i+1],
                # End with special token
                torch.ones(1, 1, dtype=torch.long, device=device) * eov_token_id
            ], dim=1))
            
            new_uncond_input_ids.append(torch.cat([
                uncond_input_ids[i:i+1],
                # Add a special token to separate source from target
                torch.ones(1, 1, dtype=torch.long, device=device) * sov_token_id,
                target_image_tokens[i:i+1],
                # End with special token
                torch.ones(1, 1, dtype=torch.long, device=device) * eov_token_id
            ], dim=1))
        
        input_ids = torch.cat(new_input_ids, dim=0)
        uncond_input_ids = torch.cat(new_uncond_input_ids, dim=0)
        
        attention_mask = create_attention_mask_lvg(
            torch.cat([input_ids, uncond_input_ids], dim=0),
            pad_id=int(pad_token_id.item()),
            soi_id=int(soi_token_id.item()),
            eoi_id=int(eoi_token_id.item())
        )
    else:
        # Similar approach for when not using guidance
        new_input_ids = []
        for i in range(config.training.batch_size):
            new_input_ids.append(torch.cat([
                input_ids[i:i+1],
                # Add a special token to separate source from target
                torch.ones(1, 1, dtype=torch.long, device=device) * sov_token_id,
                target_image_tokens[i:i+1],
                # End with special token
                torch.ones(1, 1, dtype=torch.long, device=device) * eov_token_id
            ], dim=1))
        
        input_ids = torch.cat(new_input_ids, dim=0)
        uncond_input_ids = None
        
        attention_mask = create_attention_mask_lvg(
            input_ids,
            pad_id=int(pad_token_id.item()),
            soi_id=int(soi_token_id.item()),
            eoi_id=int(eoi_token_id.item())
        )
    
    # Get mask schedule
    if config.get("mask_schedule", None) is not None:
        schedule = config.mask_schedule.schedule
        args = config.mask_schedule.get("params", {})
        mask_schedule = get_mask_chedule(schedule, **args)
    else:
        mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))
    
    # Find the position where the target image starts
    # It will be after the <|sov|> token that follows the source image
    # We need to specify which part of the sequence to actually generate
    target_start_indices = []
    for i in range(input_ids.shape[0]):
        # Find the last occurrence of <|sov|> token
        sov_indices = (input_ids[i] == sov_token_id).nonzero().flatten()
        if len(sov_indices) > 0:
            target_start_indices.append(sov_indices[-1].item() + 1)
        else:
            target_start_indices.append(input_ids.shape[1] - config.model.showo.num_vq_tokens - 1)
    
    # Generate the target image tokens
    from debugging import decode_input_ids
    print(decode_input_ids(input_ids[0], tokenizer=tokenizer))
    print("Generating new image based on source image and text...")
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
            start_position=target_start_indices[0]  # Start generating from the target image position
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
    
    # Log image pairs to wandb (source → generated)
    for i, (gen_img, prompt) in enumerate(zip(pil_images, prompts)):
        pair_images = [
            wandb.Image(original_pil, caption="Source"),
            wandb.Image(gen_img, caption=f"Generated: {prompt}")
        ]
        wandb.log({f"sequence_pair_{i}": pair_images}, step=0)
    
    # Also log all images in a single panel
    wandb_images.extend([wandb.Image(image, caption=f"Generated: {prompts[i]}") 
                        for i, image in enumerate(pil_images)])
    wandb.log({"all_images": wandb_images}, step=0)
    
    print("Done!") 