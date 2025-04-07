import os
from PIL import Image
import torch
import wandb
from transformers import AutoTokenizer, CLIPImageProcessor
from models import Showo, MAGVITv2, CLIPVisionTower, get_mask_chedule
from training.prompting_utils import UniversalPrompting, create_attention_mask_for_mmu_vit, create_attention_mask_for_mmu, create_attention_mask_predict_next
from training.utils import get_config, image_transform
from omegaconf import OmegaConf

from llava.llava import conversation as conversation_lib

# Initialize the default conversation format
conversation_lib.default_conversation = conversation_lib.conv_templates["phi1.5"]
SYSTEM_PROMPT = "A chat between a curious user and an artificial intelligence assistant. " \
                "The assistant gives helpful, detailed, and polite answers to the user's questions."
SYSTEM_PROMPT_LEN = 28


def main():
    # Load base config and merge with command-line overrides
    config = get_config()
    cli_conf = OmegaConf.from_cli()
    config = OmegaConf.merge(config, cli_conf)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize models
    tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side="left")
    uni_prompting = UniversalPrompting(
        tokenizer,
        max_text_len=config.dataset.preprocessing.max_seq_length,
        special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>", "<|pad|>", "<|sot|>", "<|eot|>"),
        ignore_id=-100,
        cond_dropout_prob=config.training.cond_dropout_prob
    )

    vision_tower = CLIPVisionTower("openai/clip-vit-large-patch14-336").to(device)
    clip_image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    
    vq_model = MAGVITv2.from_pretrained(config.model.vq_model.vq_model_name).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()

    model = Showo.from_pretrained(config.model.showo.pretrained_model_path).to(device)
    model.eval()

    # Process inputs and determine if they're images or text
    input_items = []
    for item in config.inputs:
        if os.path.isfile(item) and item.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_items.append(('image', item))
        else:
            input_items.append(('text', item))
    
    # Get mask schedule for text-to-image generation
    mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))
    
    # Generate responses based on input type
    generated_content = []
    i = 0
    
    if config.model.showo.w_clip_vit:
        # CLIP-ViT based approach (similar to MMU)
        while i < len(input_items):
            input_type, item = input_items[i]
            
            if input_type == 'image':
                # Process image input
                image_ori = Image.open(item).convert("RGB")
                
                # Get prompt text if available
                if i+1 < len(input_items) and input_items[i+1][0] == 'text':
                    prompt_text = input_items[i+1][1]
                    i += 1  # Skip the text item in the next iteration
                else:
                    prompt_text = "Describe this image."
                
                # Process image with CLIP
                pixel_values = clip_image_processor.preprocess(image_ori, return_tensors="pt")["pixel_values"][0].to(device)
                
                # Get image embeddings
                images_embeddings = vision_tower(pixel_values[None])
                images_embeddings = model.mm_projector(images_embeddings)
                
                # Create conversation prompt
                conv = conversation_lib.default_conversation.copy()
                conv.append_message(conv.roles[0], prompt_text)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()
                
                # Process system prompt
                input_ids_system = uni_prompting.text_tokenizer(SYSTEM_PROMPT, return_tensors="pt", padding="longest").input_ids
                input_ids_system = input_ids_system.to(device)
                
                # Process user prompt
                input_ids = uni_prompting.text_tokenizer(prompt_question.strip(), return_tensors="pt", padding="longest").input_ids
                input_ids = input_ids.to(device)
                
                # Create input IDs with special tokens
                input_ids_llava = torch.cat([
                    (torch.ones(1, 1) * uni_prompting.sptids_dict['<|mmu|>']).to(device),
                    input_ids_system,
                    (torch.ones(1, 1) * uni_prompting.sptids_dict['<|soi|>']).to(device),
                    # Image embeddings will go here
                    (torch.ones(1, 1) * uni_prompting.sptids_dict['<|eoi|>']).to(device),
                    input_ids,
                ], dim=1).long()
                
                # Get text embeddings
                text_embeddings = model.showo.model.embed_tokens(input_ids_llava)
                
                # Combine text and image embeddings
                part1 = text_embeddings[:, :2 + SYSTEM_PROMPT_LEN, :]
                part2 = text_embeddings[:, 2 + SYSTEM_PROMPT_LEN:, :]
                input_embeddings = torch.cat((part1, images_embeddings, part2), dim=1)
                
                # Create attention mask
                attention_mask = create_attention_mask_for_mmu_vit(input_embeddings, 
                                                               system_prompt_len=SYSTEM_PROMPT_LEN)
                
                # Generate response
                cont_toks_list = model.mmu_generate(
                    input_embeddings=input_embeddings,
                    attention_mask=attention_mask[0].unsqueeze(0),
                    max_new_tokens=config.max_new_tokens,
                    top_k=config.top_k,
                    temperature=config.temperature,
                    eot_token=tokenizer.eos_token_id
                )
                
                # Process generated tokens
                cont_toks_list = torch.stack(cont_toks_list).squeeze()[None]
                text = tokenizer.batch_decode(cont_toks_list, skip_special_tokens=True)[0]
                
                # Save results
                generated_content.append(('image', image_ori))
                generated_content.append(('text', text))
            
            elif input_type == 'text':
                # Process as text-to-image generation
                prompt = item
                
                # Create masked image tokens
                mask_token_id = model.config.mask_token_id
                image_tokens = torch.ones((1, config.model.showo.num_vq_tokens),
                                      dtype=torch.long, device=device) * mask_token_id
                
                # Prepare input IDs for text-to-image generation
                input_ids, _ = uni_prompting(([prompt], image_tokens), 't2i_gen')
                
                # Set up for classifier-free guidance if needed
                if config.guidance_scale > 0:
                    uncond_input_ids, _ = uni_prompting(([''], image_tokens), 't2i_gen')
                    attention_mask = create_attention_mask_predict_next(
                        torch.cat([input_ids, uncond_input_ids], dim=0),
                        pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                        soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                        eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                        rm_pad_in_image=True
                    )
                    uncond_input_ids = uncond_input_ids
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
                        guidance_scale=config.guidance_scale if hasattr(config, 'guidance_scale') else 0,
                        temperature=config.temperature if hasattr(config, 'temperature') else 1.0,
                        timesteps=config.generation_timesteps if hasattr(config, 'generation_timesteps') else 18,
                        noise_schedule=mask_schedule,
                        noise_type=config.training.get("noise_type", "mask"),
                        seq_len=config.model.showo.num_vq_tokens,
                        uni_prompting=uni_prompting,
                        config=config,
                    )
                
                # Process and decode the generated image tokens
                gen_token_ids = torch.clamp(gen_token_ids, max=config.model.showo.codebook_size - 1, min=0)
                generated_image = vq_model.decode_code(gen_token_ids)
                
                # Convert tensor to PIL image
                generated_image = torch.clamp((generated_image + 1.0) / 2.0, min=0.0, max=1.0)
                generated_image = generated_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
                generated_image = (generated_image * 255).astype('uint8')
                generated_image = Image.fromarray(generated_image)
                
                # Save results
                generated_content.append(('text', prompt))
                generated_content.append(('image', generated_image))
            
            i += 1
    else:
        # Standard VQ-based approach
        while i < len(input_items):
            input_type, item = input_items[i]
            
            if input_type == 'image':
                # Process image input
                image_ori = Image.open(item).convert("RGB")
                transformed_image = image_transform(image_ori, resolution=config.dataset.params.resolution).to(device)
                
                # Get image tokens
                image_tokens = vq_model.get_code(transformed_image.unsqueeze(0)) + len(tokenizer)
                
                # Get prompt text if available
                if i+1 < len(input_items) and input_items[i+1][0] == 'text':
                    prompt_text = input_items[i+1][1]
                    i += 1  # Skip the text item in the next iteration
                else:
                    prompt_text = "Describe this image."
                
                # Tokenize the text prompt
                input_ids = uni_prompting.text_tokenizer(['USER: \n' + prompt_text + ' ASSISTANT:'])['input_ids']
                input_ids = torch.tensor(input_ids).to(device)
                
                # Create input IDs with special tokens and image tokens
                input_ids = torch.cat([
                    (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|mmu|>']).to(device),
                    (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|soi|>']).to(device),
                    image_tokens,
                    (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|eoi|>']).to(device),
                    (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|sot|>']).to(device),
                    input_ids
                ], dim=1).long()
                
                # Create attention mask
                attention_mask = create_attention_mask_for_mmu(input_ids.to(device), 
                                                           eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']))
                
                # Generate response
                cont_toks_list = model.mmu_generate(
                    input_ids, 
                    attention_mask=attention_mask,
                    max_new_tokens=config.max_new_tokens, 
                    top_k=config.top_k,
                    temperature=config.temperature,
                    eot_token=uni_prompting.sptids_dict['<|eot|>']
                )
                
                # Process generated tokens
                cont_toks_list = torch.stack(cont_toks_list).squeeze()[None]
                text = uni_prompting.text_tokenizer.batch_decode(cont_toks_list, skip_special_tokens=True)[0]
                
                # Save results
                generated_content.append(('image', image_ori))
                generated_content.append(('text', text))
            
            elif input_type == 'text':
                # Process as text-to-image generation
                prompt = item
                
                # Create masked image tokens
                mask_token_id = model.config.mask_token_id
                image_tokens = torch.ones((1, config.model.showo.num_vq_tokens),
                                      dtype=torch.long, device=device) * mask_token_id
                
                # Prepare input IDs for text-to-image generation
                input_ids, _ = uni_prompting(([prompt], image_tokens), 't2i_gen')
                
                # Set up for classifier-free guidance if needed
                if config.guidance_scale > 0:
                    uncond_input_ids, _ = uni_prompting(([''], image_tokens), 't2i_gen')
                    attention_mask = create_attention_mask_predict_next(
                        torch.cat([input_ids, uncond_input_ids], dim=0),
                        pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                        soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                        eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                        rm_pad_in_image=True
                    )
                    uncond_input_ids = uncond_input_ids
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
                        guidance_scale=config.guidance_scale if hasattr(config, 'guidance_scale') else 0,
                        temperature=config.temperature if hasattr(config, 'temperature') else 1.0,
                        timesteps=config.generation_timesteps if hasattr(config, 'generation_timesteps') else 18,
                        noise_schedule=mask_schedule,
                        noise_type=config.training.get("noise_type", "mask"),
                        seq_len=config.model.showo.num_vq_tokens,
                        uni_prompting=uni_prompting,
                        config=config,
                    )
                
                # Process and decode the generated image tokens
                gen_token_ids = torch.clamp(gen_token_ids, max=config.model.showo.codebook_size - 1, min=0)
                generated_image = vq_model.decode_code(gen_token_ids)
                
                # Convert tensor to PIL image
                generated_image = torch.clamp((generated_image + 1.0) / 2.0, min=0.0, max=1.0)
                generated_image = generated_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
                generated_image = (generated_image * 255).astype('uint8')
                generated_image = Image.fromarray(generated_image)
                
                # Save results
                generated_content.append(('text', prompt))
                generated_content.append(('image', generated_image))
            
            i += 1

    # Save and display results
    os.makedirs(config.output_dir, exist_ok=True)
    
    for idx, (content_type, content) in enumerate(generated_content):
        if content_type == 'text':
            print(f"Generated Text: {content}")
            # Save text to file
            with open(os.path.join(config.output_dir, f"generated_{idx}.txt"), 'w') as f:
                f.write(content)
        elif content_type == 'image':
            if isinstance(content, torch.Tensor):
                # This is a generated image tensor
                content = torch.clamp((content + 1.0) / 2.0, min=0.0, max=1.0)
                content = content.squeeze(0).permute(1, 2, 0).cpu().numpy()
                content = (content * 255).astype('uint8')
                content = Image.fromarray(content)
            
            # Save image
            image_path = os.path.join(config.output_dir, f"generated_{idx}.png")
            content.save(image_path)
            print(f"Generated Image saved to: {image_path}")

if __name__ == "__main__":
    main()
