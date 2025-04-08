import wandb
import torch
import numpy as np
from PIL import Image
from accelerate.logging import get_logger
from models import Showo, MAGVITv2, CLIPVisionTower, get_mask_chedule
from editing_training.prompting_editing import create_attention_mask_predict_next
from torch.utils.data import DataLoader

logger = get_logger(__name__)

def log_visualizations(
    model,
    vq_model,
    train_dataloader,
    test_dataloader,
    accelerator,
    uni_prompting,
    config,
    step,
    num_samples=2,
):
    """
    Logs visualization samples to wandb showing:
    - Source image
    - Generated edited image
    - Ground truth edited image
    
    Clearly separates train and test samples into different sections.
    """
    model.eval()
    unwrapped_model = accelerator.unwrap_model(model)
    mask_token_id = unwrapped_model.mask_token_id

    # Create a function to process a batch and generate visualizations
    def process_batch(batch, dataset_type):
        source_images = torch.stack(batch['source_images']).to(accelerator.device)
        edited_images = torch.stack(batch['edited_images']).to(accelerator.device)
        texts = batch['texts']

        with torch.no_grad():
            # Prepare inputs
            offset = len(uni_prompting.text_tokenizer)
            source_tokens = vq_model.get_code(source_images) + offset
            
            # Create fully masked edited tokens
            batch_size = source_images.shape[0]
            masked_edited_tokens = torch.full(
                (batch_size, config.model.showo.num_vq_tokens),
                mask_token_id,
                device=accelerator.device
            )

            # Create model inputs using universal prompting
            input_ids, _, _ = uni_prompting(
                (texts, source_tokens, masked_edited_tokens),
                'edit_gen'
            )

            # Create attention mask
            attention_mask = create_attention_mask_predict_next(
                input_ids,
                pad_id=int(uni_prompting.sptids_dict['[PAD]']),
                soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                rm_pad_in_image=True,
                return_inverse_mask=True
            ).to(unwrapped_model.dtype)

            # Generate edited tokens
            generated_token_ids = unwrapped_model.edit_generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                config=config,
                guidance_scale=config.training.get("guidance_scale", 3.0),
                timesteps=config.training.get("generation_timesteps", 100),
                temperature=config.training.get("temperature", 1.0),
                noise_schedule=get_mask_chedule(config.training.get("mask_schedule", "cosine")),
            )

            # Extract edited tokens (last N tokens in sequence)
            generated_edited_tokens = generated_token_ids[:, -config.model.showo.num_vq_tokens:]

            # Decode all images
            generated_images = vq_model.decode_code(generated_edited_tokens)

        # Prepare visualization grid
        wandb_images = []
        for i in range(min(num_samples, batch_size)):
            # Convert tensors to PIL images
            def tensor_to_pil(x):
                x = (x + 1) / 2  # [-1,1] -> [0,1]
                x = x.cpu().permute(1, 2, 0).numpy() * 255
                x = np.clip(x, 0, 255)
                return Image.fromarray(x.astype('uint8'))

            source_pil = tensor_to_pil(source_images[i])
            edited_pil = tensor_to_pil(edited_images[i])
            generated_pil = tensor_to_pil(generated_images[i])

            # Create grid with captions and dataset type
            wandb_images.append(wandb.Image(
                source_pil,
                caption=f"{dataset_type} Source: {texts[i][:50]}..."
            ))
            wandb_images.append(wandb.Image(
                generated_pil,
                caption=f"{dataset_type} Generated: {texts[i][:50]}..."
            ))
            wandb_images.append(wandb.Image(
                edited_pil,
                caption=f"{dataset_type} Ground Truth: {texts[i][:50]}..."
            ))

        return wandb_images

    try:
        # Use dedicated visualization dataloaders for each to ensure
        # we get unique, representative samples from both
        train_vis_dataloader = DataLoader(
            train_dataloader.dataset,
            batch_size=num_samples,
            shuffle=True,
            num_workers=1,
            collate_fn=train_dataloader.collate_fn
        ) if train_dataloader else None
        
        test_vis_dataloader = test_dataloader if test_dataloader else None
        
        # Process train samples
        train_images = []
        if train_vis_dataloader:
            train_batch = next(iter(train_vis_dataloader))
            train_images = process_batch(train_batch, "TRAIN")
        
        # Process test samples
        test_images = []
        if test_vis_dataloader:
            test_batch = next(iter(test_vis_dataloader))
            test_images = process_batch(test_batch, "TEST")
        
        # Log train and test images separately for clear distinction in wandb
        if train_images:
            accelerator.log({"train_visualizations": train_images}, step=step)
        
        if test_images:
            accelerator.log({"test_visualizations": test_images}, step=step)
        
    except Exception as e:
        logger.error(f"Error in visualization: {e}")
    finally:
        model.train()