# coding=utf-8
# Copyright 2024 The HuggingFace, NUS Show Lab.
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
import logging
import math
import shutil
import time
from pathlib import Path
from itertools import cycle

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from omegaconf import OmegaConf
import wandb
from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from tqdm import tqdm

from editing_training.data_editing import ImageEditingDataset  # Custom dataset class
from editing_training.prompting_editing import UniversalPrompting, create_attention_mask_predict_next
from editing_training.visualization import log_visualizations  # Import the visualization module
from models import Showo, MAGVITv2, get_mask_chedule
from models.lr_schedulers import get_scheduler
from training.utils import get_config, flatten_omega_conf, mask_or_random_replace_tokens, AverageMeter

logger = get_logger(__name__, log_level="INFO")

def evaluate(
    model,
    test_dataloader,
    accelerator,
    uni_prompting,
    vq_model,
    mask_id,
    config,
    mask_schedule,
    num_steps,
    prepare_inputs_and_labels_fn # Pass the function reference
):
    """Runs evaluation on the test set for a specified number of steps."""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    
    test_iterator = cycle(test_dataloader) # Cycle through the dataloader

    with torch.no_grad():
        for step in range(num_steps):
            try:
                batch = next(test_iterator)
            except StopIteration:
                # Should not happen with cycle, but good practice
                logger.warning("Test dataloader exhausted unexpectedly during evaluation.")
                break 

            source_images = torch.stack(batch['source_images']).to(accelerator.device)
            edited_images = torch.stack(batch['edited_images']).to(accelerator.device)
            texts = batch['texts']

            # Use the passed function to prepare inputs/labels
            # For evaluation, we might want deterministic masking or no masking.
            # Here we use the same masking strategy as training for simplicity.
            input_ids, labels, _ = prepare_inputs_and_labels_fn(source_images, edited_images, texts)
            
            # Get attention mask (assuming same logic as training)
            mask_dtype = model.module.showo.model.embed_tokens.weight.dtype if hasattr(model, 'module') else model.showo.model.embed_tokens.weight.dtype
            attention_mask = create_attention_mask_predict_next(
                input_ids,
                pad_id=int(uni_prompting.sptids_dict['[PAD]']),
                soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                rm_pad_in_image=True,
                return_inverse_mask=True
            ).to(mask_dtype)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, model.module.output_size if hasattr(model, 'module') else model.output_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )

            # Gather loss across all processes
            avg_loss = accelerator.gather(loss.repeat(config.training.batch_size)).mean()
            total_loss += avg_loss.item()
            total_batches += 1

    model.train() # Switch back to train mode
    if total_batches == 0:
        return 0.0 # Avoid division by zero
    return total_loss / total_batches

def main():
    config = get_config()
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=True,
    )

    total_batch_size = config.training.batch_size * accelerator.num_processes * config.training.gradient_accumulation_steps

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", wandb.util.generate_id())
        config.wandb.run_id = run_id if not resume_wandb_run else run_id
        wandb_init_kwargs = dict(
            name=config.experiment.name,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint")
        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        OmegaConf.save(config, Path(config.experiment.output_dir) / "config.yaml")

    if config.training.seed is not None:
        set_seed(config.training.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side="left")
    uni_prompting = UniversalPrompting(
        tokenizer,
        max_text_len=config.dataset.preprocessing.max_seq_length,
        special_tokens=(
            "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|edit|>",
            "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"
        ),
        ignore_id=-100,
        cond_dropout_prob=config.training.cond_dropout_prob
    )

    vq_model = MAGVITv2().to(accelerator.device)
    if config.model.vq_model.get("pretrained_model_path", None):
        # Assuming MAGVITv2() returns an instance if path is not given,
        # and from_pretrained if name is given. This logic seems reversed.
        # Correcting based on typical Hugging Face patterns:
        try:
             state_dict = torch.load(config.model.vq_model.pretrained_model_path, map_location='cpu')['model']
             vq_model.load_state_dict(state_dict)
             logger.info(f"Loaded VQ model from local path: {config.model.vq_model.pretrained_model_path}")
        except Exception as e:
             logger.error(f"Failed to load VQ model from path {config.model.vq_model.pretrained_model_path}: {e}")
             # Optionally fall back to Hugging Face Hub or raise error
             vq_model = MAGVITv2.from_pretrained(config.model.vq_model.vq_model_name).to(accelerator.device)
             logger.info(f"Loaded VQ model from Hub: {config.model.vq_model.vq_model_name}")
    else:
        vq_model = MAGVITv2.from_pretrained(config.model.vq_model.vq_model_name).to(accelerator.device)
        logger.info(f"Loaded VQ model from Hub: {config.model.vq_model.vq_model_name}")


    vq_model.eval()
    vq_model.requires_grad_(False)

    if config.model.showo.load_from_showo:
        model = Showo.from_pretrained(config.model.showo.pretrained_model_path).to(accelerator.device)
        if config.model.showo.vocab_size != model.vocab_size:
            model.resize_token_embeddings(config.model.showo.vocab_size)
            model.config.codebook_size = config.model.showo.codebook_size
            model.config.vocab_size = config.model.showo.vocab_size
            model.vocab_size = config.model.showo.vocab_size
            model.output_size = config.model.showo.vocab_size
            model.config.mask_token_id = config.model.showo.vocab_size - 1
            model.mask_token_id = config.model.showo.vocab_size - 1
    else:
        model = Showo(**config.model.showo).to(accelerator.device)
    mask_id = model.mask_token_id

    optimizer_config = config.optimizer.params
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
         "weight_decay": optimizer_config.weight_decay},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    
    use_8bit_optimizer = config.optimizer.get("use_8bit_optimizer", False)
    if use_8bit_optimizer:
        try:
            import bitsandbytes as bnb
            logger.info("Using 8-bit AdamW optimizer")
            optimizer = bnb.optim.AdamW8bit(
                optimizer_grouped_parameters,
                lr=optimizer_config.learning_rate,
                betas=(optimizer_config.beta1, optimizer_config.beta2),
                weight_decay=optimizer_config.weight_decay,
                eps=optimizer_config.epsilon,
            )
        except ImportError:
            logger.warning("bitsandbytes not available, falling back to regular AdamW")
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=optimizer_config.learning_rate,
                betas=(optimizer_config.beta1, optimizer_config.beta2),
                weight_decay=optimizer_config.weight_decay,
                eps=optimizer_config.epsilon,
            )
    else:
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )

    mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))
    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
    )

    # Use the custom dataset class - create train and test splits
    train_dataset = ImageEditingDataset(
        metadata_path=config.dataset.params.train_metadata_path,
        dataset_root=config.dataset.params.dataset_root,
        resolution=config.dataset.preprocessing.resolution,
        split='train',
        train_split_ratio=config.dataset.params.train_split_ratio,
        seed=config.training.seed
    )
    test_dataset = ImageEditingDataset(
        metadata_path=config.dataset.params.train_metadata_path, # Use same metadata
        dataset_root=config.dataset.params.dataset_root,
        resolution=config.dataset.preprocessing.resolution,
        split='test', 
        train_split_ratio=config.dataset.params.train_split_ratio, # Need ratio to know where split happens
        seed=config.training.seed # Use same seed for consistent split
    )

    # Set up Train DataLoader with DistributedSampler for multi-GPU training
    train_sampler = None
    if accelerator.num_processes > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=True,
            seed=config.training.seed
        )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None), # Shuffle if not using sampler
        num_workers=config.dataset.params.num_workers,
        pin_memory=config.dataset.params.pin_memory,
        collate_fn=lambda x: {k: [d[k] for d in x] for k in x[0]} if x else {}
    )

    # Set up Test DataLoader (no sampler needed if evaluating on main process, or use standard sampler)
    # Let's create a sampler for distributed evaluation just in case, though evaluation logic might gather results
    test_sampler = None
    if accelerator.num_processes > 1:
         test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=False # No need to shuffle test set for evaluation
        )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size, # Use same batch size for eval efficiency
        sampler=test_sampler,
        shuffle=False,
        num_workers=config.dataset.params.num_workers,
        pin_memory=config.dataset.params.pin_memory,
        collate_fn=lambda x: {k: [d[k] for d in x] for k in x[0]} if x else {}
    )

    # Create visualization dataloader using the TEST dataset
    vis_batch_size = min(config.training.batch_size, config.experiment.visualization_samples, 4) # Ensure reasonable vis batch size
    vis_dataloader = DataLoader(
        test_dataset, # Use test dataset for visualization
        batch_size=vis_batch_size,
        shuffle=True, # Shuffle for diverse visualizations
        num_workers=1, # Fewer workers often fine for visualization
        pin_memory=True,
        collate_fn=lambda x: {k: [d[k] for d in x] for k in x[0]} if x else {}
    )

    # Prepare for distributed training (include test_dataloader)
    if accelerator.state.deepspeed_plugin:
        logger.info("--- DeepSpeed Config Used by Accelerate ---")
        logger.info(accelerator.state.deepspeed_plugin.deepspeed_config)
        logger.info("-----------------------------------------")
    model, optimizer, lr_scheduler, train_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader, test_dataloader
    )
    # vis_dataloader is not prepared as it's iterated manually in the visualization function

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.training.gradient_accumulation_steps)
    # Adjust num_train_epochs calculation if max_train_steps is the primary limit
    if config.training.max_train_steps:
         num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)
    else:
         # Need a way to determine epochs if max_train_steps isn't set
         # This part might need adjustment based on how training duration is defined
         num_train_epochs = config.training.get("num_train_epochs", 1) # Default to 1 epoch if not specified
         config.training.max_train_steps = num_update_steps_per_epoch * num_train_epochs

    global_step = 0
    first_epoch = 0
    if config.experiment.resume_from_checkpoint:
        # Correct logic for finding the latest checkpoint
        checkpoint_dir = Path(config.experiment.output_dir)
        if config.experiment.resume_from_checkpoint == 'latest':
            dirs = sorted([d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")], key=lambda x: int(x.name.split("-")[1]))
            path = dirs[-1] if dirs else None
        else:
             # Resume from a specific checkpoint path if provided
             path = checkpoint_dir / config.experiment.resume_from_checkpoint
             if not path.exists():
                 path = None # Reset path if specific checkpoint not found
        
        if path and path.exists():
            try:
                 logger.info(f"Resuming from checkpoint {path}")
                 accelerator.load_state(path)
                 global_step = int(path.name.split("-")[1])
                 first_epoch = global_step // num_update_steps_per_epoch
                 logger.info(f"Resumed from step {global_step}, epoch {first_epoch}")
            except Exception as e:
                 logger.error(f"Failed to load checkpoint from {path}: {e}. Starting from scratch.")
                 global_step = 0
                 first_epoch = 0
        else:
             logger.info("No valid checkpoint found or specified. Starting from scratch.")

    mask_dtype = model.module.showo.model.embed_tokens.weight.dtype if hasattr(model, 'module') else model.showo.model.embed_tokens.weight.dtype

    logger.info("***** Running training *****")
    logger.info(f"  Num train examples = {len(train_dataset)}")
    logger.info(f"  Num test examples = {len(test_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.training.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.training.max_train_steps}")
    logger.info(f"  Using optimizer: {'8-bit AdamW' if use_8bit_optimizer else 'AdamW'}")
    logger.info(f"  Evaluation interval: {config.experiment.evaluation_interval} steps")
    logger.info(f"  Evaluation steps: {config.experiment.evaluation_steps}")
    logger.info(f"  Visualization interval: {config.experiment.visualization_interval} steps")
    logger.info(f"  Visualization samples: {config.experiment.visualization_samples}")

    # Define prepare_inputs_and_labels within main or pass necessary variables
    # Keep it defined inside main for simplicity now
    @torch.no_grad()
    def prepare_inputs_and_labels(source_images, edited_images, texts):
        offset = len(uni_prompting.text_tokenizer)  # e.g., 50295 + num_special_tokens
        
        # Ensure images are tensors before passing to vq_model
        if isinstance(source_images, list):
             source_images = torch.stack(source_images).to(accelerator.device)
        if isinstance(edited_images, list):
             edited_images = torch.stack(edited_images).to(accelerator.device)

        source_image_tokens = vq_model.get_code(source_images) + offset
        edited_image_tokens = vq_model.get_code(edited_images) + offset
        
        # Use the model available on the current accelerator device
        current_model = accelerator.unwrap_model(model) 
        current_mask_id = current_model.mask_token_id

        masked_edited_image_tokens, labels, loss_weight, mask_prob = mask_or_random_replace_tokens(
            edited_image_tokens, current_mask_id, config, mask_schedule=mask_schedule
        )
        input_ids, _, labels = uni_prompting((texts, source_image_tokens, masked_edited_image_tokens, labels), 'edit')
        return input_ids, labels, mask_prob

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    progress_bar = tqdm(range(global_step, config.training.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        if train_sampler and hasattr(train_sampler, "set_epoch"): # Check if sampler needs epoch set
            train_sampler.set_epoch(epoch)
            
        train_loss = 0.0 # Track training loss per epoch/log period
        
        for step, batch in enumerate(train_dataloader):
            # Skip steps before resuming
            if config.experiment.resume_from_checkpoint and global_step > 0 and step < global_step % num_update_steps_per_epoch:
                 progress_bar.update(1)
                 continue
                 
            source_images = batch['source_images'] # Keep as list initially
            edited_images = batch['edited_images'] # Keep as list initially
            texts = batch['texts']

            data_time_m.update(time.time() - end)

            # input prep happens inside accelerator.accumulate context if needed
            with accelerator.accumulate(model):
                # Prepare inputs and labels - now needs global_step for masking schedule
                input_ids, labels, mask_prob = prepare_inputs_and_labels(source_images, edited_images, texts)
                
                # Ensure tensors are on the correct device *before* model call
                input_ids = input_ids.to(accelerator.device)
                labels = labels.to(accelerator.device)
                
                attention_mask = create_attention_mask_predict_next(
                    input_ids,
                    pad_id=int(uni_prompting.sptids_dict['[PAD]']),
                    soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                    eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                    rm_pad_in_image=True,
                    return_inverse_mask=True
                ).to(mask_dtype)

                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                # Get the correct output size (vocab size) from the potentially wrapped model
                output_size = model.module.output_size if hasattr(model, 'module') else model.output_size
                loss = F.cross_entropy(
                    logits[:, :-1].contiguous().view(-1, output_size),
                    labels[:, 1:].contiguous().view(-1),
                    ignore_index=-100,
                    label_smoothing=config.training.get("label_smoothing", 0.0) # Add label smoothing
                )

                avg_loss = accelerator.gather(loss.repeat(config.training.batch_size)).mean()
                avg_masking_rate = accelerator.gather(mask_prob.repeat(config.training.batch_size)).mean()
                
                train_loss += avg_loss.item() / config.training.gradient_accumulation_steps
                
                accelerator.backward(loss)

                if config.training.max_grad_norm and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Checks should be done outside the accumulate context
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                batch_time_m.update(time.time() - end)
                end = time.time()

                if global_step % config.experiment.log_every == 0:
                    samples_per_second_per_gpu = (config.training.gradient_accumulation_steps * config.training.batch_size) / batch_time_m.val
                    current_train_loss = train_loss / (config.experiment.log_every / config.training.gradient_accumulation_steps)
                    logs = {
                        "step_loss": current_train_loss, # Log avg loss over the interval
                        "lr": lr_scheduler.get_last_lr()[0],
                        "avg_masking_rate": avg_masking_rate.item(),
                        "samples/sec/gpu": samples_per_second_per_gpu,
                        "data_time": data_time_m.val,
                        "batch_time": batch_time_m.val,
                        "epoch": epoch,
                    }
                    accelerator.log(logs, step=global_step)
                    progress_bar.set_postfix(logs)
                    logger.info(
                        f"Step: {global_step} Loss: {current_train_loss:0.4f} "
                        f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                        f"Batch (t): {batch_time_m.val:0.4f} LR: {lr_scheduler.get_last_lr()[0]:0.6f} Epoch: {epoch}"
                    )
                    # Reset trackers
                    batch_time_m.reset()
                    data_time_m.reset()
                    train_loss = 0.0 # Reset train loss accumulator
                
                # --- Evaluation and Visualization --- 
                if global_step % config.experiment.evaluation_interval == 0:
                    logger.info(f"Running evaluation at step {global_step}...")
                    test_loss = evaluate(
                        model=model,
                        test_dataloader=test_dataloader,
                        accelerator=accelerator,
                        uni_prompting=uni_prompting,
                        vq_model=vq_model,
                        mask_id=mask_id,
                        config=config,
                        mask_schedule=mask_schedule,
                        num_steps=config.experiment.evaluation_steps,
                        prepare_inputs_and_labels_fn=prepare_inputs_and_labels
                    )
                    logger.info(f"Evaluation finished. Test Loss: {test_loss:.4f}")
                    accelerator.log({"test_loss": test_loss}, step=global_step)

                # Run visualization using both train and test dataloaders
                if global_step % config.experiment.visualization_interval == 0:
                        logger.info(f"Running visualization at step {global_step}...")
                        log_visualizations(
                        model=model,
                        vq_model=vq_model,
                        train_dataloader=train_dataloader,  # Pass train dataloader
                        test_dataloader=vis_dataloader,    # Use the test-based vis_dataloader
                        accelerator=accelerator,
                        uni_prompting=uni_prompting,
                        config=config,
                        step=global_step,
                        num_samples=config.experiment.visualization_samples // 2,  # Split samples between train/test
                        )
                        logger.info("Visualization finished.")

                # --- Checkpointing --- 
                if global_step % config.experiment.save_every == 0:
                    save_path = Path(config.experiment.output_dir) / f"checkpoint-{global_step}"
                    # Use accelerator's save_state for robust checkpointing
                    accelerator.save_state(save_path)
                    logger.info(f"Saved accelerator state to {save_path}")
                    # Optionally save unwrapped model separately if needed, but save_state should be sufficient
                    # if accelerator.is_main_process:
                    #     unwrapped_model = accelerator.unwrap_model(model)
                    #     unwrapped_model.save_pretrained(save_path / "unwrapped_model", save_function=accelerator.save)
                    #     logger.info(f"Saved unwrapped model state to {save_path / 'unwrapped_model'}")

            if global_step >= config.training.max_train_steps:
                break
        
        if global_step >= config.training.max_train_steps:
            logger.info(f"Reached max_train_steps ({config.training.max_train_steps}). Stopping training.")
            break

    progress_bar.close()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Save final model using accelerator's recommended way
        logger.info("Saving final model...")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            config.experiment.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
            safe_serialization=False # Adjust as needed
        )
        logger.info(f"Final model saved to {config.experiment.output_dir}")
    accelerator.end_training()

if __name__ == "__main__":
    main()