import os
import sys
import torch
import gc
from torch.cuda.amp import autocast
from models import Showo
from training.utils import get_config
import time

def log_memory(stage):
    """Log current GPU memory usage"""
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"{stage} - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    return allocated, reserved

def reset_memory():
    """Clear memory and print how much was freed"""
    before = torch.cuda.memory_allocated() / 1e9
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    after = torch.cuda.memory_allocated() / 1e9
    print(f"Memory cleanup: {before-after:.2f}GB freed")

def setup_config(config_path):
    """Set up the configuration for testing"""
    sys.argv = [sys.argv[0], f"config={config_path}"]
    return get_config()

def test_inference_style():
    """Test memory usage with inference-style optimizations (no gradients)"""
    print("\n=== Test 1: Inference Style (No Gradients) ===")
    config = setup_config("configs/showo_demo.yaml")
    
    # Load model in eval mode with no grad
    with torch.no_grad():
        print("Loading model in inference mode...")
        model = Showo.from_pretrained(config.model.showo.pretrained_model_path)
        model.eval().cuda()
        
        log_memory("After model load")
        
        # Create dummy input - using a smaller sequence length
        input_ids = torch.randint(0, config.model.showo.vocab_size, (1, 128), device='cuda')
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        
        # Run with mixed precision
        with autocast(dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            log_memory("After forward pass")
            
        del model, outputs, input_ids, attention_mask
    
    reset_memory()

def test_basic_training():
    """Test memory usage with basic training setup (tracking gradients)"""
    print("\n=== Test 2: Basic Training (With Gradients) ===")
    config = setup_config("configs/showo_demo.yaml")
    
    print("Loading model in basic training mode...")
    model = Showo.from_pretrained(config.model.showo.pretrained_model_path)
    model.train().cuda()
    
    log_memory("After model load")
    
    # Create dummy input - using a smaller sequence length
    input_ids = torch.randint(0, config.model.showo.vocab_size, (1, 128), device='cuda')
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    labels = torch.randint(0, config.model.showo.vocab_size, (1, 128), device='cuda')
    
    # Forward and backward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    loss = torch.nn.functional.cross_entropy(
        outputs[:, :-1].reshape(-1, model.output_size),
        labels[:, 1:].reshape(-1)
    )
    log_memory("After forward pass")
    
    loss.backward()
    log_memory("After backward pass")
    
    del model, outputs, input_ids, attention_mask, labels, loss
    
    reset_memory()

def test_mixed_precision_training():
    """Test memory usage with mixed precision training"""
    print("\n=== Test 3: Mixed Precision Training ===")
    config = setup_config("configs/showo_demo.yaml")
    
    print("Loading model with mixed precision training...")
    model = Showo.from_pretrained(config.model.showo.pretrained_model_path)
    model.train().cuda()
    
    log_memory("After model load")
    
    # Create dummy input - using a smaller sequence length
    input_ids = torch.randint(0, config.model.showo.vocab_size, (1, 128), device='cuda')
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    labels = torch.randint(0, config.model.showo.vocab_size, (1, 128), device='cuda')
    
    # Forward and backward pass with mixed precision
    with autocast(dtype=torch.bfloat16):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = torch.nn.functional.cross_entropy(
            outputs[:, :-1].reshape(-1, model.output_size),
            labels[:, 1:].reshape(-1)
        )
    log_memory("After forward pass (mixed precision)")
    
    loss.backward()
    log_memory("After backward pass (mixed precision)")
    
    del model, outputs, input_ids, attention_mask, labels, loss
    
    reset_memory()

def test_with_optimizer():
    """Test memory usage with optimizer states"""
    print("\n=== Test 4: Training with Optimizer States ===")
    config = setup_config("configs/showo_demo.yaml")
    
    print("Loading model with optimizer...")
    model = Showo.from_pretrained(config.model.showo.pretrained_model_path)
    model.train().cuda()
    
    log_memory("After model load")
    
    # Fix the parameter selection
    optimizer_params = []
    for name, param in model.named_parameters():
        if 'showo.model.embed_tokens' in name or 'showo.model.layers.0' in name:
            param.requires_grad = True
            optimizer_params.append(param)
            print(f"Adding parameter: {name}")
        else:
            param.requires_grad = False
    
    print(f"Number of parameters to optimize: {len(optimizer_params)}")
    optimizer = torch.optim.AdamW(optimizer_params, lr=1e-5)
    log_memory("After optimizer creation")
    
    # Create dummy input - using a smaller sequence length
    input_ids = torch.randint(0, config.model.showo.vocab_size, (1, 128), device='cuda')
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    labels = torch.randint(0, config.model.showo.vocab_size, (1, 128), device='cuda')
    
    # Forward and backward pass
    with autocast(dtype=torch.bfloat16):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = torch.nn.functional.cross_entropy(
            outputs[:, :-1].reshape(-1, model.output_size),
            labels[:, 1:].reshape(-1)
        )
    log_memory("After forward pass")
    
    loss.backward()
    log_memory("After backward pass")
    
    optimizer.step()
    log_memory("After optimizer step")
    
    optimizer.zero_grad()
    log_memory("After zero_grad")
    
    del model, outputs, input_ids, attention_mask, labels, loss, optimizer
    
    reset_memory()

def test_gradient_accumulation():
    """Test memory usage with gradient accumulation"""
    print("\n=== Test 5: Gradient Accumulation ===")
    config = setup_config("configs/showo_demo.yaml")
    
    print("Loading model with gradient accumulation...")
    model = Showo.from_pretrained(config.model.showo.pretrained_model_path)
    model.train().cuda()
    
    log_memory("After model load")
    
    # Fix the parameter selection
    optimizer_params = []
    for name, param in model.named_parameters():
        if 'showo.model.embed_tokens' in name or 'showo.model.layers.0' in name:
            param.requires_grad = True
            optimizer_params.append(param)
            print(f"Adding parameter: {name}")
        else:
            param.requires_grad = False
    
    print(f"Number of parameters to optimize: {len(optimizer_params)}")
    optimizer = torch.optim.AdamW(optimizer_params, lr=1e-5)
    log_memory("After optimizer creation")
    
    # Gradient accumulation steps
    accumulation_steps = 4
    
    for step in range(accumulation_steps):
        print(f"\nAccumulation step {step+1}/{accumulation_steps}")
        
        # Create dummy input - using a smaller sequence length
        input_ids = torch.randint(0, config.model.showo.vocab_size, (1, 128), device='cuda')
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        labels = torch.randint(0, config.model.showo.vocab_size, (1, 128), device='cuda')
        
        # Forward and backward pass with scaled loss
        with autocast(dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = torch.nn.functional.cross_entropy(
                outputs[:, :-1].reshape(-1, model.output_size),
                labels[:, 1:].reshape(-1)
            ) / accumulation_steps  # Scale the loss
        
        log_memory(f"Step {step+1} after forward pass")
        
        loss.backward()
        log_memory(f"Step {step+1} after backward pass")
        
        del outputs, input_ids, attention_mask, labels, loss
        
        # Only step and zero_grad after accumulation is complete
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            log_memory("After optimizer step")
            
            optimizer.zero_grad()
            log_memory("After zero_grad")
    
    del model, optimizer
    reset_memory()

def test_editing_config():
    """Test memory usage with the editing config"""
    print("\n=== Test 6: Training with Editing Config ===")
    config = setup_config("configs/showo_editing.yaml")
    
    print("Loading model with editing config...")
    model = Showo.from_pretrained(config.model.showo.pretrained_model_path)
    
    # Handle custom vocab size like in edit_training.py
    if hasattr(config.model.showo, 'vocab_size') and config.model.showo.vocab_size != model.vocab_size:
        print(f"Resizing token embeddings from {model.vocab_size} to {config.model.showo.vocab_size}")
        model.resize_token_embeddings(config.model.showo.vocab_size)
        model.config.codebook_size = config.model.showo.codebook_size
        model.config.vocab_size = config.model.showo.vocab_size
        model.vocab_size = config.model.showo.vocab_size
        model.output_size = config.model.showo.vocab_size
        model.config.mask_token_id = config.model.showo.vocab_size - 1
        model.mask_token_id = config.model.showo.vocab_size - 1
    
    model.train().cuda()
    log_memory("After model load with vocab resize")
    
    # Fix the parameter selection
    optimizer_params = []
    for name, param in model.named_parameters():
        if 'showo.model.embed_tokens' in name or 'showo.model.layers.0' in name:
            param.requires_grad = True
            optimizer_params.append(param)
            print(f"Adding parameter: {name}")
        else:
            param.requires_grad = False
    
    print(f"Number of parameters to optimize: {len(optimizer_params)}")
    
    # Create optimizer with weight decay split
    optimizer_config = config.optimizer.params
    
    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=optimizer_config.learning_rate,
        betas=(optimizer_config.beta1, optimizer_config.beta2),
        weight_decay=optimizer_config.weight_decay,
        eps=optimizer_config.epsilon,
    )
    log_memory("After optimizer creation")
    
    # Create dummy input - using a smaller sequence length
    batch_size = 1
    input_ids = torch.randint(0, config.model.showo.vocab_size, 
                            (batch_size, 128), 
                            device='cuda')
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    labels = torch.randint(0, config.model.showo.vocab_size, 
                          (batch_size, 128), 
                          device='cuda')
    
    # Forward and backward pass with mixed precision
    with autocast(dtype=torch.bfloat16):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = torch.nn.functional.cross_entropy(
            outputs[:, :-1].reshape(-1, model.output_size),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
    
    log_memory("After forward pass")
    loss.backward()
    log_memory("After backward pass")
    
    optimizer.step()
    log_memory("After optimizer step")
    
    del model, outputs, input_ids, attention_mask, labels, loss, optimizer
    
    reset_memory()

def test_long_sequence():
    """Test memory usage with long sequence (5000 tokens)"""
    print("\n=== Test 7: Long Sequence (5000 tokens) ===")
    config = setup_config("configs/showo_demo.yaml")
    
    print("Loading model for long sequence processing...")
    model = Showo.from_pretrained(config.model.showo.pretrained_model_path)
    model.train().cuda()
    
    log_memory("After model load")
    
    # Fix the parameter selection - only train embeddings and first layer
    optimizer_params = []
    for name, param in model.named_parameters():
        if 'showo.model.embed_tokens' in name or 'showo.model.layers.0' in name:
            param.requires_grad = True
            optimizer_params.append(param)
        else:
            param.requires_grad = False
    
    print(f"Number of parameters to optimize: {len(optimizer_params)}")
    
    # Try to use 8-bit optimizer if available
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(optimizer_params, lr=1e-5)
        print("Using 8-bit AdamW optimizer")
    except ImportError:
        print("bitsandbytes not available, using regular AdamW")
        optimizer = torch.optim.AdamW(optimizer_params, lr=1e-5)
    
    log_memory("After optimizer creation")
    
    # Process in chunks (gradient accumulation)
    chunk_size = 1024
    full_seq_length = 5000
    accumulation_steps = (full_seq_length + chunk_size - 1) // chunk_size
    
    print(f"Processing {full_seq_length} tokens in {accumulation_steps} chunks of {chunk_size}")
    
    # Zero the gradients before accumulation
    optimizer.zero_grad()
    
    # Create long input (5000 tokens)
    long_input_ids = torch.randint(0, config.model.showo.vocab_size, (1, full_seq_length), device='cuda')
    long_attention_mask = torch.ones_like(long_input_ids, dtype=torch.bool)
    long_labels = torch.randint(0, config.model.showo.vocab_size, (1, full_seq_length), device='cuda')
    
    # Process in chunks with gradient accumulation
    for i in range(accumulation_steps):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, full_seq_length)
        
        print(f"\nProcessing chunk {i+1}/{accumulation_steps} (tokens {start_idx}-{end_idx})")
        
        # Extract current chunk
        input_ids = long_input_ids[:, start_idx:end_idx]
        attention_mask = long_attention_mask[:, start_idx:end_idx] 
        labels = long_labels[:, start_idx:end_idx]
        
        # Forward and backward pass with scaled loss
        with autocast(dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Only compute loss on this chunk
            loss = torch.nn.functional.cross_entropy(
                outputs[:, :-1].reshape(-1, model.output_size),
                labels[:, 1:].reshape(-1)
            ) / accumulation_steps  # Scale the loss
        
        log_memory(f"Chunk {i+1} after forward pass")
        
        loss.backward()
        log_memory(f"Chunk {i+1} after backward pass")
        
        # Free memory after processing each chunk
        del outputs, input_ids, attention_mask, labels, loss
        torch.cuda.empty_cache()
    
    # Step after all chunks are processed
    optimizer.step()
    log_memory("After optimizer step")
    
    optimizer.zero_grad()
    log_memory("After zero_grad")
    
    # Clean up the long tensors
    del long_input_ids, long_attention_mask, long_labels
    del model, optimizer
    
    reset_memory()

def test_long_sequence_full_params():
    """Test memory usage with long sequence (5000 tokens) and no parameter freezing"""
    print("\n=== Test 8: Long Sequence with All Parameters ===")
    config = setup_config("configs/showo_demo.yaml")
    
    print("Loading model for long sequence processing (all parameters)...")
    model = Showo.from_pretrained(config.model.showo.pretrained_model_path)
    model.train().cuda()
    
    log_memory("After model load")
    
    # No freezing - keep all parameters trainable
    optimizer_params = list(model.parameters())
    trainable_params = sum(p.numel() for p in optimizer_params if p.requires_grad)
    print(f"Number of parameters to optimize: {len(optimizer_params)} ({trainable_params:,} total)")
    
    # Try to use 8-bit optimizer if available
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(optimizer_params, lr=1e-5)
        print("Using 8-bit AdamW optimizer")
    except ImportError:
        print("bitsandbytes not available, using regular AdamW")
        optimizer = torch.optim.AdamW(optimizer_params, lr=1e-5)
    
    log_memory("After optimizer creation")
    
    # Process in chunks (gradient accumulation)
    chunk_size = 1024
    full_seq_length = 5000
    accumulation_steps = (full_seq_length + chunk_size - 1) // chunk_size
    
    print(f"Processing {full_seq_length} tokens in {accumulation_steps} chunks of {chunk_size}")
    
    # Zero the gradients before accumulation
    optimizer.zero_grad()
    
    # Create long input (5000 tokens)
    long_input_ids = torch.randint(0, config.model.showo.vocab_size, (1, full_seq_length), device='cuda')
    long_attention_mask = torch.ones_like(long_input_ids, dtype=torch.bool)
    long_labels = torch.randint(0, config.model.showo.vocab_size, (1, full_seq_length), device='cuda')
    
    try:
        # Process in chunks with gradient accumulation
        for i in range(accumulation_steps):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, full_seq_length)
            
            print(f"\nProcessing chunk {i+1}/{accumulation_steps} (tokens {start_idx}-{end_idx})")
            
            # Extract current chunk
            input_ids = long_input_ids[:, start_idx:end_idx]
            attention_mask = long_attention_mask[:, start_idx:end_idx] 
            labels = long_labels[:, start_idx:end_idx]
            
            # Forward and backward pass with scaled loss
            with autocast(dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                # Only compute loss on this chunk
                loss = torch.nn.functional.cross_entropy(
                    outputs[:, :-1].reshape(-1, model.output_size),
                    labels[:, 1:].reshape(-1)
                ) / accumulation_steps  # Scale the loss
            
            log_memory(f"Chunk {i+1} after forward pass")
            
            loss.backward()
            log_memory(f"Chunk {i+1} after backward pass")
            
            # Free memory after processing each chunk
            del outputs, input_ids, attention_mask, labels, loss
            torch.cuda.empty_cache()
        
        # Step after all chunks are processed
        optimizer.step()
        log_memory("After optimizer step")
        
        optimizer.zero_grad()
        log_memory("After zero_grad")
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"OOM Error: {e}")
            print("\nOut of memory with full parameters - confirming memory limitations")
        else:
            raise e
    
    # Clean up the long tensors
    del long_input_ids, long_attention_mask, long_labels
    del model, optimizer
    
    reset_memory()

def print_memory_summary():
    """Print a summary of all memory tests"""
    print("\n=== Memory Test Summary ===")
    print("Test results show memory consumption at each stage.")
    print("The difference between inference and training modes is primarily due to:")
    print("1. Gradient storage (training only)")
    print("2. Optimizer states (AdamW creates 2 additional buffers per parameter)")
    print("3. Forward activation storage for backward pass")
    print("4. Mixed precision can help reduce memory usage")
    print("5. Token embedding resizing may increase memory requirements")
    
    print("\nTo reduce memory usage in training:")
    print("1. Use mixed precision (bf16/fp16)")  
    print("2. Use smaller batch sizes")
    print("3. Enable gradient accumulation")
    print("4. Use parameter-efficient fine-tuning methods")
    print("5. Avoid unnecessarily large vocab expansions")
    print("6. Freeze parts of the model (only train a subset of parameters)")

if __name__ == "__main__":
    orig_argv = sys.argv.copy()
    
    try:
        test_inference_style()
        sys.argv = orig_argv.copy()
        
        test_basic_training()
        sys.argv = orig_argv.copy()
        
        test_mixed_precision_training()
        sys.argv = orig_argv.copy()
        
        test_with_optimizer()
        sys.argv = orig_argv.copy()
        
        test_gradient_accumulation()
        sys.argv = orig_argv.copy()
        
        test_editing_config()
        sys.argv = orig_argv.copy()
        
        test_long_sequence()
        sys.argv = orig_argv.copy()
        
        test_long_sequence_full_params()
        
        print_memory_summary()
        
    except Exception as e:
        import traceback
        print(f"Error during testing: {e}")
        traceback.print_exc()
        print("\nContinuing with remaining tests...") 