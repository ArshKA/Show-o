import torch
from transformers import AutoTokenizer

def decode_input_ids(input_ids, tokenizer=None, model_path=None, replace_images=True):
    """
    Decode input IDs to text, replacing image token sequences with <image> placeholder.
    
    Args:
        input_ids: Tensor or list of input IDs
        tokenizer: Optional pre-loaded tokenizer
        model_path: Path to tokenizer if not provided directly
        replace_images: Whether to replace image tokens with <image> placeholder
        
    Returns:
        Decoded text string with image tokens replaced
    """
    print(tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)
    # Load tokenizer if not provided
    if tokenizer is None:
        if model_path is None:
            raise ValueError("Either tokenizer or model_path must be provided")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Convert to tensor if not already
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids)
    
    # Special tokens dictionary - these are the IDs to look for
    # These values may need to be adjusted based on your specific tokenizer
    special_tokens = {
        'soi': tokenizer.convert_tokens_to_ids("<|soi|>") if "<|soi|>" in tokenizer.get_vocab() else -1,
        'eoi': tokenizer.convert_tokens_to_ids("<|eoi|>") if "<|eoi|>" in tokenizer.get_vocab() else -1,
        'sov': tokenizer.convert_tokens_to_ids("<|sov|>") if "<|sov|>" in tokenizer.get_vocab() else -1,
        'eov': tokenizer.convert_tokens_to_ids("<|eov|>") if "<|eov|>" in tokenizer.get_vocab() else -1,
    }
    
    # If we're not replacing images or none of the special tokens were found, just decode normally
    if not replace_images or all(v == -1 for v in special_tokens.values()):
        return tokenizer.decode(input_ids, skip_special_tokens=False)
    
    # Convert to list for easier manipulation
    ids_list = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids
    
    # Identify image token sequences and build a new list of tokens
    new_ids = []
    i = 0
    while i < len(ids_list):
        if ids_list[i] == special_tokens['soi']:
            # Found start of an image - find the end
            image_end = i
            for j in range(i+1, len(ids_list)):
                if ids_list[j] == special_tokens['eoi']:
                    image_end = j
                    break
            
            # Calculate image token length (excluding start and end tokens)
            image_length = image_end - i - 1
            
            # Add start token
            new_ids.append(special_tokens['soi'])
            
            # Add placeholder token for the middle part
            new_ids.append(-image_length)  # Negative to ensure it doesn't conflict with real token IDs
            
            # Add end token
            new_ids.append(special_tokens['eoi'])
            
            i = image_end + 1
        
        elif ids_list[i] == special_tokens['sov']:
            # Found start of video/second image - find the end
            video_end = i
            for j in range(i+1, len(ids_list)):
                if ids_list[j] == special_tokens['eov']:
                    video_end = j
                    break
            
            # Calculate video token length (excluding start and end tokens)
            video_length = video_end - i - 1
            
            # Add start token
            new_ids.append(special_tokens['sov'])
            
            # Add placeholder token for the middle part
            new_ids.append(-video_length)  # Negative to ensure it doesn't conflict with real token IDs
            
            # Add end token
            new_ids.append(special_tokens['eov'])
            
            i = video_end + 1
        
        else:
            # Regular token
            new_ids.append(ids_list[i])
            i += 1
    
    # Decode the modified sequence
    # Need to handle the negative placeholder IDs separately
    decoded_parts = []
    i = 0
    while i < len(new_ids):
        if new_ids[i] < 0:
            # This is our placeholder for image token count
            decoded_parts.append(f"<{-new_ids[i]} tokens>")
        else:
            # Regular token - decode individually to maintain spacing
            decoded_parts.append(tokenizer.decode([new_ids[i]], skip_special_tokens=False))
        i += 1
    
    return ''.join(decoded_parts)

def print_token_breakdown(input_ids, tokenizer=None, model_path=None):
    """
    Print a breakdown of tokens with their IDs and text for debugging purposes.
    
    Args:
        input_ids: Tensor or list of input IDs
        tokenizer: Optional pre-loaded tokenizer
        model_path: Path to tokenizer if not provided directly
    """
    # Load tokenizer if not provided
    if tokenizer is None:
        if model_path is None:
            raise ValueError("Either tokenizer or model_path must be provided")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Convert to list
    ids_list = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids
    
    # Print token breakdown
    print("Token ID | Token")
    print("-" * 30)
    
    for token_id in ids_list:
        # Handle special cases separately
        if token_id in tokenizer.all_special_ids:
            # Get the special token text
            for token, id in tokenizer.get_vocab().items():
                if id == token_id:
                    token_text = token
                    break
            else:
                token_text = f"<special:{token_id}>"
        else:
            # Regular token
            token_text = tokenizer.decode([token_id])
        
        print(f"{token_id:8} | {token_text}")
