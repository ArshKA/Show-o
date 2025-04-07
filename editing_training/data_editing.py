import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import random

class ImageEditingDataset(Dataset):
    def __init__(self, metadata_path, dataset_root, resolution=256, min_size=64, split='train', train_split_ratio=0.9, seed=42):
        self.dataset_root = dataset_root
        self.min_size = min_size
        self.split = split
        self.train_split_ratio = train_split_ratio
        self.seed = seed
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        all_samples = []
        
        # Build list of all valid (source, edited) pairs first
        for post_id, post_data in tqdm(list(self.metadata.items())[:100], desc="Building full dataset", total=len(self.metadata)):
            # Construct source image path
            source_path = os.path.join(self.dataset_root, post_id, post_data['source_name'])
            if not os.path.exists(source_path):
                continue

            text = post_data.get('title', '') + '. ' + post_data.get('body', '')
            if len(text) > 200:
                continue
                
            # Validate source image
            if not self._is_valid_image(source_path):
                continue
            
            # Process all comments for edited images
            for comment_id, comment_data in post_data['comments'].items():
                for idx, img_entry in enumerate(comment_data['shopped_images']):
                    if isinstance(img_entry, str):
                        # Extract filename and build edited path
                        filename = os.path.basename(img_entry)
                        edited_path = os.path.join(
                            self.dataset_root,
                            post_id,
                            'photoshopped',
                            filename
                        )
                        if os.path.exists(edited_path):
                            
                            if post_data.get('title', '') in ['[removed]', '[deleted]'] or comment_data.get('body', '') in ['[removed]', '[deleted]']:
                                continue
                            

                            if not self._is_valid_image(edited_path):
                                continue
                                
                            all_samples.append({
                                'source': source_path,
                                'edited': edited_path,
                                'text': text
                            })
        
        # Shuffle and split the data
        random.seed(self.seed)
        random.shuffle(all_samples)
        
        split_idx = int(len(all_samples) * self.train_split_ratio)
        
        if self.split == 'train':
            self.samples = all_samples[:split_idx]
            print(f"Train dataset built with {len(self.samples)} valid image pairs.")
        elif self.split == 'test':
            self.samples = all_samples[split_idx:]
            print(f"Test dataset built with {len(self.samples)} valid image pairs.")
        else:
            raise ValueError(f"Invalid split name: {self.split}. Choose 'train' or 'test'.")

        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def _is_valid_image(self, img_path):
        """Validate if an image is usable by checking if it can be opened and meets minimum size requirements."""
        try:
            img = Image.open(img_path)
            # Verify it's a valid image
            img.verify()
            # Reopen after verify (which closes the image)
            img = Image.open(img_path)
            # Check dimensions
            if img.width < self.min_size or img.height < self.min_size:
                return False
            # Check if the image has valid content
            img.load()
            return True
        except Exception as e:
            # Optionally log the error for debugging
            # print(f"Warning: Could not validate image {img_path}: {e}")
            return False

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            source_img = Image.open(sample['source']).convert('RGB')
            edited_img = Image.open(sample['edited']).convert('RGB')
        except Exception as e:
            # More informative error message
            raise RuntimeError(f"Error loading images for sample {idx} (Source: {sample['source']}, Edited: {sample['edited']}): {e}")
        
        return {
            'source_images': self.transform(source_img),
            'edited_images': self.transform(edited_img),
            'texts': sample['text']
        }