#!/usr/bin/env python
"""
Minimal testing script for visualizations.
Before running, execute:
  conda activate showo
  source env.sh

This script sets up minimal dummy objects required to run the visualization logging.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
import wandb

# Import the log_visualizations function from the visualization module
from editing_training.visualization import log_visualizations

# Insert DummyShowo class after the imports
class DummyShowo:
    def __init__(self, vocab_size=256):
         self.model = nn.Module()
         self.model.embed_tokens = nn.Embedding(vocab_size, 32)

# Define a dummy model (torch.nn.Module) that returns dummy logits
class DummyModel(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.training = True
        self.showo = DummyShowo(vocab_size=256)
    def forward(self, input_ids, attention_mask):
        batch_size, seq_len = input_ids.shape
        # Return dummy logits with zeros
        return torch.zeros((batch_size, seq_len, self.output_size))
    def eval(self):
        self.training = False
    def train(self):
        self.training = True

model = DummyModel(output_size=256)

# Define a dummy VQ model with get_code and decode_code methods
class DummyVQModel:
    def __init__(self):
        self.codebook_size = 8192  # Match the default in the config
    def get_code(self, images):
        # Return a dummy code with shape (B, 10)
        B = images.shape[0]
        return torch.zeros((B, 10), dtype=torch.long)
    def decode_code(self, tokens):
        # Return a dummy generated image: tensor with shape (1, 3, 64, 64)
        return torch.rand((1, 3, 64, 64))

vq_model = DummyVQModel()

# Define a dummy UniversalPrompting
class DummyUniversalPrompting:
    def __init__(self):
        self.sptids_dict = {"[PAD]": 0, "<|soi|>": 1, "<|eoi|>": 2}
        self.text_tokenizer = list(range(100))
    def __call__(self, inputs, mode):
        texts, source_image_tokens, masked_tokens, edited_image_tokens = inputs
        batch_size = len(texts)
        # Create dummy input_ids with two occurrences of <|soi|> and <|eoi|>
        # Sequence: [0, 1, 50, 1, 55, 2, 60, 2, 70, 80, 81, 82, 83, 0, 0]
        dummy_ids = [0, 1, 50, 1, 55, 2, 60, 2, 70, 80, 81, 82, 83, 0, 0]
        input_ids = torch.tensor([dummy_ids for _ in range(batch_size)], dtype=torch.long)
        return input_ids, None, None

uni_prompting = DummyUniversalPrompting()

mask_id = 255

# Create a minimal dummy config with necessary attributes
class DummyExperiment:
    visualization_interval = 1
    visualization_samples = 1
    output_dir = "./dummy_output"
    name = "dummy_test"
    project = "dummy_project"

class DummyWandb:
    resume = False
    def get(self, key, default=None):
        return default

class DummyModelConfig:
    pretrained_model_path = "dummy_path"
    vocab_size = 256
    codebook_size = 256
    load_from_showo = False

class DummyPreprocessing:
    max_seq_length = 15

class DummyDatasetConfig:
    preprocessing = DummyPreprocessing()

class DummyTraining:
    gradient_accumulation_steps = 1
    batch_size = 1
    cond_dropout_prob = 0.0
    seed = None
    max_train_steps = 1
    enable_tf32 = False
    def get(self, key, default):
        return default

class DummyOptimizerParams:
    weight_decay = 0.0
    learning_rate = 0.001
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

class DummyOptimizer:
    params = DummyOptimizerParams()

class DummyLRSchedulerParams:
    warmup_steps = 0

class DummyLRScheduler:
    scheduler = "dummy"
    params = DummyLRSchedulerParams()

class DummyConfig:
    experiment = DummyExperiment()
    wandb = DummyWandb()
    model = type("ModelConfig", (), {"showo": DummyModelConfig()})
    dataset = DummyDatasetConfig()
    training = DummyTraining()
    optimizer = DummyOptimizer()
    lr_scheduler = DummyLRScheduler()

config = DummyConfig()

# Dummy mask schedule function
mask_schedule = lambda x: 1.0

# Create a dummy dataloader
default_collate = lambda batch: {key: [d[key] for d in batch] for key in batch[0]}

class DummyDataset(Dataset):
    def __len__(self):
        return 1
    def __getitem__(self, idx):
        dummy_image = torch.rand((3, 64, 64))
        return {"source_images": dummy_image,
                "edited_images": dummy_image,
                "texts": "dummy prompt"}

dummy_dataset = DummyDataset()
dummy_dataloader = DataLoader(dummy_dataset, batch_size=1, collate_fn=default_collate)

# Set step to a value that triggers visualization (since visualization_interval=1)
step = 1

if __name__ == '__main__':
    # Initialize wandb
    wandb.init(project="test-visualization", entity=None)
    
    # Initialize accelerator with wandb
    accelerator = Accelerator(log_with="wandb", split_batches=False)

    log_visualizations(
         accelerator=accelerator,
         model=model,
         vq_model=vq_model,
         dataloader=dummy_dataloader,
         uni_prompting=uni_prompting,
         mask_id=mask_id,
         config=config,
         mask_schedule=mask_schedule,
         step=step
     ) 