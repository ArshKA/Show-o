# coding=utf-8
# Copyright 2024 NUS Show Lab, HuggingFace.
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

import torch
import torch.nn.functional as F
from transformers import AutoConfig
from .modeling_utils import ConfigMixin, ModelMixin, register_to_config
from .sampling import cosine_schedule, mask_by_random_topk
from .phi import PhiForCausalLM

class Showo(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            w_clip_vit,
            vocab_size,
            llm_vocab_size,
            llm_model_path='',
            codebook_size=8192,
            num_vq_tokens=256,
            load_from_showo=True,
            **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.register_to_config(mask_token_id=vocab_size - 1)
        if load_from_showo:
            config = AutoConfig.from_pretrained(llm_model_path)
            self.showo = PhiForCausalLM(config)
        else:
            self.showo = PhiForCausalLM.from_pretrained(llm_model_path, attn_implementation='sdpa')
        self.showo.resize_token_embeddings(self.vocab_size)
        self.output_size = self.vocab_size

        if self.w_clip_vit:
            self.mm_projector = torch.nn.Sequential(
                torch.nn.Linear(1024, 2048),
                torch.nn.GELU(),
                torch.nn.Linear(2048, 2048)
            )

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = True

    def forward(
            self,
            input_ids,
            input_embeddings=None,
            attention_mask=None,
            labels=None,
            label_smoothing=0.0,
            batch_size_t2i=0,
            batch_size_lm=0,
            batch_size_mmu=0,
            max_seq_length=128,
            labels_mask_text=None,
            labels_mask_image=None,
            **kwargs,
    ):

        if input_embeddings is None:
            logits = self.showo(input_ids=input_ids, attention_mask=attention_mask)['logits']
        else:
            logits = self.showo(inputs_embeds=input_embeddings, attention_mask=attention_mask)['logits']

        if labels is not None:
            # 1. Mask token prediction (discrete diffusion) for image generation
            # Note that, max_seq_length indicates the maximum number of text tokens, maybe a bit confused.
            loss_t2i = F.cross_entropy(
                logits[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1, self.output_size),
                labels[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1), ignore_index=-100,
            )

            # 2. Next token prediction for language modeling
            loss_lm = F.cross_entropy(
                logits[batch_size_t2i:batch_size_t2i + batch_size_lm, :-1].contiguous().view(-1, self.output_size),
                labels[batch_size_t2i:batch_size_t2i + batch_size_lm, 1:].contiguous().view(-1), ignore_index=-100,
            )

            # 3. Next token prediction for captioning/multimodal understanding
            loss_mmu = F.cross_entropy(
                logits[-batch_size_mmu:, :-1].contiguous().view(-1, self.output_size),
                labels[-batch_size_mmu:, 1:].contiguous().view(-1), ignore_index=-100,
            )

            return logits, loss_t2i, loss_lm, loss_mmu

        return logits

    def t2i_generate(
            self,
            input_ids: torch.LongTensor = None,
            uncond_input_ids: torch.LongTensor = None,
            attention_mask=None,
            temperature=1.0,
            timesteps=18,  # ideal number of steps is 18 in maskgit paper
            guidance_scale=0,
            noise_schedule=cosine_schedule,
            generator: torch.Generator = None,
            config=None,
            **kwargs,
    ):
        """
        Generate 1:1 similar to the original MaskGit repo
        https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py#L79
        """
        # begin with all image token ids masked
        mask_token_id = self.config.mask_token_id
        num_vq_tokens = config.model.showo.num_vq_tokens
        num_new_special_tokens = config.model.showo.num_new_special_tokens

        input_ids_minus_lm_vocab_size = input_ids[:, -(num_vq_tokens + 1):-1].clone()
        input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == mask_token_id,
                                                    mask_token_id,
                                                    input_ids_minus_lm_vocab_size - config.model.showo.llm_vocab_size - num_new_special_tokens)

        # for classifier-free guidance
        if uncond_input_ids is not None:
            uncond_prefix = uncond_input_ids[:, :config.dataset.preprocessing.max_seq_length + 1]

        for step in range(timesteps):
            if uncond_input_ids is not None and guidance_scale > 0:
                uncond_input_ids = torch.cat(
                    [uncond_prefix, input_ids[:, config.dataset.preprocessing.max_seq_length + 1:]], dim=1)
                model_input = torch.cat([input_ids, uncond_input_ids])
                cond_logits, uncond_logits = self(model_input, attention_mask=attention_mask).chunk(2)
                # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                # it seems that muse has a different cfg setting
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                logits = logits[:, -(num_vq_tokens + 1):-1, config.model.showo.llm_vocab_size + num_new_special_tokens:-1]
            else:
                logits = self(input_ids, attention_mask=attention_mask)
                logits = logits[:, -(num_vq_tokens + 1):-1, config.model.showo.llm_vocab_size + num_new_special_tokens:-1]

            probs = logits.softmax(dim=-1)
            sampled = probs.reshape(-1, logits.size(-1))
            sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1])

            unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
            sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
            # Defines the mask ratio for the next round. The number to mask out is
            # determined by mask_ratio * unknown_number_in_the_beginning.
            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(torch.tensor(ratio))
            # Computes the probabilities of each selected tokens.
            selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
            selected_probs = selected_probs.squeeze(-1)

            # Ignores the tokens given in the input by overwriting their confidence.
            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
            # Gets mask lens for each sample in the batch according to the mask ratio.
            mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(logits.device)
            # Keeps at least one of prediction in this round and also masks out at least
            # one and for the next iteration
            mask_len = torch.max(
                torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            )
            # Adds noise for randomness
            temperature = temperature * (1.0 - ratio)
            masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
            # Masks tokens with lower confidence.
            input_ids[:, -(num_vq_tokens + 1):-1] = torch.where(masking, mask_token_id,
                                                          sampled_ids + config.model.showo.llm_vocab_size
                                                          + num_new_special_tokens)
            input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)

        return sampled_ids

    @torch.no_grad()
    def mmu_generate(self, idx=None, input_embeddings=None, attention_mask=None, max_new_tokens=100, temperature=1.0, top_k=None, eot_token=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        try:
            device = idx.device
        except:
            device = input_embeddings.device

        result = []
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            # idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            # logits, _ = self(idx_cond)
            logits = self(idx, input_embeddings=input_embeddings, attention_mask=attention_mask)

            L = attention_mask.shape[-1]
            attention_mask = attention_mask.squeeze()
            attention_mask_a = torch.hstack(
                [
                    attention_mask,  # L, L
                    torch.zeros((L, 1)).to(device) + torch.finfo(logits.dtype).min,
                ]
            )
            attention_mask_b = torch.vstack(
                [
                    attention_mask_a,  # L, L+1
                    torch.hstack([attention_mask[-1, :], torch.tensor([0]).to(device)]).unsqueeze(0),
                ]
            )
            attention_mask = attention_mask_b

            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            result.append(idx_next[0][0])
            # append sampled index to the running sequence and continue
            if self.config.w_clip_vit:
                idx_next_embeddings = self.showo.model.embed_tokens(idx_next)
                input_embeddings = torch.cat([input_embeddings, idx_next_embeddings], dim=1)
            else:
                idx = torch.cat((idx, idx_next), dim=1)

            if eot_token is not None and idx_next.cpu() == eot_token:
                break

        return result

    def edit_generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        config,
        guidance_scale: float = 3.0,
        timesteps: int = 100,
        temperature: float = 1.0,
        noise_schedule: callable = cosine_schedule,
        generator: torch.Generator = None,
    ):
        """
        Generate edited image tokens by iteratively refining the input sequence.
        
        Args:
            input_ids (torch.LongTensor): Input sequence with text, source image, and masked edited image tokens.
            attention_mask (torch.Tensor): Attention mask for the sequence.
            config: Configuration object containing model parameters.
            guidance_scale (float): Scale for classifier-free guidance (not implemented here).
            timesteps (int): Number of generation steps.
            temperature (float): Sampling temperature.
            noise_schedule (callable): Function to compute mask ratio (e.g., cosine_schedule).
            generator (torch.Generator): Random number generator for reproducibility.
        
        Returns:
            torch.LongTensor: Full generated sequence with edited image tokens filled in.
        """
        mask_token_id = self.mask_token_id
        num_vq_tokens = config.model.showo.num_vq_tokens
        llm_vocab_size = config.model.showo.llm_vocab_size
        num_new_special_tokens = config.model.showo.num_new_special_tokens
        codebook_size = config.model.showo.codebook_size

        # Define the slice for edited image tokens: between second <|soi|> and second <|eoi|>
        edited_image_slice = slice(-(num_vq_tokens + 1), -1)

        # Clone the edited image tokens and adjust to 0-based indices for sampling
        edited_image_tokens = input_ids[:, edited_image_slice].clone()
        edited_image_tokens = torch.where(
            edited_image_tokens == mask_token_id,
            mask_token_id,
            edited_image_tokens - llm_vocab_size - num_new_special_tokens
        )

        for step in range(timesteps):
            # Update input_ids with current edited image tokens
            input_ids[:, edited_image_slice] = torch.where(
                edited_image_tokens == mask_token_id,
                mask_token_id,
                edited_image_tokens + llm_vocab_size + num_new_special_tokens
            )

            # Get model logits
            logits = self(input_ids, attention_mask=attention_mask)

            # Extract logits for edited image positions, focusing on image token vocabulary
            logits_edited = logits[:, edited_image_slice, llm_vocab_size + num_new_special_tokens : llm_vocab_size + num_new_special_tokens + codebook_size]

            # Sample from logits
            probs = F.softmax(logits_edited / temperature, dim=-1)
            # Sample indices relative to the codebook (0 to codebook_size-1)
            sampled_ids_raw = torch.multinomial(probs.view(-1, probs.size(-1)), 1, generator=generator).view(probs.shape[:2])

            # Preserve known tokens and update masked ones
            unknown_map = edited_image_tokens == mask_token_id
            # Create the final set of sampled_ids, potentially including original tokens or mask_token_id
            sampled_ids_final = torch.where(unknown_map, sampled_ids_raw, edited_image_tokens)

            # Compute mask ratio for this step
            ratio = (step + 1) / timesteps
            # Ensure ratio is a tensor for noise_schedule if needed
            mask_ratio = noise_schedule(torch.tensor(ratio, device=probs.device))

            # Determine number of tokens to mask
            # Calculate mask_len based on num_vq_tokens and mask_ratio
            mask_len_float = num_vq_tokens * mask_ratio
            # Ensure mask_len is calculated per batch item if unknown_map varies
            current_unknown = unknown_map.sum(dim=-1, keepdim=True)
            # Clamp mask_len: max(1, min(floor(mask_len_float), current_unknown - 1))
            # Use torch.clamp for tensor operations
            mask_len_tensor = torch.clamp(mask_len_float.floor(), min=1.0)
            mask_len_tensor = torch.min(mask_len_tensor, torch.clamp(current_unknown - 1, min=0.0)).long()

            # Compute confidence scores for the *raw* sampled tokens (indices 0 to codebook_size-1)
            # Gather using sampled_ids_raw which has valid indices for probs
            selected_probs = torch.gather(probs, -1, sampled_ids_raw.unsqueeze(-1)).squeeze(-1)
            # Ignore known tokens by setting their confidence high
            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(probs.dtype).max)

            # Apply masking based on confidence
            temperature_step = temperature * (1.0 - ratio)
            # Pass the tensor version of mask_len
            masking = mask_by_random_topk(mask_len_tensor, selected_probs, temperature_step, generator=generator)

            # Update edited image tokens using the final sampled ids
            edited_image_tokens = torch.where(masking, mask_token_id, sampled_ids_final)

        # --- Final Processing ---
        # At the end of the loop, edited_image_tokens contains the generated sequence
        # with indices relative to the codebook (0 to codebook_size-1) or mask_token_id.

        # Handle any remaining mask tokens. Replacing with 0 is a common strategy.
        # Ensure the replacement token (e.g., 0) is valid for the VQ decoder.
        final_edited_tokens = torch.where(
            edited_image_tokens == mask_token_id,
            torch.zeros_like(edited_image_tokens), # Replace mask with token 0
            edited_image_tokens
        )

        # Ensure all tokens are within the valid range [0, codebook_size - 1]
        # This prevents out-of-bounds errors if something unexpected happened.
        final_edited_tokens = torch.clamp(final_edited_tokens, 0, codebook_size - 1)

        # Return *only* the processed edited image tokens (shape: batch_size, num_vq_tokens)
        return final_edited_tokens