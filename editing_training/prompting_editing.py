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

import torch

class UniversalPrompting:
    def __init__(self, text_tokenizer, special_tokens=("<|soi|>", "<|eoi|>", "<|edit|>"), max_text_len=8000, ignore_id=-100, cond_dropout_prob=0.1):
        self.text_tokenizer = text_tokenizer
        self.text_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.text_tokenizer.add_tokens(list(special_tokens))
        self.sptids_dict = {token: torch.tensor(self.text_tokenizer.convert_tokens_to_ids([token])) for token in special_tokens}
        self.sptids_dict['<|sot|>'] = torch.tensor([self.text_tokenizer.bos_token_id])
        self.sptids_dict['<|eot|>'] = torch.tensor([self.text_tokenizer.eos_token_id])
        self.sptids_dict['[PAD]'] = torch.tensor([self.text_tokenizer.pad_token_id])
        self.max_text_len = max_text_len + 1
        self.pad_id = self.text_tokenizer.pad_token_id
        self.ignore_id = ignore_id
        self.cond_dropout_prob = cond_dropout_prob

    def edit_prompt(self, text_ids, source_image_ids, masked_edited_image_ids, labels=None):
        device = source_image_ids.device
        sequence_ids = []
        label_ids = [] if labels is not None else None
        for i in range(len(text_ids)):
            # Same sequence construction as before
            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]

            temp_ids = [int(self.sptids_dict['<|edit|>'])] + text_ids[i] + [self.text_tokenizer.eos_token_id]
            if self.max_text_len >= len(temp_ids):
                temp_ids = [self.pad_id] * (self.max_text_len - len(temp_ids)) + temp_ids
            else:
                temp_ids = temp_ids[:self.max_text_len - 1] + [self.text_tokenizer.eos_token_id]

            temp_ids = torch.cat([
                torch.tensor(temp_ids).to(device),
                self.sptids_dict['<|soi|>'].to(device),
                source_image_ids[i],
                self.sptids_dict['<|eoi|>'].to(device),
                self.sptids_dict['<|soi|>'].to(device),
                masked_edited_image_ids[i],
                self.sptids_dict['<|eoi|>'].to(device)
            ], dim=0)

            sequence_ids.append(temp_ids.unsqueeze(0))

            if labels is not None:
                text_source_len = len(temp_ids) - len(masked_edited_image_ids[i]) - 2
                temp_label_ids = torch.cat([
                    torch.tensor([self.ignore_id] * (text_source_len + 1)).to(device),
                    labels[i],
                    torch.tensor([self.ignore_id]).to(device)
                ], dim=0)
                label_ids.append(temp_label_ids.unsqueeze(0))

        attention_masks = torch.ones_like(torch.cat(sequence_ids, dim=0))
        if labels is not None:
            return torch.cat(sequence_ids, dim=0), attention_masks, torch.cat(label_ids, dim=0)
        else:
            return torch.cat(sequence_ids, dim=0), attention_masks, None

    def __call__(self, input, task, padding=True, config=None):
        if task == "edit":
            text_ids = self.text_tokenizer(input[0])['input_ids']
            source_image_ids = input[1]
            masked_edited_image_ids = input[2]
            labels = input[3]
            return self.edit_prompt(text_ids, source_image_ids, masked_edited_image_ids, labels)
        elif task == "edit_gen":
            text_ids = self.text_tokenizer(input[0])['input_ids']
            source_image_ids = input[1]
            masked_edited_image_ids = input[2]
            return self.edit_prompt(text_ids, source_image_ids, masked_edited_image_ids, labels=None)
        else:
            raise NotImplementedError

def create_attention_mask_predict_next(sequence, pad_id, soi_id, eoi_id, rm_pad_in_image=False, return_inverse_mask=True):
    N, L = sequence.shape
    is_padding = sequence == pad_id
    is_start_image = sequence == soi_id
    is_end_image = sequence == eoi_id
    cumulative_start = torch.cumsum(is_start_image, dim=1)
    cumulative_end = torch.cumsum(is_end_image, dim=1)
    in_image_segment = (cumulative_start > cumulative_end) | is_start_image | is_end_image
    is_text = ~in_image_segment
    causal_mask = torch.tril(torch.ones((L, L), dtype=torch.bool)).to(sequence.device)
    mask_text = is_text[:, :, None] * causal_mask[None, :, :]
    is_text_image = is_text | in_image_segment
    mask_text_image_bi = is_text_image[:, :, None] * is_text_image[:, None, :]
    if rm_pad_in_image:
        sid_img = torch.where(sequence == soi_id)[1]
        for i in range(mask_text_image_bi.shape[0]):
            pad_end_idx = torch.where(sequence[i] == pad_id)[0][-1] if (sequence[i] == pad_id).any() else 0
            mask_text[i][pad_end_idx + 1:, :pad_end_idx + 1] = 0
            id_padding = torch.where(is_padding[i])[0]
            mask_text_image_bi[i][sid_img[i]:, id_padding] = 0
    mask_text[in_image_segment] = mask_text_image_bi[in_image_segment]
    if return_inverse_mask:
        inverted_mask = 1.0 - mask_text.type(sequence.dtype)
        inverted_mask = inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.iinfo(sequence.dtype).min)
        return inverted_mask.unsqueeze(1)
    return mask_text.unsqueeze(1)

if __name__ == '__main__':
    pass