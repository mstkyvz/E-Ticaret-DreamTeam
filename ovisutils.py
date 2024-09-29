import logging
import os
from datetime import datetime
from importlib import import_module
from typing import List, Union, Callable, Optional, Dict
import PIL.Image
import deepspeed
import torch
from torch import Tensor
from torch.nn import init
from transformers import PreTrainedModel, AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import HybridCache
from transformers.generation.utils import GenerateOutput
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled, deepspeed_config

from ovis.model.configuration_ovis import OvisConfig
from ovis.model.conversation_formatter import ConversationFormatter
from ovis.util.constants import IGNORE_ID, BEGIN_LINE, END_LINE, IMAGE_ATOM_ID, IMAGE_INDICATOR_IDS, \
    IMAGE_TOKEN_ID
from ovis.util.utils import rank0_print

def generate(
    model,
    inputs: Optional[torch.Tensor] = None,
    **kwargs
) -> Union[GenerateOutput, torch.LongTensor]:
    assert inputs.shape[0] == 1, 'Currently, only support `batch_size=1`'
    _, inputs_embeds, labels, attention_mask = merge_multimodal(
        text_input_ids=inputs,
        text_attention_masks=kwargs.pop('attention_mask'),
        text_labels=None,
        pixel_values=kwargs.pop('pixel_values')
    )
    if getattr(model.generation_config, 'cache_implementation') == 'hybrid':  
        kwargs['past_key_values'] = model._get_hybrid_cache_for_llm(
            getattr(kwargs, "num_beams", 1), kwargs['max_new_tokens'] + inputs_embeds.shape[-2])
        model.get_llm()._supports_cache_class = True
        kwargs['cache_implementation'] = None

    return model.llm.generate(inputs=None, inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)


def merge_multimodal(
    model,
    text_input_ids: torch.Tensor,
    text_attention_masks: torch.Tensor,
    text_labels: Optional[torch.Tensor],
    pixel_values: List[Optional[torch.Tensor]]
):
    input_device = text_input_ids.device
    visual_vocab_szie = model.get_visual_tokenizer().config.vocab_size
    visual_indicator_embeds = model.get_vte()(
        torch.tensor(
            list(range(visual_vocab_szie - 5, visual_vocab_szie)),
            dtype=torch.long,
            device=model.get_visual_tokenizer().device
        )
    ).to(device=input_device)

    # When inference, sample can include only text with `None` pixel_value
    num_images = [x.shape[0] if x is not None else 0 for x in pixel_values]
    if sum(num_images) > 0:
        visual_tokens = model.visual_tokenizer(torch.cat([x for x in pixel_values if x is not None], dim=0))
        visual_embeds = torch.split(model.get_vte()(visual_tokens).to(dtype=model.dtype, device=input_device),
                                    split_size_or_sections=num_images, dim=0)
        visual_input_ids = torch.split(torch.argmax(visual_tokens, dim=-1).to(device=input_device),
                                       split_size_or_sections=num_images, dim=0)
        visual_labels = [torch.full(x.shape, IGNORE_ID, dtype=torch.long, device=input_device) for x in
                         visual_input_ids]
    else:
        # just placeholders
        visual_embeds = [None] * len(num_images)
        visual_input_ids = [None] * len(num_images)
        visual_labels = [None] * len(num_images)
    # just placeholders
    text_labels = torch.full(text_input_ids.shape, IGNORE_ID, dtype=torch.long, device=input_device)

    input_embeds = []
    attention_masks = []
    labels = []
    for text_input_id, text_label, text_attention_mask, visual_embed, visual_input_id, visual_label in zip(
            text_input_ids, text_labels, text_attention_masks, visual_embeds, visual_input_ids, visual_labels
    ):
        placeholder_token_mask = torch.lt(text_input_id, 0)
        text_embed = model.get_wte()(torch.masked_fill(text_input_id, placeholder_token_mask, 0))
        for i, indicator_id in enumerate(IMAGE_INDICATOR_IDS):
            text_embed[text_input_id == indicator_id] = visual_indicator_embeds[i]
        image_atom_positions = torch.where(torch.eq(text_input_id, IMAGE_ATOM_ID))[0].tolist()
        if len(image_atom_positions) > 0:
            input_embed_parts = []
            attention_mask_parts = []
            label_parts = []
            prev_image_atom_position = -1
            for index, image_atom_position in enumerate(image_atom_positions):
                input_embed_parts.append(
                    text_embed[prev_image_atom_position + 1:image_atom_position, :])
                label_parts.append(
                    text_label[prev_image_atom_position + 1:image_atom_position])
                attention_mask_parts.append(
                    text_attention_mask[prev_image_atom_position + 1:image_atom_position])
                input_embed_parts.append(visual_embed[index])
                attention_mask_parts.append(
                    torch.ones_like(visual_label[index], dtype=torch.bool))
                label_parts.append(visual_label[index])
                prev_image_atom_position = image_atom_position
            if prev_image_atom_position + 1 < text_input_id.shape[0]:
                input_embed_parts.append(
                    text_embed[prev_image_atom_position + 1:, :])
                attention_mask_parts.append(
                    text_attention_mask[prev_image_atom_position + 1:])
                label_parts.append(
                    text_label[prev_image_atom_position + 1:])
            input_embed = torch.cat(input_embed_parts, dim=0)
            attention_mask = torch.cat(attention_mask_parts, dim=0)
            label = torch.cat(label_parts, dim=0)
        else:
            input_embed = text_embed
            attention_mask = text_attention_mask
            label = text_label
            if model.training:
                # Make visual_embed & visual_indicator_embeds involved in the backward graph,
                # to be compatible with deepspeed zero and ddp.
                input_embed += torch.sum(visual_embed * 0.0) + torch.sum(visual_indicator_embeds * 0.0)
        input_embeds.append(input_embed)
        attention_masks.append(attention_mask)
        labels.append(label)

    batch_input_embeds = torch.nn.utils.rnn.pad_sequence(input_embeds, batch_first=True, padding_value=0.0)[:,
                         :model.config.multimodal_max_length, :]
    batch_attention_mask = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=False)[
                           :,
                           :model.config.multimodal_max_length]
    batch_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_ID)[:,
                   :model.config.multimodal_max_length]

    return visual_input_ids, batch_input_embeds, batch_labels, batch_attention_mask