from __future__ import annotations
import re
from typing import Any, Sequence
import torch
from .cache import logits_to_keep_kwargs as _logits_to_keep_kwargs, to_legacy_cache as _to_legacy_cache
DOCUMENT_PATTERN = re.compile('(?m)^Document \\d+:\\n')
QUESTION_MARKER = '\n\nAnswer the question based on the given documents.'

def _token_index_at(offsets: Sequence[Sequence[int]], char_position: int) -> int:
    for index, (_, end) in enumerate(offsets):
        if int(end) > char_position:
            return index
    return len(offsets)

def _split_prompt_tokens(tokenizer, text: str) -> dict[str, Any]:
    document_matches = list(DOCUMENT_PATTERN.finditer(text))
    question_start = text.rfind(QUESTION_MARKER)
    if not document_matches or question_start < 0:
        raise ValueError('Could not parse RULER document or question markers')
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = torch.tensor(encoded['input_ids'], dtype=torch.long)
    offsets = encoded['offset_mapping']
    document_starts = [_token_index_at(offsets, match.start()) for match in document_matches]
    suffix_start = _token_index_at(offsets, question_start)
    boundaries = document_starts + [suffix_start]
    prefix_ids = input_ids[:document_starts[0]]
    documents = [input_ids[boundaries[index]:boundaries[index + 1]] for index in range(len(document_starts))]
    document_ids = torch.cat((prefix_ids, *documents))
    suffix_ids = input_ids[suffix_start:]
    if not torch.equal(torch.cat((document_ids, suffix_ids)), input_ids):
        raise AssertionError('RULER token partition is not lossless')
    ranges = []
    cursor = 0
    for index, document in enumerate(documents):
        length = int(document.numel()) + (int(prefix_ids.numel()) if index == 0 else 0)
        ranges.append([cursor, cursor + length])
        cursor += length
    return {'input_ids': input_ids, 'prefix_ids': prefix_ids, 'documents': documents, 'document_ids': document_ids, 'suffix_ids': suffix_ids, 'document_ranges': ranges}

@torch.no_grad()
def _extract_local_kv(model, *, prefix_ids: torch.Tensor, documents: Sequence[torch.Tensor], batch_size: int) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
    if batch_size <= 0:
        raise ValueError('local batch size must be positive')
    device = next(model.parameters()).device
    pad_id = int(getattr(model.config, 'pad_token_id', 0) or 0)
    local_layers: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
    sequences = [torch.cat((prefix_ids, document)) for document in documents]
    for start in range(0, len(sequences), batch_size):
        batch_sequences = sequences[start:start + batch_size]
        lengths = [int(sequence.numel()) for sequence in batch_sequences]
        maximum = max(lengths)
        input_ids = torch.full((len(batch_sequences), maximum), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros_like(input_ids)
        for batch_index, sequence in enumerate(batch_sequences):
            length = lengths[batch_index]
            input_ids[batch_index, :length] = sequence.to(device)
            attention_mask[batch_index, :length] = 1
        position_ids = torch.arange(maximum, device=device).unsqueeze(0).expand(len(batch_sequences), -1)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=True, return_dict=True, **_logits_to_keep_kwargs(model))
        cache = _to_legacy_cache(outputs.past_key_values)
        for batch_index, length in enumerate(lengths):
            layers = []
            for key, value in cache:
                layers.append((key[batch_index, :, :length, :].transpose(0, 1).detach().cpu().contiguous(), value[batch_index, :, :length, :].transpose(0, 1).detach().cpu().contiguous()))
            local_layers.append(layers)
        del outputs, cache, input_ids, attention_mask, position_ids
        _empty_cuda_cache(device)
    return local_layers

def _cuda_device(device: torch.device | str | None) -> torch.device | None:
    if device is None:
        return None
    resolved = torch.device(device)
    if resolved.type != 'cuda':
        return None
    return resolved

def _empty_cuda_cache(device: torch.device | str | None) -> None:
    resolved = _cuda_device(device)
    if resolved is None or not torch.cuda.is_available():
        return
    with torch.cuda.device(resolved):
        torch.cuda.empty_cache()
