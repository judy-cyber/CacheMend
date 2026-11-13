from __future__ import annotations
from typing import Sequence
import time
import torch
from .cache import logits_to_keep_kwargs, to_legacy_cache
from .data import _empty_cuda_cache

def cache_for_model(model, cache):
    from transformers import Cache, DynamicCache
    if cache is None or isinstance(cache, Cache):
        return cache
    return DynamicCache.from_legacy_cache(cache)

def cache_from_output(model, cache):
    return cache

def sample_top_p(logits: torch.Tensor, *, temperature: float, top_p: float, generator: torch.Generator) -> torch.Tensor:
    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
    sampled_sorted = torch.multinomial(sorted_probs, num_samples=1, generator=generator)
    return sorted_indices.gather(-1, sampled_sorted)

def repeat_past(past, batch_size: int) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    return tuple(((key.expand(batch_size, -1, -1, -1).contiguous(), value.expand(batch_size, -1, -1, -1).contiguous()) for key, value in to_legacy_cache(past)))

@torch.no_grad()
def forward_suffix_chunked(model, *, past, suffix_ids: torch.Tensor, chunk_size: int) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
    if chunk_size <= 0:
        raise ValueError('suffix chunk size must be positive')
    current_past = past
    last_logits = None
    for start in range(0, suffix_ids.shape[1], chunk_size):
        outputs = model(input_ids=suffix_ids[:, start:start + chunk_size], past_key_values=cache_for_model(model, current_past), use_cache=True, return_dict=True, **logits_to_keep_kwargs(model))
        current_past = cache_from_output(model, outputs.past_key_values)
        last_logits = outputs.logits[0, -1, :].float()
    if last_logits is None:
        raise ValueError('Suffix tokenization produced no tokens')
    return (last_logits, current_past)

_cache_for_model = cache_for_model
_cache_from_output = cache_from_output
_logits_to_keep_kwargs = logits_to_keep_kwargs
_sample_top_p = sample_top_p
_repeat_past = repeat_past

@torch.no_grad()
def _greedy_text(model, tokenizer, *, first_logits: torch.Tensor, suffix_past, max_new_tokens: int, first_token_callback=None) -> str:
    logits = first_logits.unsqueeze(0)
    past = suffix_past
    generated: list[int] = []
    eos_id = tokenizer.eos_token_id
    for step in range(max_new_tokens):
        token = torch.argmax(logits, dim=-1, keepdim=True)
        token_id = int(token.item())
        if step == 0 and first_token_callback is not None:
            first_token_callback(token_id)
        if eos_id is not None and token_id == int(eos_id):
            break
        generated.append(token_id)
        if step + 1 < max_new_tokens:
            outputs = model(input_ids=token, past_key_values=_cache_for_model(model, past), use_cache=True, return_dict=True, **_logits_to_keep_kwargs(model))
            logits = outputs.logits[:, -1, :]
            past = _cache_from_output(model, outputs.past_key_values)
    return tokenizer.decode(generated, skip_special_tokens=True)

@torch.no_grad()
def _sample_texts(model, tokenizer, *, first_logits: torch.Tensor, suffix_past, sample_count: int, sample_batch_size: int, max_new_tokens: int, temperature: float, top_p: float, seed: int) -> list[str]:
    device = first_logits.device
    eos_id = tokenizer.eos_token_id
    fallback_id = int(eos_id if eos_id is not None else 0)
    texts = []
    for batch_start in range(0, sample_count, sample_batch_size):
        batch_size = min(sample_batch_size, sample_count - batch_start)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + batch_start)
        logits = first_logits.unsqueeze(0).expand(batch_size, -1)
        past = _repeat_past(suffix_past, batch_size)
        generated: list[list[int]] = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for step in range(max_new_tokens):
            token = _sample_top_p(logits, temperature=temperature, top_p=top_p, generator=generator)
            token = token.squeeze(-1)
            for batch_index, token_id in enumerate(token.detach().cpu().tolist()):
                if finished[batch_index]:
                    continue
                if eos_id is not None and token_id == int(eos_id):
                    finished[batch_index] = True
                else:
                    generated[batch_index].append(int(token_id))
            if bool(torch.all(finished)) or step + 1 >= max_new_tokens:
                break
            model_token = token.masked_fill(finished, fallback_id).unsqueeze(-1)
            outputs = model(input_ids=model_token, past_key_values=_cache_for_model(model, past), use_cache=True, return_dict=True, **_logits_to_keep_kwargs(model))
            logits = outputs.logits[:, -1, :]
            past = _cache_from_output(model, outputs.past_key_values)
        texts.extend((tokenizer.decode(token_ids, skip_special_tokens=True) for token_ids in generated))
        del past, generated
        _empty_cuda_cache(device)
    return texts


@torch.no_grad()
def decode(model, tokenizer, *, past, suffix_ids, max_new_tokens=32,
           do_sample=False, sample_count=1, sample_batch_size=1,
           temperature=.7, top_p=.9, seed=0, suffix_chunk_size=64,
           timing_origin=None):
    if min(max_new_tokens, sample_count, sample_batch_size, suffix_chunk_size) < 1:
        raise ValueError('Decoding sizes must be positive')
    if temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError('Invalid temperature/top-p')
    logits, suffix_past = forward_suffix_chunked(
        model, past=past, suffix_ids=suffix_ids, chunk_size=suffix_chunk_size)
    suffix_past = to_legacy_cache(suffix_past)
    timing = {}
    def first_token_ready(token_id):
        if logits.device.type == 'cuda':
            torch.cuda.synchronize(logits.device)
        timing['ttft_seconds'] = time.perf_counter() - timing_origin
    greedy = _greedy_text(model, tokenizer, first_logits=logits,
                          suffix_past=suffix_past, max_new_tokens=max_new_tokens,
                          first_token_callback=first_token_ready if timing_origin is not None else None)
    samples = []
    if do_sample:
        samples = _sample_texts(model, tokenizer, first_logits=logits,
            suffix_past=suffix_past, sample_count=sample_count,
            sample_batch_size=sample_batch_size, max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p, seed=seed)
    return {'greedy_text': greedy, 'sample_texts': samples,
            'answers': samples if do_sample else [greedy] * sample_count, **timing}
