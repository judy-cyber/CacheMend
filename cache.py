from __future__ import annotations
import math
import inspect
from pathlib import Path
from typing import Sequence
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
_LOGITS_KEEP_KEY = {}

def load_model(path: Path, *, attention_eager: bool=False, attn_implementation: str | None=None, device_map: str | None='cuda', target_device: str | None=None, torch_dtype: torch.dtype=torch.float16, max_memory: dict[int, str] | None=None):
    kwargs = {'torch_dtype': torch_dtype, 'trust_remote_code': True}
    if device_map is not None:
        kwargs['device_map'] = device_map
    if max_memory is not None:
        kwargs['max_memory'] = max_memory
    if attn_implementation is not None:
        kwargs['attn_implementation'] = attn_implementation
    elif attention_eager:
        kwargs['attn_implementation'] = 'eager'
    model = AutoModelForCausalLM.from_pretrained(str(path), **kwargs)
    if target_device is not None:
        model.to(target_device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
    return (model, tokenizer)

def to_legacy_cache(cache) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    if hasattr(cache, 'to_legacy_cache'):
        cache = cache.to_legacy_cache()
    return tuple(((key, value) for key, value in cache))

def logits_to_keep_kwargs(model, count: int=1) -> dict[str, int]:
    model_name = type(model).__name__
    keyword = _LOGITS_KEEP_KEY.get(model_name)
    if keyword is None:
        try:
            params = inspect.signature(model.forward).parameters
        except (TypeError, ValueError):
            params = {}
        if 'logits_to_keep' in params:
            keyword = 'logits_to_keep'
        elif 'num_logits_to_keep' in params:
            keyword = 'num_logits_to_keep'
        else:
            keyword = 'logits_to_keep'
        _LOGITS_KEEP_KEY[model_name] = keyword
    return {keyword: count}

def _rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    half = tensor.shape[-1] // 2
    first, second = (tensor[..., :half], tensor[..., half:])
    return torch.cat((-second, first), dim=-1)

def _rope_inverse_frequencies(head_dim: int, rope_theta: float, rope_scaling: object | None, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    index = torch.arange(0, head_dim, 2, device=device, dtype=dtype)
    inverse_frequency = 1.0 / float(rope_theta) ** (index / head_dim)
    if not rope_scaling:
        return inverse_frequency
    rope_type = rope_scaling.get('rope_type') or rope_scaling.get('type')
    if rope_type != 'llama3':
        raise NotImplementedError(f'Only standard and llama3 RoPE are supported, got {rope_type!r}')
    factor = float(rope_scaling['factor'])
    low_freq_factor = float(rope_scaling['low_freq_factor'])
    high_freq_factor = float(rope_scaling['high_freq_factor'])
    original = float(rope_scaling['original_max_position_embeddings'])
    low_wavelen = original / low_freq_factor
    high_wavelen = original / high_freq_factor
    wavelen = 2.0 * math.pi / inverse_frequency
    inv_low = torch.where(wavelen > low_wavelen, inverse_frequency / factor, inverse_frequency)
    smooth = (original / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed = (1.0 - smooth) * inv_low / factor + smooth * inv_low
    is_medium = ~(wavelen < high_wavelen) & ~(wavelen > low_wavelen)
    return torch.where(is_medium, smoothed, inv_low)

def apply_rope_delta(keys: torch.Tensor, delta_positions: torch.Tensor, *, rope_theta: float, rope_scaling: object | None) -> torch.Tensor:
    if keys.ndim != 3:
        raise ValueError(f'keys must have shape [S,H,D], got {tuple(keys.shape)}')
    seq_len, _, head_dim = keys.shape
    if head_dim <= 0 or head_dim % 2:
        raise ValueError(f'head_dim must be positive and even, got {head_dim}')
    if rope_theta <= 0:
        raise ValueError('rope_theta must be positive')
    delta = torch.as_tensor(delta_positions, device=keys.device).reshape(-1)
    if delta.numel() != seq_len:
        raise ValueError(f'Expected {seq_len} position deltas, got {delta.numel()}')
    work_dtype = torch.float64 if keys.dtype == torch.float64 else torch.float32
    inverse_frequency = _rope_inverse_frequencies(head_dim, rope_theta, rope_scaling, keys.device, work_dtype)
    phases = delta.to(work_dtype).unsqueeze(1) * inverse_frequency.unsqueeze(0)
    phases = torch.cat((phases, phases), dim=-1).unsqueeze(1)
    work = keys.to(work_dtype)
    return (work * phases.cos() + _rotate_half(work) * phases.sin()).to(keys.dtype)

def splice_local_kv(local_layers: Sequence[Sequence[tuple[torch.Tensor, torch.Tensor]]], *, prefix_len: int, model, restore_rope: bool=True) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if not local_layers:
        raise ValueError('local_layers must not be empty')
    num_layers = len(local_layers[0])
    if any((len(layers) != num_layers for layers in local_layers)):
        raise ValueError('local KV layer counts differ')
    local_positions: list[int] = []
    for document_index, layers in enumerate(local_layers):
        sequence_length = int(layers[0][0].shape[0])
        start = 0 if document_index == 0 else prefix_len
        local_positions.extend(range(start, sequence_length))
    local_position_ids = torch.tensor(local_positions, dtype=torch.long)
    global_position_ids = torch.arange(len(local_positions), dtype=torch.long)
    delta_positions = global_position_ids - local_position_ids
    result = []
    for layer_index in range(num_layers):
        keys = []
        values = []
        for document_index, layers in enumerate(local_layers):
            start = 0 if document_index == 0 else prefix_len
            keys.append(layers[layer_index][0][start:])
            values.append(layers[layer_index][1][start:])
        key = torch.cat(keys, dim=0)
        value = torch.cat(values, dim=0)
        if not restore_rope:
            pass
        else:
            key = apply_rope_delta(key, delta_positions, rope_theta=float(getattr(model.config, 'rope_theta', 10000.0)), rope_scaling=getattr(model.config, 'rope_scaling', None))
        result.append((key.detach().cpu().contiguous(), value))
    return result
