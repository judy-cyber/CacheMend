from __future__ import annotations
import importlib
from typing import Sequence
import torch
from transformers import Cache as _HFCache, DynamicCache as _DynamicCache
import json
from .masks import attention_mask
_WO_HEAD_NORM_CACHE = {}
_HEAD_AGG_MODES = ('max', 'sum', 'zsum', 'ranksum')
_F2H_AGG_LOGGED = False

def _as_score_tensor(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32, device=device)

def _wo_head_norms(model, n_heads: int, head_dim: int) -> 'torch.Tensor':
    key = (id(model), int(n_heads), int(head_dim))
    hit = _WO_HEAD_NORM_CACHE.get(key)
    if hit is not None:
        return hit
    rows = []
    for layer in model.model.layers:
        weight = layer.self_attn.o_proj.weight.detach().float()
        rows.append(weight.view(weight.shape[0], int(n_heads), int(head_dim)).norm(dim=(0, 2)))
    stacked = torch.stack(rows)
    _WO_HEAD_NORM_CACHE[key] = stacked
    return stacked

def _aggregate_head_scores(head: 'torch.Tensor', mode: str, nkv: int=0, kvagg: str='0') -> 'torch.Tensor':
    if kvagg in ('mean', 'max') and nkv > 1 and (head.shape[0] % nkv == 0):
        rep = head.shape[0] // nkv
        if rep > 1:
            head = head.view(nkv, rep, -1)
            head = head.mean(dim=1) if kvagg == 'mean' else head.amax(dim=1)
    if mode == 'max':
        return head.max(dim=0).values
    if mode == 'sum':
        return head.sum(dim=0)
    if mode == 'zsum':
        mu = head.mean(dim=1, keepdim=True)
        sd = head.std(dim=1, keepdim=True).clamp_min(1e-06)
        return ((head - mu) / sd).clamp_min(0.0).sum(dim=0)
    if mode == 'ranksum':
        order = torch.argsort(torch.argsort(head, dim=1, stable=True), dim=1)
        denom = float(max(1, head.shape[1] - 1))
        return (order.float() / denom).sum(dim=0)
    raise ValueError(f'unknown head aggregation mode: {mode}')

@torch.no_grad()
def _main_query_attn_v(model, past, suffix_ids, document_length, vnorm_h, device, *, wo_head=None, chunk_size: int=128, aggregation: str='max'):
    if document_length <= 0 or suffix_ids is None or suffix_ids.numel() == 0:
        return None
    config = getattr(model, 'config', None)
    prev = getattr(config, '_attn_implementation', None)
    global _F2H_AGG_LOGGED
    _agg = aggregation
    _kvagg = '0'
    if _agg not in _HEAD_AGG_MODES:
        raise ValueError(f'Unknown aggregation: {_agg}')
    _nkv = int(getattr(config, 'num_key_value_heads', 0) or 0) if config is not None else 0
    _nq = int(getattr(config, 'num_attention_heads', 0) or 0) if config is not None else 0
    if not _F2H_AGG_LOGGED and (_agg != 'max' or _kvagg in ('mean', 'max')):
        print(json.dumps({'f2h_head_agg': {'mode': _agg, 'kvagg': _kvagg, 'n_kv_heads': _nkv, 'n_q_heads': _nq}}), flush=True)
        _F2H_AGG_LOGGED = True
    if config is not None:
        config._attn_implementation = 'eager'
    try:
        cache = past if isinstance(past, _HFCache) else _DynamicCache.from_legacy_cache(past)
        total = int(suffix_ids.shape[1])
        per_head = None
        step = max(1, int(chunk_size))
        for start in range(0, total, step):
            chunk = suffix_ids[:, start:start + step]
            outputs = model(input_ids=chunk, past_key_values=cache, use_cache=True, output_attentions=True, return_dict=True)
            atts = getattr(outputs, 'attentions', None)
            if not atts:
                return None
            for li, layer_attn in enumerate(atts):
                a = layer_attn[0][:, :, :document_length].float().sum(dim=1)
                vh = None if vnorm_h is None else _as_score_tensor(vnorm_h[li][:, :document_length], a.device)
                if wo_head is not None:
                    wh = _as_score_tensor(wo_head[li], a.device).view(-1, 1)
                    a = a * wh
                head = a if vh is None else a * vh
                if per_head is None:
                    per_head = [torch.zeros_like(head) for _ in atts]
                per_head[li] += head
            cache = outputs.past_key_values
            del outputs, atts
        score = torch.zeros(document_length, dtype=torch.float32, device=device)
        for head in per_head or []:
            score += _aggregate_head_scores(head, _agg, nkv=_nkv, kvagg=_kvagg)
    finally:
        if config is not None:
            config._attn_implementation = prev
    return score.cpu().numpy()

def regroup_windows(candidates, doclen, *, window=8, tau=5, forced=None):
    cand = set((int(t) for t in candidates))
    forced = set((int(t) for t in forced or []))
    budget = len(cand)
    if budget == 0:
        return []
    windows = []
    for i in sorted(cand):
        win = [p for p in range(i, min(i + window, doclen))]
        density = sum((1 for p in win if p in cand))
        if density >= tau:
            windows.append((density, win))
    windows.sort(key=lambda x: -x[0])
    recompute = set(forced)
    for density, win in windows:
        if len(recompute) >= budget:
            break
        remaining = budget - len(recompute)
        add = sorted(win)[:remaining]
        recompute.update(add)
    return sorted(recompute)[:budget]

def pre_rope_k(attn, hidden_normed: 'torch.Tensor', model) -> 'torch.Tensor':
    if hasattr(attn, 'k_proj'):
        return attn.k_proj(hidden_normed)
    if hasattr(attn, 'qkv_proj'):
        fused = attn.qkv_proj(hidden_normed)
        head_dim = int(getattr(model.config, 'head_dim', 0) or 0)
        n_heads = int(getattr(model.config, 'num_attention_heads', 0) or 0)
        n_kv = int(getattr(model.config, 'num_key_value_heads', n_heads) or n_heads)
        if not head_dim and n_heads:
            head_dim = int(fused.shape[-1] // (n_heads + 2 * n_kv))
        q_size = n_heads * head_dim
        kv_size = n_kv * head_dim
        if q_size + 2 * kv_size != fused.shape[-1]:
            raise RuntimeError(f'unexpected fused qkv width {fused.shape[-1]} for heads={n_heads} kv_heads={n_kv} head_dim={head_dim}')
        _, key, _ = fused.split([q_size, kv_size, kv_size], dim=-1)
        return key
    if hasattr(attn, 'c_attn'):
        fused = attn.c_attn(hidden_normed)
        _, key, _ = fused.split(fused.shape[-1] // 3, dim=-1)
        return key
    raise AttributeError(f'{type(attn).__name__} exposes none of k_proj / qkv_proj / c_attn; add its key projection to pre_rope_k()')

@torch.no_grad()
def key_deviation(model, document_ids: torch.Tensor, spliced: Sequence[Sequence[tuple[torch.Tensor, torch.Tensor]]], *, check_layer: int=1, device: torch.device | str, full_past=None) -> torch.Tensor:
    seq_len = int(document_ids.numel())
    if full_past is not None:
        k_new = full_past[check_layer][0][0, :, :seq_len, :].transpose(0, 1).reshape(seq_len, -1).float()
        k_old = spliced[check_layer][0].to(device).reshape(seq_len, -1).float()
        return ((k_new - k_old) ** 2).sum(dim=1)
    core = model.model
    ids = document_ids.view(1, -1).to(device)
    hidden = core.embed_tokens(ids)
    position_ids = torch.arange(seq_len, device=device).view(1, -1)
    position_embeddings = core.rotary_emb(hidden, position_ids)
    if getattr(model.config, 'sliding_window', None) is None:
        additive = torch.full((1, 1, seq_len, seq_len), float('-inf'), device=device, dtype=hidden.dtype)
        causal = torch.triu(additive, diagonal=1)
    else:
        causal = attention_mask(model, position_ids.reshape(-1), seq_len, hidden.dtype, device)
    for layer_index in range(check_layer):
        out = core.layers[layer_index](hidden, attention_mask=causal, position_ids=position_ids, position_embeddings=position_embeddings, use_cache=False)
        hidden = out[0] if isinstance(out, tuple) else out
    layer = core.layers[check_layer]
    normed = layer.input_layernorm(hidden)
    attn = layer.self_attn
    k_pre = pre_rope_k(attn, normed, model)
    kv_heads = int(getattr(model.config, 'num_key_value_heads', getattr(model.config, 'num_attention_heads', 1)))
    head_dim = int(getattr(attn, 'head_dim', k_pre.shape[-1] // kv_heads))
    k_pre = k_pre.reshape(1, seq_len, kv_heads, head_dim)
    if hasattr(attn, 'k_norm'):
        k_pre = attn.k_norm(k_pre)
    k_pre = k_pre.transpose(1, 2)
    module = importlib.import_module(type(attn).__module__)
    # Only the key is needed. Applying the native helper to a dummy query
    # with the same shape avoids an unnecessary query projection.
    _, k_new = module.apply_rotary_pos_emb(k_pre, k_pre, *position_embeddings)
    k_new = k_new[0].transpose(0, 1).reshape(seq_len, -1).float()
    k_old = spliced[check_layer][0].to(device).reshape(seq_len, -1).float()
    return ((k_new - k_old) ** 2).sum(dim=1)
