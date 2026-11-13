import importlib
import numpy as np
import torch
import torch.nn.functional as F
from .cache import to_legacy_cache
from .masks import attention_mask
from .scoring import _main_query_attn_v, _wo_head_norms, key_deviation, regroup_windows


def native_cache(spliced, device):
    return tuple((k.transpose(0,1).unsqueeze(0).to(device).contiguous(),
                  v.transpose(0,1).unsqueeze(0).to(device).contiguous()) for k,v in spliced)


@torch.no_grad()
def packed_recompute(model, document_ids, spliced, positions):
    device = next(model.parameters()).device
    base = list(native_cache(spliced, device))
    positions = torch.tensor(sorted(set(map(int, positions))),device=device,dtype=torch.long)
    if positions.numel() == 0: return tuple(base)
    core, config = model.model, model.config
    length = int(document_ids.numel())
    if int(positions.min()) < 0 or int(positions.max()) >= length:
        raise ValueError('Repair position outside document')
    nh, nk = config.num_attention_heads, config.num_key_value_heads
    hd = getattr(config,'head_dim',config.hidden_size // nh)
    module = importlib.import_module(type(core.layers[0].self_attn).__module__)
    apply_rotary = module.apply_rotary_pos_emb
    ids = document_ids.to(device)
    cos,sin = core.rotary_emb(core.embed_tokens(ids[:1][None]),torch.arange(length,device=device)[None])
    cos,sin = cos.index_select(1,positions),sin.index_select(1,positions)
    mask = attention_mask(model, positions, length, base[0][0].dtype, device)
    h = core.embed_tokens(ids.index_select(0,positions)[None])
    for li,layer in enumerate(core.layers):
        attn = layer.self_attn
        residual = h
        x = layer.input_layernorm(h)
        q = attn.q_proj(x).view(1,-1,nh,hd)
        k = attn.k_proj(x).view(1,-1,nk,hd)
        v = attn.v_proj(x).view(1,-1,nk,hd).transpose(1,2)
        if hasattr(attn,'q_norm'): q = attn.q_norm(q)
        if hasattr(attn,'k_norm'): k = attn.k_norm(k)
        q,k = apply_rotary(q.transpose(1,2),k.transpose(1,2),cos,sin)
        kc,vc = base[li]
        kc.index_copy_(2,positions,k)
        vc.index_copy_(2,positions,v)
        output = F.scaled_dot_product_attention(q,kc.repeat_interleave(nh//nk,dim=1),
                    vc.repeat_interleave(nh//nk,dim=1),attn_mask=mask,dropout_p=0.,
                    scale=attn.scaling,is_causal=False)
        h = residual + attn.o_proj(output.transpose(1,2).contiguous().view(1,positions.numel(),-1))
        residual = h
        mlp = layer.mlp(layer.post_attention_layernorm(h))
        h = residual + (mlp[0] if isinstance(mlp,tuple) else mlp)
    return tuple(base)


def normalize(signal, signal_name='score'):
    signal = np.asarray(signal)
    if not np.isfinite(signal).all():
        raise ValueError(f'{signal_name} contains NaN or infinity')
    with np.errstate(over='ignore', invalid='ignore'):
        shifted = np.clip(signal-np.min(signal),0,None)
    if not np.isfinite(shifted).all():
        raise ValueError(f'{signal_name} normalization produced NaN or infinity')
    peak = float(shifted.max())
    return shifted/peak if peak > 0 else np.zeros_like(shifted)


@torch.no_grad()
def repair(model, document_ids, suffix_ids, spliced, *, budget=.10, check_layer=1,
           window_size=64, density_threshold=8, prefix_length=0,
           head_aggregation='max', value_weight=True, output_weight=True,
           query_chunk_size=128):
    if not 0 < budget <= 1: raise ValueError('budget must be in (0,1]')
    if not 0 <= check_layer < len(model.model.layers): raise ValueError('Invalid check layer')
    if window_size < 1 or not 1 <= density_threshold <= window_size:
        raise ValueError('Expected 1 <= density_threshold <= window_size')
    length = int(document_ids.numel())
    if not 0 <= prefix_length <= length: raise ValueError('Invalid prefix length')
    device = next(model.parameters()).device
    nh,nk = model.config.num_attention_heads,model.config.num_key_value_heads
    hd = getattr(model.config,'head_dim',model.config.hidden_size//nh)
    values = None
    if value_weight:
        values = torch.stack([v.to(device=device,dtype=torch.float32).norm(dim=2)
                    .repeat_interleave(nh//nk,dim=1).transpose(0,1) for _,v in spliced])
    weights = _wo_head_norms(model,nh,hd) if output_weight else None
    relevance = _main_query_attn_v(model,native_cache(spliced,device),suffix_ids.to(device),
                    length,values,device,wo_head=weights,chunk_size=query_chunk_size,
                    aggregation=head_aggregation)
    if relevance is None: raise RuntimeError('Model did not return query attention')
    relevance = normalize(np.asarray(relevance,dtype=np.float64), 'Q2D score')
    drift = key_deviation(model,document_ids,spliced,check_layer=check_layer,device=device)
    score = relevance * normalize(drift.cpu().numpy().astype(np.float64), 'key drift')
    # Retain the original selector's float32 rounding before argsort.
    score = score.astype(np.float32)
    candidates = np.argsort(-score)
    candidates = candidates[candidates >= prefix_length]
    count = min(length,max(1,round(budget*length)))
    candidates = candidates[:count].tolist()
    selected = regroup_windows(candidates,length,window=window_size,tau=density_threshold)
    return packed_recompute(model,document_ids,spliced,selected),selected
