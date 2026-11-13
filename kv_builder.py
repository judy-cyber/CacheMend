import hashlib
import json
from pathlib import Path
import torch
from .cache import splice_local_kv
from .data import _extract_local_kv

FORMAT_VERSION = 3


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


def _file_sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def model_fingerprint(model_path, *, weight_check='stat', model=None, tokenizer=None):
    if weight_check not in ('stat', 'sha256'):
        raise ValueError('weight_check must be stat or sha256')
    root = Path(model_path).resolve()
    if not root.is_dir():
        raise ValueError(f'Model checkpoint directory does not exist: {root}')
    metadata, weights = [], []
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        if path.suffix == '.safetensors' or path.name.startswith('pytorch_model') and path.suffix == '.bin':
            stat = path.stat()
            entry = {'file': path.name, 'size': stat.st_size}
            if weight_check == 'sha256':
                entry['sha256'] = _file_sha256(path)
            else:
                entry['mtime_ns'] = stat.st_mtime_ns
            weights.append(entry)
        elif path.suffix in ('.json', '.model', '.txt', '.py', '.tiktoken', '.vocab'):
            metadata.append({'file': path.name, 'sha256': _file_sha256(path)})
    identity = {'model_path': str(root), 'weight_check': weight_check,
                'metadata': metadata, 'weights': weights}
    if model is not None:
        identity['model_class'] = type(model).__module__ + '.' + type(model).__name__
        identity['model_config'] = model.config.to_dict()
    if tokenizer is not None:
        identity['tokenizer_class'] = type(tokenizer).__module__ + '.' + type(tokenizer).__name__
        identity['tokenizer_settings'] = getattr(tokenizer, 'init_kwargs', {})
    # Persist only primitive metadata so caches remain weights_only-loadable.
    identity = json.loads(json.dumps(identity, sort_keys=True, default=str))
    return {'digest': _digest(identity), 'identity': identity}


def cache_signature(document_ids, model_path, dtype, restore_rope, *, fingerprint=None,
                    prefix_length=None, document_ranges=None):
    if fingerprint is None:
        fingerprint = model_fingerprint(model_path)
    identity = {'format_version': FORMAT_VERSION, 'document_ids': document_ids,
                'model_fingerprint': fingerprint['digest'], 'dtype': dtype,
                'restore_rope': restore_rope, 'prefix_length': prefix_length,
                'document_ranges': document_ranges}
    return _digest(identity)


@torch.no_grad()
def build_document_kv(model, split, *, local_batch_size=4, restore_rope=True):
    local = _extract_local_kv(model, prefix_ids=split['prefix_ids'],
                             documents=split['documents'], batch_size=local_batch_size)
    spliced = splice_local_kv(local, prefix_len=int(split['prefix_ids'].numel()),
                              model=model, restore_rope=restore_rope)
    return {'spliced': [(k.cpu(), v.cpu()) for k, v in spliced],
            'document_ids': split['document_ids'].cpu(),
            'prefix_length': int(split['prefix_ids'].numel()),
            'document_ranges': split['document_ranges'], 'restore_rope': restore_rope}


def _normalize_and_validate(payload, *, model, split, restore_rope, row_index):
    if not isinstance(payload, dict):
        raise ValueError('KV cache payload must be a dictionary')
    result = dict(payload)
    prefix = result.get('prefix_length', result.get('prefix_len'))
    if prefix != int(split['prefix_ids'].numel()):
        raise ValueError('KV cache prefix length differs from current input')
    ids = result.get('document_ids')
    if not isinstance(ids, torch.Tensor) or ids.ndim != 1 or not torch.equal(ids, split['document_ids'].cpu()):
        raise ValueError('KV cache document token IDs differ from current input')
    if result.get('document_ranges') != split['document_ranges']:
        raise ValueError('KV cache chunk boundaries differ from current input')
    saved_index = result.get('row_index', result.get('index'))
    if saved_index is not None and int(saved_index) != row_index:
        raise ValueError('KV cache sample index differs from requested sample')
    if 'restore_rope' in result and result['restore_rope'] != restore_rope:
        raise ValueError('KV cache RoPE setting differs from current setting')
    layers = result.get('packed_spliced', result.get('spliced'))
    if not isinstance(layers, (tuple, list)) or len(layers) != len(model.model.layers):
        raise ValueError('KV cache layer count differs from current model')
    config = model.config
    kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
    head_dim = getattr(model.model.layers[0].self_attn, 'head_dim', config.hidden_size // config.num_attention_heads)
    expected_shape = (int(ids.numel()), kv_heads, head_dim)
    expected_dtype = next(model.parameters()).dtype
    normalized = []
    for pair in layers:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError('Each KV layer must contain one key/value pair')
        converted = []
        for tensor in pair:
            if not isinstance(tensor, torch.Tensor):
                raise ValueError('KV entries must be tensors')
            if tensor.ndim == 4 and tensor.shape[0] == 1:
                tensor = tensor[0].transpose(0, 1).contiguous()
            if tuple(tensor.shape) != expected_shape or tensor.dtype != expected_dtype:
                raise ValueError('KV cache tensor shape/dtype differs from current model/input')
            converted.append(tensor)
        normalized.append(tuple(converted))
    result.update(spliced=normalized, prefix_length=prefix)
    return result


def _cache_candidates(path, row_index, legacy_cache_dir):
    candidates = [path, path.parent / f'{row_index:04d}.pt']
    if legacy_cache_dir is not None:
        candidates += [Path(legacy_cache_dir) / f'{row_index:06d}.pt',
                       Path(legacy_cache_dir) / f'{row_index:04d}.pt']
    return list(dict.fromkeys(candidates))


def load_or_build_kv(path, *, signature, model, split, build=False,
                     local_batch_size=4, restore_rope=True, dataset_name='custom', row_index=0,
                     fingerprint=None, allow_legacy=False, legacy_cache_dir=None):
    path = Path(path)
    found = next((p for p in _cache_candidates(path, row_index, legacy_cache_dir) if p.is_file()), None)
    if found is not None:
        payload = torch.load(found, map_location='cpu', weights_only=True)
        modern = isinstance(payload, dict) and payload.get('format_version') == FORMAT_VERSION
        if modern:
            if payload.get('signature') != signature:
                raise ValueError(f'KV cache fingerprint/input/settings mismatch: {found}; use a new cache directory')
            saved_fingerprint = payload.get('model_fingerprint')
            if fingerprint is not None and (not isinstance(saved_fingerprint, dict) or saved_fingerprint.get('digest') != fingerprint['digest']):
                raise ValueError(f'KV cache model/tokenizer fingerprint mismatch: {found}')
        elif isinstance(payload, dict) and payload.get('format_version') is not None:
            raise ValueError(f'Unsupported KV cache format version: {payload["format_version"]}')
        elif isinstance(payload, dict) and payload.get('version') not in (None, 1, 2):
            raise ValueError(f'Unsupported legacy KV cache version: {payload["version"]}')
        elif not allow_legacy:
            raise ValueError(f'Legacy KV cache found: {found}. It has no strong model fingerprint; '
                             'use --allow-legacy-kv for explicit reuse, or build into a new cache directory')
        result = _normalize_and_validate(payload, model=model, split=split,
                                         restore_rope=restore_rope, row_index=row_index)
        result['cache_verification'] = 'verified' if modern else 'legacy-unverified'
        result['cache_source'] = str(found)
        return result
    if not build:
        raise FileNotFoundError(f'Missing KV cache: {path}; run build_kv.py or the all stage first')
    if fingerprint is None:
        raise ValueError('Building verified caches requires a model fingerprint')
    payload = build_document_kv(model, split, local_batch_size=local_batch_size, restore_rope=restore_rope)
    payload.update(format_version=FORMAT_VERSION, signature=signature,
                   model_fingerprint=fingerprint, dataset_name=dataset_name, row_index=row_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    temp.replace(path)
    payload.update(cache_verification='verified', cache_source=str(path))
    return payload
