import inspect
import torch


def attention_mask(model, positions, key_length, dtype, device):
    positions = positions.to(device=device, dtype=torch.long).reshape(-1)
    window = getattr(model.config, 'sliding_window', None)
    if window is not None:
        if not isinstance(window, int) or isinstance(window, bool) or window <= 0:
            raise ValueError('Model sliding_window must be a positive integer or None')
        if getattr(model.config, 'model_type', None) not in ('mistral', 'qwen2', 'qwen3'):
            raise NotImplementedError('Sliding-window attention is supported only for Mistral, Qwen2 and Qwen3')
        builder = getattr(model.model, '_prepare_4d_causal_attention_mask_with_cache_position', None)
        if builder is None:
            raise NotImplementedError('This model does not expose the native sliding-window mask builder')
        kwargs = dict(attention_mask=None, sequence_length=positions.numel(),
                      target_length=key_length, dtype=dtype, device=device,
                      cache_position=positions, batch_size=1,
                      config=model.config, past_key_values=None)
        params = inspect.signature(builder).parameters
        if not {'config', 'cache_position'} <= set(params):
            raise NotImplementedError('Unsupported native sliding-window mask signature')
        return builder(**{name: value for name, value in kwargs.items() if name in params})
    # Keep the original full-causal mask for models without a window.
    mask = torch.zeros((positions.numel(), key_length), dtype=dtype, device=device)
    mask.masked_fill_(torch.arange(key_length, device=device)[None] > positions[:, None],
                      torch.finfo(dtype).min)
    return mask[None, None]
