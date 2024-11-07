import torch


def slice2d(x, start, end):
    out = x[:, :, start:end, ...]
    return out


def compress2d(x, start, end):
    if (end - start) % 2 != 0:
        x = x[:, :, start:end - 1, ...]
        out = (x[:, :, ::2, ...] + x[:, :, 1::2, ...]) / 2
        return out
        # avg_tensor = (x[:, :, ::2, ...] + x[:, :, 1::2, ...]) / 2
        # last_slice = x[:, :, -1:, ...]  # Keep the last slice in 4D with singleton dimension
        # return torch.cat((avg_tensor, last_slice), dim=2)
    else:
        x = x[:, :, start:end, ...]
        out = (x[:, :, ::2, ...] + x[:, :, 1::2, ...]) / 2
        return out


def compress_fn(x, sink_tokens, compression_window):
    seq_len = x.size(2)
    if seq_len <= sink_tokens + compression_window:
        return x
    sink_cache = slice2d(x, 0, sink_tokens)
    other_token_cache = compress2d(x, sink_tokens, x.size(2))
    compressed_cache = torch.cat((sink_cache, other_token_cache), dim=2)
    return compressed_cache


def compress_kv_cache(past_key_values, sink_tokens, compression_window, prefill=False):
    if past_key_values is None:
        return None
    seq_len = past_key_values[0][0].size(2)
    if prefill:
        while seq_len > sink_tokens + compression_window:
            past_key_values = tuple(
                (compress_fn(key, sink_tokens, compression_window), compress_fn(value, sink_tokens, compression_window)) for key, value in past_key_values
            )
            seq_len = past_key_values[0][0].size(2)
    else:
        past_key_values = tuple(
            (compress_fn(key, sink_tokens, compression_window), compress_fn(value, sink_tokens, compression_window)) for key, value in past_key_values
        )
    return past_key_values
