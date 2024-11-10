import torch


def slice2d(x, start, end):
    out = x[:, :, start:end, ...]
    return out


def compress2d(x, start, end):
    if (end - start) % 2 != 0:
        x = x[:, :, start:end - 1, ...]
        out = (x[:, :, ::2, ...] + x[:, :, 1::2, ...]) / 2
        out = torch.cat((out, x[:, :, -1:, ...]), dim=2)
    else:
        x = x[:, :, start:end, ...]
        out = (x[:, :, ::2, ...] + x[:, :, 1::2, ...]) / 2
    return out


def compress_decode(x, sink_tokens):
    seq_len = x.size(2)
    sink_cache = slice2d(x, 0, sink_tokens)
    compressed_cache = compress2d(x, sink_tokens, seq_len)
    complete_cache = torch.cat((sink_cache, compressed_cache), dim=2)
    return complete_cache


def compress_prefill(x, sink_tokens, retention_window_length, retention_window_start):
    seq_len = x.size(2)
    while seq_len - retention_window_start > retention_window_length:
        x_sink = slice2d(x, 0, sink_tokens)
        x_compress_chunk = slice2d(x, sink_tokens, retention_window_start + retention_window_length)
        x_future = slice2d(x, retention_window_start + retention_window_length, seq_len)
        x_compressed = compress2d(x_compress_chunk, 0, x_compress_chunk.size(2))
        x = torch.cat((x_sink, x_compressed, x_future), dim=2)
        retention_window_start = sink_tokens + x_compressed.size(2)
        seq_len = x.size(2)
    return x


def compress_kv_cache(past_key_values, sink_tokens, retention_window_length, retention_window_start, skip_prefill_compression, prefill=False):
    if past_key_values is None:
        return None
    seq_len = past_key_values[0][0].size(2)
    current_uncompressed_window_length = seq_len - retention_window_start
    if (seq_len - retention_window_start) < 0 or current_uncompressed_window_length <= retention_window_length:
        return past_key_values, retention_window_start
    else:
        if prefill and not skip_prefill_compression:
            past_key_values = tuple(
                (compress_prefill(key, sink_tokens, retention_window_length, retention_window_start),
                 compress_prefill(value, sink_tokens, retention_window_length, retention_window_start)) for key, value
                in past_key_values
            )
        else:
            past_key_values = tuple(
                (compress_decode(key, sink_tokens), compress_decode(value, sink_tokens)) for key, value in
                past_key_values
            )
        next_retention_window_start = past_key_values[0][0].size(2)
        return past_key_values, next_retention_window_start
