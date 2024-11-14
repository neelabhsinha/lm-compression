import torch

from src.enum.sequence_compression import SequenceCompressionPoolingType


class SequenceKVCompressor:
    def __init__(self, sink_tokens, pooling_type, initial_local_window, steepness_coefficient,
                 num_transformer_blocks, kv_seq_dim_idx=2):
        self.sink_tokens = sink_tokens
        self.pooling_type = pooling_type
        self.initial_local_window = initial_local_window
        self.steepness_coefficient = steepness_coefficient
        self.num_transformer_blocks = num_transformer_blocks
        self.kv_seq_dim_idx = kv_seq_dim_idx
        self.all_local_windows = self.get_all_local_windows()

    def get_all_local_windows(self):
        all_local_windows = []
        for layer_idx in range(self.num_transformer_blocks):
            local_window = int(self.initial_local_window * (1 + layer_idx/self.num_transformer_blocks
                                                            * (1/self.steepness_coefficient - 1)))
            all_local_windows.append(local_window)
        return tuple(all_local_windows)

    def compress_kv_cache(self, past_key_values, retention_window_start, prefill=False):
        next_retention_window_start = retention_window_start.copy()
        new_past_key_values = []
        if past_key_values is None:
            return None, None
        for layer_idx in range(self.num_transformer_blocks):
            seq_len = past_key_values[layer_idx][0].size(self.kv_seq_dim_idx)
            current_uncompressed_window_length = seq_len - retention_window_start[layer_idx]
            if current_uncompressed_window_length > self.all_local_windows[layer_idx]:
                if prefill:
                    new_keys = self.compress_prefill(past_key_values[layer_idx][0],
                                                     retention_window_start[layer_idx], layer_idx)
                    new_values = self.compress_prefill(past_key_values[layer_idx][1],
                                                       retention_window_start[layer_idx], layer_idx)
                    new_past_key_values.append((new_keys, new_values))
                else:
                    new_keys = self.compress_decode(past_key_values[layer_idx][0])
                    new_values = self.compress_decode(past_key_values[layer_idx][1])
                    new_past_key_values.append((new_keys, new_values))
                next_retention_window_start[layer_idx] = new_keys.size(self.kv_seq_dim_idx)
            else:
                new_past_key_values.append(past_key_values[layer_idx])
        return tuple(new_past_key_values), next_retention_window_start

    def compress_prefill(self, x, retention_window_start, layer_idx):
        x_copy = x.clone()
        seq_len = x_copy.size(self.kv_seq_dim_idx)
        while seq_len - retention_window_start > self.all_local_windows[layer_idx]:
            x_sink = self.slice2d(x_copy, 0, self.sink_tokens)
            x_compress_chunk = self.slice2d(x_copy, self.sink_tokens,
                                            retention_window_start + self.all_local_windows[layer_idx])
            x_future = self.slice2d(x_copy, retention_window_start + self.all_local_windows[layer_idx], seq_len)
            x_compressed = self.compress2d(x_compress_chunk, 0, x_compress_chunk.size(2))
            x_copy = torch.cat((x_sink, x_compressed, x_future), dim=self.kv_seq_dim_idx)
            retention_window_start = self.sink_tokens + x_compressed.size(self.kv_seq_dim_idx)
            seq_len = x_copy.size(self.kv_seq_dim_idx)
        return x_copy

    def compress_decode(self, x):
        x_copy = x.clone()
        seq_len = x_copy.size(self.kv_seq_dim_idx)
        sink_cache = self.slice2d(x_copy, 0, self.sink_tokens)
        compressed_cache = self.compress2d(x_copy, self.sink_tokens, seq_len)
        complete_cache = torch.cat((sink_cache, compressed_cache), dim=self.kv_seq_dim_idx)
        return complete_cache

    @staticmethod
    def slice2d(x, start, end):
        out = x[:, :, start:end, ...].clone()
        return out

    def compress2d(self, x, start, end):
        x_copy = x.clone()
        if (end - start) % 2 != 0:
            x_copy = torch.cat((x_copy, x_copy[:, :, -1:, ...]), dim=2)
        if SequenceCompressionPoolingType.MEAN is self.pooling_type:
            out = (x_copy[:, :, ::2, ...] + x_copy[:, :, 1::2, ...]) / 2
        elif SequenceCompressionPoolingType.MAX is self.pooling_type:
            out = torch.maximum(x_copy[:, :, ::2, ...], x_copy[:, :, 1::2, ...])
        elif SequenceCompressionPoolingType.BEST is self.pooling_type:
            x_first = x_copy[:, :, ::2, ...]
            x_second = x_copy[:, :, 1::2, ...]
            max_indices = x_first > x_second
            first_count = max_indices.sum(dim=2, keepdim=True)
            second_count = (~max_indices).sum(dim=2, keepdim=True)
            out = torch.where(first_count >= second_count, x_first, x_second)
        else:
            raise NotImplementedError(f"Sequence compression type {self.pooling_type} not implemented.")
        return out
