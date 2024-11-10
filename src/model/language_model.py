import torch

from src.enum.decoding_strategy import DecodingStrategy
from src.model.huggingface_model import HuggingFaceModel
from src.compression.sequence_kv_compress import SequenceKVCompressor


class LanguageModel:
    def __init__(self, model_name, sink_tokens, initial_local_window, steepness_coefficient, skip_prefill_compression,
                 sequence_pooling_type, kv_seq_dim=2, device='cpu'):
        model_loader = HuggingFaceModel(model_name)
        self.model = model_loader.get_model()
        self.max_context_size = self.model.config.max_position_embeddings
        self.model = self.model.to(device)
        self.tokenizer = model_loader.get_tokenizer()
        self.tokenizer.padding_side = "left"
        self.sink_tokens = sink_tokens
        self.steepness_coefficient = steepness_coefficient
        self.num_transformer_blocks = self.model.config.num_hidden_layers
        self.sequence_kv_compressor = SequenceKVCompressor(sink_tokens, sequence_pooling_type,
                                                           initial_local_window, steepness_coefficient,
                                                           skip_prefill_compression, self.num_transformer_blocks,
                                                           kv_seq_dim)

    @torch.no_grad()
    def decode(self, input_batch, decoding_strategy, max_length=100, top_k=None, top_p=None):
        self.model.eval()
        tokenized_inputs = self.tokenizer(
            input_batch,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_context_size
        )
        tokenized_inputs = {k: v.to(self.model.device) for k, v in tokenized_inputs.items()}
        input_ids = tokenized_inputs["input_ids"]
        past_key_values = None
        batch_size = input_ids.size(0)
        output_tokens = input_ids
        generated_tokens = torch.zeros(batch_size, 0).to(self.model.device)
        retention_window_start = [self.sink_tokens] * self.num_transformer_blocks
        for decode_step in range(max_length):
            is_prefill = decode_step == 0
            with torch.no_grad():
                if is_prefill:
                    out = self.model(input_ids=output_tokens, past_key_values=past_key_values, use_cache=True)
                else:
                    out = self.model(input_ids=output_tokens[:, -1:], past_key_values=past_key_values, use_cache=True)
            last_token_logit = out.logits[:, -1, :]

            past_key_values, next_retention_window = (
                self.sequence_kv_compressor.compress_kv_cache(out.past_key_values, retention_window_start,
                                                              prefill=is_prefill))
            retention_window_start = next_retention_window
            if decoding_strategy == DecodingStrategy.GREEDY:
                next_tokens = self._greedy_decode(last_token_logit)
            elif decoding_strategy == DecodingStrategy.TOP_K:
                next_tokens = self._top_k_sampling(last_token_logit, top_k)
            elif decoding_strategy == DecodingStrategy.TOP_P:
                next_tokens = self._top_p_sampling(last_token_logit, top_p)
            elif decoding_strategy == DecodingStrategy.RANDOM:
                next_tokens = self._random_sampling(last_token_logit)
            else:
                raise NotImplementedError(f"Decoding strategy {decoding_strategy} not implemented.")
            output_tokens = torch.cat((output_tokens, next_tokens.unsqueeze(1)), dim=1)
            generated_tokens = torch.cat((generated_tokens, next_tokens.unsqueeze(1)), dim=1)
            if torch.all(next_tokens == self.tokenizer.eos_token_id):
                break
        decoded_results = self.tokenizer.batch_decode(generated_tokens.to(torch.int32), skip_special_tokens=True)
        return decoded_results

    @staticmethod
    def _greedy_decode(logit):
        return logit.argmax(dim=-1)

    @staticmethod
    def _top_k_sampling(logit, k):
        top_k_probs, top_k_indices = torch.topk(logit, k, dim=-1)
        top_k_probs = torch.softmax(top_k_probs, dim=-1)
        return top_k_indices[torch.multinomial(top_k_probs, 1).squeeze(-1)]

    @staticmethod
    def _top_p_sampling(logit, p):
        sorted_logit, sorted_indices = torch.sort(logit, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logit, dim=-1), dim=-1)
        top_p_mask = cumulative_probs < p
        top_p_probs = torch.softmax(sorted_logit.masked_fill(~top_p_mask, -float("inf")), dim=-1)
        return sorted_indices[torch.multinomial(top_p_probs, 1).squeeze(-1)]

    @staticmethod
    def _random_sampling(logit):
        probabilities = torch.softmax(logit, dim=-1)
        return torch.multinomial(probabilities, num_samples=1).squeeze(-1)
