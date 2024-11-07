import torch

from src.enum.decoding_strategy import DecodingStrategy
from src.model.huggingface_model import HuggingFaceModel
from src.util.kv_compressor import compress_kv_cache


class LanguageModel:
    def __init__(self, model_name, sink_tokens, compression_window, device='cpu'):
        model_loader = HuggingFaceModel(model_name)
        self.model = model_loader.get_model()
        self.model = self.model.to(device)
        self.tokenizer = model_loader.get_tokenizer()
        self.sink_tokens = sink_tokens
        self.compression_window = compression_window

    @torch.no_grad()
    def decode(self, input_batch, decoding_strategy, max_length=100, top_k=None, top_p=None):
        self.model.eval()
        tokenized_inputs = self.tokenizer(input_batch, return_tensors="pt", padding=True, truncation=True)
        tokenized_inputs = {k: v.to(self.model.device) for k, v in tokenized_inputs.items()}
        input_ids = tokenized_inputs["input_ids"]
        past_key_values = None
        batch_size = input_ids.size(0)
        output_tokens = input_ids
        generated_tokens = torch.zeros(batch_size, 0).to(self.model.device)
        for decode_step in range(max_length):
            is_prefill = decode_step == 0
            # print('Starting decode step:', decode_step)
            # print('Token shape - ', output_tokens.shape)
            with torch.no_grad():
                if is_prefill:
                    out = self.model(input_ids=output_tokens, past_key_values=past_key_values, use_cache=True)
                else:
                    out = self.model(input_ids=output_tokens[:, -1:], past_key_values=past_key_values, use_cache=True)
            last_token_logit = out.logits[:, -1, :]

            past_key_values = compress_kv_cache(out.past_key_values, self.sink_tokens, self.compression_window,
                                                prefill=is_prefill)
            # print('Cache shape after compression - ', past_key_values[0][0].shape)
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
