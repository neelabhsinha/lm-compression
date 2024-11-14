import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from const import cache_dir


class HuggingFaceModel:
    def __init__(self, model_name, for_eval=True):
        if 'local' in model_name:
            model_name.replace('/', '--')
        self.model_name = model_name
        self.nf4_config = BitsAndBytesConfig( # config for 4-bit quantization - can be considered later
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.for_eval = for_eval
        os.makedirs(cache_dir, exist_ok=True)

    def get_model(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, cache_dir=cache_dir, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
        print(f'Loaded model {self.model_name} on device {model.device}')

        return model

    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=cache_dir, padding_side='left', trust_remote_code=True)
        print(f'Loaded tokenizer {self.model_name}')
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Padding token was not set, setting EOS token as padding token.")
        return tokenizer
