class ContextCompressor:
    def __init__(self, tokenizer, max_context_size):
        self.tokenizer = tokenizer
        self.max_context_size = max_context_size

    def compress(self, prompt_template, context, input_text, direction="right"):
        test_prompt = prompt_template.format(context="", input=input_text)
        test_prompt_tokens = self.tokenizer(test_prompt, return_tensors="pt")["input_ids"]
        remaining_tokens = self.max_context_size - test_prompt_tokens.size(1)
        if direction == "left":
            self.tokenizer.truncation_side = "left"
        else:
            self.tokenizer.truncation_side = "right"
        context_tokens = self.tokenizer(context, return_tensors="pt", truncation=True,
                                        max_length=remaining_tokens)["input_ids"]
        truncated_context = self.tokenizer.decode(context_tokens[0], skip_special_tokens=True)

        return truncated_context
