def cnn_dailymail_map(example):
    article = example['article']
    example['input_text'] = f"Summarize the following article:\n{article}\nSummary:"
    example['target'] = example['highlights']
    del example['highlights']
    del example['article']
    return example


class PromptMap:
    def __init__(self):
        self.map = {
            'abisee/cnn_dailymail': cnn_dailymail_map
        }

    def get_map(self, dataset):
        return self.map[dataset]
