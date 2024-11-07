def cnn_dailymail_map(example):
    article = example['article']
    example['input_text'] = f"Summarize the following article:\n{article}\nSummary:"
    example['target'] = example['highlights']
    del example['highlights']
    del example['article']
    return example


def eli5_prompt(example):
    question = example['question']
    example['input_text'] = f"Answer the question: {question}\nAnswer:"
    example['target'] = example['answer']
    del example['question']
    del example['answer']
    return example


class PromptMap:
    def __init__(self):
        self.map = {
            'abisee/cnn_dailymail': cnn_dailymail_map,
            'sentence-transformers/eli5': eli5_prompt
        }

    def get_map(self, dataset):
        return self.map[dataset]
