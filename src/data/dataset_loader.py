import os

from datasets import load_dataset
from const import cache_dir
from torch.utils.data import DataLoader

from src.data.prompt import PromptMap


def collate_fn(batch):
    batch_dict = {key: [d[key] for d in batch] for key in batch[0]}
    return batch_dict


class DatasetLoader:
    def __init__(self, dataset_split, tokenizer, max_context_size, compress_context=False):
        self.dataset_names = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
                    "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum",
                    "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

        self.data_loaders = None
        self.dataset_split = dataset_split
        self.prompt_mapper = PromptMap(tokenizer, compress_context, max_context_size)
        os.makedirs(cache_dir, exist_ok=True)

    def load(self, dataset_name=None):
        dataset = load_dataset('THUDM/LongBench', dataset_name, split=self.dataset_split, cache_dir=cache_dir)
        mapper = self.prompt_mapper.get_prompt_function(dataset_name)
        dataset = dataset.map(mapper)
        return dataset

    def get_loader(self, batch_size=64):
        data_loaders = {}
        shuffle = False
        for dataset_name in self.dataset_names:
            dataset = self.load(dataset_name)
            data_loaders[dataset_name] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
        return data_loaders
