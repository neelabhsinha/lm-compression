import os

from datasets import load_dataset
from const import cache_dir
from torch.utils.data import DataLoader

from src.data.prompt import PromptMap


class DatasetLoader:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.dataset = None
        prompt_mapper = PromptMap()
        self.map = prompt_mapper.get_map(dataset_name)
        os.makedirs(cache_dir, exist_ok=True)

    def load(self, split='test'):
        dataset = load_dataset(self.dataset_name, '3.0.0', split=split, cache_dir=cache_dir)
        dataset = dataset.map(self.map)
        self.dataset = dataset

    def get_loader(self, split='test', batch_size=64):
        if self.dataset is None:
            self.load(split)
        shuffle = split == 'train'
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader
