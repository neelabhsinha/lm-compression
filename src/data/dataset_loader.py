import os

from datasets import load_dataset
from const import cache_dir
from torch.utils.data import DataLoader

from src.data.prompt import PromptMap


class DatasetLoader:
    def __init__(self, dataset_name, dataset_split):
        self.dataset_name = dataset_name
        self.dataset = None
        self.dataset_split = dataset_split
        prompt_mapper = PromptMap()
        self.map = prompt_mapper.get_map(dataset_name)
        os.makedirs(cache_dir, exist_ok=True)

    def load(self):
        if 'cnn_dailymail' in self.dataset_name:
            dataset = load_dataset(self.dataset_name, '3.0.0', split=self.dataset_split, cache_dir=cache_dir)
        else:
            dataset = load_dataset(self.dataset_name, split=self.dataset_split, cache_dir=cache_dir)
        dataset = dataset.map(self.map)
        self.dataset = dataset

    def get_loader(self, batch_size=64):
        if self.dataset is None:
            self.load()
        shuffle = False
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader
