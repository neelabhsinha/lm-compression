import pandas as pd
import torch.cuda
from tqdm import tqdm

from src.data.dataset_loader import DatasetLoader
from src.enum.decoding_strategy import DecodingStrategy
from src.model.language_model import LanguageModel
from src.metrics.rouge import RougeScore
from src.util.results_io import save_results


def execute(model_name, decoding_strategy, dataset_name, dataset_split, batch_size, sink_tokens, retention_window_length,
            skip_prefill_compression, max_length, device):
    model = LanguageModel(model_name=model_name, sink_tokens=sink_tokens,
                          retention_window_length=retention_window_length,
                          skip_prefill_compression=skip_prefill_compression,
                          device=device)
    data_loader = DatasetLoader(dataset_name, dataset_split).get_loader(batch_size=batch_size)
    decoding_strategy = DecodingStrategy[decoding_strategy.upper()]
    rouge_scorer = RougeScore()
    results_path = f'{model_name}_{dataset_name}_{decoding_strategy.name.lower()}_{str(sink_tokens)}_{str(retention_window_length)}_{str(skip_prefill_compression)}'
    results_path = results_path.replace('/', '--')
    results = {'input': [], 'output': [], 'target': [], 'rouge1': [], 'rouge2': [], 'rougeL': []}
    pbar = tqdm(data_loader, total=len(data_loader), desc=f'Generating results')
    for i, batch in enumerate(pbar):
        input_text = batch['input_text']
        try:
            response = model.decode(input_batch=input_text, decoding_strategy=decoding_strategy, max_length=max_length)
            outputs = response
            labels = batch['target']
            rouge = rouge_scorer.get_score(outputs, labels)
            results['input'].extend(input_text)
            results['output'].extend(outputs)
            results['target'].extend(labels)
            results['rouge1'].extend(rouge['rouge1'])
            results['rouge2'].extend(rouge['rouge2'])
            results['rougeL'].extend(rouge['rougeL'])
        except RuntimeError or torch.cuda.OutOfMemoryError:
            print('Out of memory error occurred. Skipping this batch.')
        if i > 1:
            break
    results_df = pd.DataFrame(results)
    save_results(results_path, results_df)
