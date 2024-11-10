import pandas as pd
import torch.cuda
from tqdm import tqdm

from src.data.dataset_loader import DatasetLoader
from src.enum.decoding_strategy import DecodingStrategy
from src.enum.sequence_compression import SequenceCompressionPoolingType
from src.model.language_model import LanguageModel
from src.metrics.longbench_scorer import LongBenchEvaluationMetric
from src.util.results_io import save_results


def execute(model_name, decoding_strategy, dataset_split, batch_size, sink_tokens,
            retention_window_length,
            skip_prefill_compression, seq_pooling_type, max_length, device):
    seq_pooling_type = SequenceCompressionPoolingType[seq_pooling_type.upper()]
    model = LanguageModel(model_name=model_name, sink_tokens=sink_tokens,
                          retention_window_length=retention_window_length,
                          skip_prefill_compression=skip_prefill_compression,
                          sequence_pooling_type=seq_pooling_type,
                          device=device)
    data_loaders = DatasetLoader(dataset_split).get_loader(batch_size=batch_size)
    decoding_strategy = DecodingStrategy[decoding_strategy.upper()]
    evaluation_metrics = LongBenchEvaluationMetric()
    results_path = f'{model_name}_{decoding_strategy.name.lower()}_{str(sink_tokens)}_{str(retention_window_length)}_{str(skip_prefill_compression)}_{seq_pooling_type.name.lower()}'
    results_path = results_path.replace('/', '--')
    results = {'dataset': [], 'input': [], 'output': [], 'target': [], 'metric': []}
    total_datasets = len(data_loaders)
    for j, dataset in enumerate(data_loaders):
        print(f'\nRunning inference on {dataset} dataset ({j + 1}/{total_datasets})')
        data_loader = data_loaders[dataset]
        pbar = tqdm(data_loader, total=len(data_loader), desc=f'Generating results')
        for i, batch in enumerate(pbar):
            input_text = batch['input_text']
            # input_text = ['Hi. How are you?', 'What is the capital of France?']
            all_classes_batch = batch['all_classes']
            dataset_names = batch['dataset']
            try:
                response = model.decode(input_batch=input_text, decoding_strategy=decoding_strategy, max_length=max_length)
                outputs = response
                labels = batch['target']
                scores = [evaluation_metrics.get_score(dataset_name, output, label, all_classes) for dataset_name, output, label, all_classes in zip(dataset_names, outputs, labels, all_classes_batch)]
                results['dataset'].extend(dataset_names)
                results['input'].extend(input_text)
                results['output'].extend(outputs)
                results['target'].extend(labels)
                results['metric'].extend(scores)
            except RuntimeError or torch.cuda.OutOfMemoryError:
                print('Out of memory error occurred. Skipping this batch.')
            if i > 1:
                break
    results_df = pd.DataFrame(results)
    save_results(results_path, results_df)
