import pandas as pd
import torch.cuda
from tqdm import tqdm

from src.data.dataset_loader import DatasetLoader
from src.enum.decoding_strategy import DecodingStrategy
from src.enum.sequence_compression import SequenceCompressionPoolingType
from src.model.language_model import LanguageModel
from src.metrics.longbench_scorer import LongBenchEvaluationMetric
from src.util.results_io import save_results

from const import dataset2max_len

def execute(model_name, decoding_strategy, dataset_split, batch_size, sink_tokens,
            initial_local_window, steepness_coefficient, seq_pooling_type, compress_context, mode, prompt):
    if mode == 'user':
        # Hardcoded user prompt
        chat = [
            {"role": "user", "content": prompt}
        ]

        # Apply chat template to get formatted input

        seq_pooling_type = SequenceCompressionPoolingType[seq_pooling_type.upper()]
        model = LanguageModel(
            model_name=model_name,
            sink_tokens=sink_tokens,
            initial_local_window=initial_local_window,
            steepness_coefficient=steepness_coefficient,
            sequence_pooling_type=seq_pooling_type,
        )
        tokenizer = model.tokenizer
        decoding_strategy = DecodingStrategy[decoding_strategy.upper()]

        formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False)



        try:
            # Perform inference on the hardcoded prompt
            response = model.decode(
                input_batch=[formatted_prompt],
                decoding_strategy=decoding_strategy,
                max_length=model.max_context_size
            )

            # Print results
            # print(f"\nPrompt: {user_prompt}")
            print(f"Response: {response[0]}")
        except torch.cuda.OutOfMemoryError:
            print("Out of memory error occurred during inference. Clearing cache.")
            torch.cuda.empty_cache()


    else:
        seq_pooling_type = SequenceCompressionPoolingType[seq_pooling_type.upper()]
        model = LanguageModel(model_name=model_name, sink_tokens=sink_tokens,
                            initial_local_window=initial_local_window,
                            steepness_coefficient=steepness_coefficient,
                            sequence_pooling_type=seq_pooling_type)
        tokenizer = model.tokenizer
        max_context_size = model.max_context_size
        data_loaders = (DatasetLoader(dataset_split, tokenizer, max_context_size, compress_context)
                        .get_loader(batch_size=batch_size))
        decoding_strategy = DecodingStrategy[decoding_strategy.upper()]
        evaluation_metrics = LongBenchEvaluationMetric()
        results_path = (f'{model_name}__{str(initial_local_window)}__{str(steepness_coefficient)}__'
                        f'{seq_pooling_type.name.lower()}__'
                        f'{"context_compress" if compress_context else "no_context_compress"}')
        results_path = results_path.replace('/', '--')
        results = {'dataset': [], 'output': [], 'target': [], 'metric': []}
        total_datasets = len(data_loaders)
        for j, dataset in enumerate(data_loaders):
            print(f'\nRunning inference on {dataset} dataset ({j + 1}/{total_datasets})')
            data_loader = data_loaders[dataset]
            max_length = dataset2max_len[dataset]
            pbar = tqdm(data_loader, total=len(data_loader), desc=f'Generating results')
            for i, batch in enumerate(pbar):
                input_text = batch['input_text']
                all_classes_batch = batch['all_classes']
                dataset_names = batch['dataset']
                try:
                    response = model.decode(input_batch=input_text, decoding_strategy=decoding_strategy,
                                            max_length=max_length)
                    outputs = response
                    labels = batch['target']
                    scores = [evaluation_metrics.get_score(dataset_name, output, label, all_classes)
                            for dataset_name, output, label, all_classes in
                            zip(dataset_names, outputs, labels, all_classes_batch)]
                    results['dataset'].extend(dataset_names)
                    results['output'].extend(outputs)
                    results['target'].extend(labels)
                    results['metric'].extend(scores)
                except torch.cuda.OutOfMemoryError:
                    print('Out of memory error occurred. Skipping this batch.')
                    torch.cuda.empty_cache()
                if i > 10:
                    break
        results_df = pd.DataFrame(results)
        save_results(results_path, results_df)
