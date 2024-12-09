
# Memory Reduction for LLM Inference via KV-Cache Compression

This project provides an efficient method for compressing the key-value cache in transformer models to optimize memory usage and speed during inference. The key-value cache compression framework allows for more efficient generation by dynamically reducing the cache size while retaining critical context, making it ideal for deployment in memory-constrained environments.

## Table of Contents
- [Key-Value Cache Compression for Transformers](#key-value-cache-compression-for-transformers)
  - [Features](#features)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
  - [Usage](#usage)
  - [Arguments](#arguments)
  - [Example Usage](#example-usage)
  - [Project Structure](#project-structure)
  - [File Descriptions](#file-descriptions)

## Features
- Compresses the key-value (KV) cache in transformers to enable faster and memory-efficient inference.
- Supports various pooling types to control compression: **mean**, **max**, and **best**.
- Offers flexible control over initial window sizes, steepness of local window changes, and sink tokens.
- Compatible with any Hugging Face transformer model for quick experimentation with different models.
- Provides support for multiple decoding strategies (e.g., greedy, beam search).

## Requirements
- Python 3.7+
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- torch

Detailed requirements can be found in the `requirements.txt` file.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/neelabhsinha/lm-compression.git
   cd lm-compression
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Environment Variables
Set your Hugging Face API token in your environment to enable access to models:
- Add the token to your shell configuration file (e.g., `.bashrc` or `.zshrc`):
  ```bash
  export HF_API_KEY="your_huggingface_api_token"
  ```

## Usage
Run the script with the desired configuration for model compression and inference:
```bash
python main.py --model_name <model_name> --decoding_strategy <strategy> --initial_local_window <size> ...
```

## Arguments
The following arguments can be passed to the script:

| Argument               | Type    | Default                          | Description                                                                                       |
|------------------------|---------|----------------------------------|---------------------------------------------------------------------------------------------------|
| `--model_name`         | `str`   | `meta-llama/Llama-3-8B-Instruct` | Hugging Face model ID to be used.                                                                 |
| `--decoding_strategy`  | `str`   | `greedy`                         | Decoding strategy for inference. Options: `greedy`, `beam_search`, etc.                           |
| `--initial_local_window`| `int`  | `512`                            | Initial window size of the key-value cache for the first transformer layer.                       |
| `--steepness_coefficient`| `float`| `1.0`                            | Controls the steepness of the local window size decrease across layers.                           |
| `--sink_tokens`        | `int`   | `4`                              | Number of tokens to retain without compression in each layer.                                     |
| `--skip_prefill_compression` | `flag` | -                                | Skips cache compression during the prefill stage if set.                                          |
| `--seq_pooling_type`   | `str`   | `mean`                           | Pooling type for sequence compression. Options: `mean`, `max`, `best`.                            |
| `--compress_context`   | `flag`  | -                                | Enables context compression.                                                                      |
| `--device`             | `str`   | `cpu`                            | Device to be used for inference. Options: `cpu`, `cuda`.                                         |
| `--max_length`         | `int`   | `128`                            | Maximum output sequence length during generation.                                                 |
| `--batch_size`         | `int`   | `4`                              | Batch size for inference.                                                                         |
| `--dataset_split`      | `str`   | `test`                           | Dataset split to be used. Options: `train`, `validation`, `test`.                                 |

## Example Usage
To run the project with a Llama-3-8B-Instruct model, a greedy decoding strategy, and a specific configuration for local window compression:

```bash
python main.py \
    --model_name "meta-llama/Llama-3-8B-Instruct" \
    --decoding_strategy "greedy" \
    --initial_local_window 512 \
    --steepness_coefficient 1.5 \
    --sink_tokens 4 \
    --skip_prefill_compression \
    --seq_pooling_type "mean" \
    --compress_context \
    --device "cuda" \
    --max_length 64 \
    --batch_size 8 \
    --dataset_split "test"
```

After execution, results will be stored in `results` folder.

## Project Structure
```
.
├── cache_dir/                      # Directory for cached data (e.g., model checkpoints) - created during execution.
├── results/                        # Directory for storing results of the experiments (created during execution).
├── src/                            # Source code directory
│   ├── compression/                # Contains code for context and sequence compression.
│   │   ├── context_compress.py     # Contains logic for compressing prompt context.
│   │   └── sequence_kv_compress.py # Contains logic for compressing key-value cache along sequence length for different layers.
│   ├── data/                       # Contains utilities for dataset loading and processing.
│   │   ├── dataset_loader.py       # Contains functions for loading and preparing datasets for model inference.
│   │   └── prompt.py               # Manages prompts and input sequences for model inference.
│   ├── enum/                       # Enums for defining constants such as decoding strategies.
│   │   ├── decoding_strategy.py    # Defines different decoding strategies.
│   │   └── sequence_compression.py # Defines various compression types for KV cache compression.
│   ├── metrics/                    # Contains modules for evaluation metrics.
│   │   ├── longbench_scorer.py     # Longbench performance score generator.
│   │   └── metrics.py              # General metrics for model evaluation used by longbench.
│   ├── model/                      # Defines model utilities.
│   │   ├── huggingface_model.py    # Sets up and configures Hugging Face transformer models.
│   │   └── language_model.py       # Decoding logic with cache-compression support.
│   ├── util/                       # Utilities for execution and result handling.
│   │   ├── execute.py              # Core logic for executing inferences on datasets.
│   │   └── results_io.py           # Handles input/output operations for experiment results.
├── const.py                        # Defines constants used across the project.
├── main.py                         # Main script for configuring and running the project.
└── requirements.txt                # Lists project dependencies.
```

## File Descriptions

### Main Scripts
- **main.py**: The main entry point of the project, which parses command-line arguments and executes the KV cache compression with the specified parameters.

### Modules
- **compression/context_compress.py**: Contains functions for compressing the context information during inference, focusing on reducing the size of the input context.
- **compression/sequence_kv_compress.py**: Implements key-value (KV) cache compression methods, which reduce the cache size while retaining essential information across transformer layers.

- **data/dataset_loader.py**: Loads datasets for the project and prepares them for model inference.
- **data/prompt.py**: Manages prompts and input sequences, which can be passed to the model for inference and evaluation.

- **enum/decoding_strategy.py**: Defines different decoding strategies (e.g., greedy, beam search) as constants or enumerations.
- **enum/sequence_compression.py**: Defines various compression types (e.g., mean, max, best) for use during KV cache compression.

- **metrics/longbench_scorer.py**: Contains custom scoring functions, potentially aligned with the LongBench benchmark, to evaluate model performance over long sequences.
- **metrics/metrics.py**: Provides general metrics used for model evaluation, such as accuracy, F1, or BLEU scores.

- **model/huggingface_model.py**: Sets up and configures Hugging Face transformer models, providing functions to load and initialize models.
- **model/language_model.py**: Implements high-level functions for interacting with language models, making it easier to integrate Hugging Face models with the rest of the project.

- **util/execute.py**: Contains the core logic for executing the main compression task, using the parameters specified by the user.
- **util/results_io.py**: Handles input/output operations for experiment results, such as saving and loading compressed output or evaluation metrics.

### Other Files
- **const.py**: Contains global constants used throughout the project, such as default configuration values or constants used for logging.
- **requirements.txt**: Lists the necessary dependencies for running the project, including libraries such as `torch` and `transformers`.

