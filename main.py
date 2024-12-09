import os
import argparse

from src.util.execute import execute


def configure_huggingface():
    try:
        hf_token = os.getenv('HF_API_KEY')  # Make sure to add HF_API_KEY to environment variables
        # Add it in .bashrc or .zshrc file to access it globally
        os.environ['HF_TOKEN'] = hf_token
    except (TypeError, KeyError):
        print('Not able to set HF token. Please set HF_API_KEY in environment variables.')


def get_args():
    parser = argparse.ArgumentParser(
        description='Project for compressing key-value cache in transformers models for efficient inference.')

    parser.add_argument('--model_name', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct',
                        help='Huggingface model ID to be used.')
    parser.add_argument('--decoding_strategy', type=str, default='greedy', help='Decoding strategy to be used.')
    parser.add_argument('--initial_local_window', type=int, default=512,
                        help='Window size of initial layer\'s key-value cache to retain while compressing.')
    parser.add_argument('--steepness_coefficient', type=float, default=1,
                        help='Steepness coefficient for the pyramid of local window sizes for KV cache of each layer.')
    parser.add_argument('--sink_tokens', type=int, default=4, help='Number of sink tokens to consider.')
    parser.add_argument('--seq_pooling_type', type=str, default='mean', help='Pooling type for sequence compression.')
    parser.add_argument('--compress_context', action='store_true', help='Whether to compress the context.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for inference.')
    parser.add_argument('--dataset_split', type=str, default='test', help='Dataset split to be used.')
    parser.add_argument('--mode', type=str, default='test', help='Mode in which you want to run. Enter user to provide custom input.')
    parser.add_argument('--prompt', type=str, default='What is Machine Learning', help='User Defined Prompt')



    return parser.parse_args()


if __name__ == '__main__':
    configure_huggingface()
    args = get_args()
    execute(args.model_name, args.decoding_strategy, args.dataset_split, args.batch_size, args.sink_tokens,
            args.initial_local_window, args.steepness_coefficient, args.seq_pooling_type, args.compress_context, args.mode, args.prompt)
