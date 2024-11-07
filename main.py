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
        description='Project for compressing key-value cache in transformers models for efficient inference')

    parser.add_argument('--model_name', type=str, default='gpt2', help='Huggingface model ID to be used')
    parser.add_argument('--decoding_strategy', type=str, default='greedy', help='Decoding strategy to be used')
    parser.add_argument('--dataset_name', type=str, default='abisee/cnn_dailymail', help='Dataset name to be used')
    parser.add_argument('--compression_window', type=int, default=64, help='Window size for key-value cache compression')
    parser.add_argument('--sink_tokens', type=int, default=4, help='Number of sink tokens to consider')
    parser.add_argument('--device', type=str, default='cpu', help='Device to be used for inference')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum length of the output sequence')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')

    return parser.parse_args()


if __name__ == '__main__':
    configure_huggingface()
    args = get_args()
    execute(args.model_name, args.decoding_strategy, args.dataset_name, args.batch_size, args.sink_tokens,
            args.compression_window, args.max_length, args.device)
