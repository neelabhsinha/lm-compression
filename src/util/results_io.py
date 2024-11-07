import os

from const import results_dir


def save_results(results_folder, df):
    full_results_dir = os.path.join(results_dir, results_folder)
    os.makedirs(full_results_dir, exist_ok=True)
    df.to_csv(os.path.join(full_results_dir, 'results.csv'), index=False)
    summary = df.describe()
    summary.to_csv(os.path.join(full_results_dir, 'results_summary.csv'), index=True)
    print(f'Results saved to {full_results_dir}')
