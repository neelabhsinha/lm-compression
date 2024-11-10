import os

from const import results_dir


def save_results(results_folder, df):
    full_results_dir = os.path.join(results_dir, results_folder)
    os.makedirs(full_results_dir, exist_ok=True)
    df.to_csv(os.path.join(full_results_dir, 'results.csv'), index=False)
    summary_df = df.groupby('dataset')['metric'].agg(
        count='count',
        mean='mean',
        std='std',
        min='min',
        q25=lambda x: x.quantile(0.25),
        median='median',
        q75=lambda x: x.quantile(0.75),
        max='max'
    )
    overall_stats = df['metric'].agg(
        count='count',
        mean='mean',
        std='std',
        min='min',
        q25=lambda x: x.quantile(0.25),
        median='median',
        q75=lambda x: x.quantile(0.75),
        max='max'
    )
    summary_df.loc['overall'] = overall_stats
    summary_df.to_csv(os.path.join(full_results_dir, 'summary.csv'))
    print(f'Results saved to {full_results_dir}')
