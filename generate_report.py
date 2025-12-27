#!/usr/bin/env python3
"""
Generate summary statistics and histograms from evaluation CSV.

Usage:
  python scripts/generate_report.py --csv ./results_eval_real/metrics.csv --outdir ./results_eval_real/report

Outputs:
  - summary_overall.csv  : overall metrics mean/median/std/count
  - summary_by_Q.csv     : same stats grouped by Q
  - hist_<metric>.png    : histograms for each metric
"""
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def summarize(df, metrics):
    stats = []
    for m in metrics:
        col = df[m].dropna()
        if len(col) == 0:
            stats.append({'metric': m, 'mean': np.nan, 'median': np.nan, 'std': np.nan, 'count': 0})
        else:
            stats.append({'metric': m, 'mean': float(col.mean()), 'median': float(col.median()), 'std': float(col.std()), 'count': int(len(col))})
    return pd.DataFrame(stats)


def plot_hist(df, metric, outpath, bins=40):
    data = df[metric].dropna()
    plt.figure(figsize=(6,4))
    if len(data) == 0:
        plt.text(0.5, 0.5, 'No data', ha='center', va='center')
    else:
        plt.hist(data, bins=bins, color='#007acc', alpha=0.8)
        plt.xlabel(metric)
        plt.ylabel('Count')
        plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True, help='Input metrics CSV')
    p.add_argument('--outdir', required=True, help='Output directory for report')
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)

    # Metrics to consider
    metrics = ['sdr', 'stoi', 'pesq', 'decay', 'energy_out_db']

    # Overall summary
    overall = summarize(df, metrics)
    overall.to_csv(os.path.join(args.outdir, 'summary_overall.csv'), index=False)

    # Summary by Q
    by_q = []
    for q, g in df.groupby('Q'):
        stats = summarize(g, metrics)
        stats.insert(0, 'Q', q)
        by_q.append(stats.assign(Q=q))
    if len(by_q) > 0:
        by_q_df = pd.concat(by_q, ignore_index=True)
        by_q_df.to_csv(os.path.join(args.outdir, 'summary_by_Q.csv'), index=False)

    # Histograms
    for m in metrics:
        outpath = os.path.join(args.outdir, f'hist_{m}.png')
        plot_hist(df, m, outpath)

    # Quick printed summary
    print('Report generated in', args.outdir)
    print('\nOverall stats:')
    print(overall)


if __name__ == '__main__':
    main()
