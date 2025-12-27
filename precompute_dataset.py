#!/usr/bin/env python3
"""
Precompute dataset samples and save to .npz files in cache dir.

Usage:
  python scripts/precompute_dataset.py --speech_list LIST --noise_list LIST --outdir ./cache --n 100

speech_list and noise_list are plaintext files with one audio path per line, or comma-separated paths.
"""
import argparse
import os
import numpy as np
from pathlib import Path

# Make sure imports work relative to project
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ReZero2.data.dataset import ReZeroOnTheFlyDataset
from multiprocessing import Pool
import functools


def worker_task_top(args_tuple):
    """Top-level worker task for multiprocessing: avoids pickling local functions.
    args_tuple: (speech_files, noise_files, outdir, worker_id, n_per_worker, start_index, total_n)
    """
    speech_files, noise_files, outdir, worker_id, n_per_worker, start_index, total_n = args_tuple
    ds_worker = ReZeroOnTheFlyDataset(speech_files, noise_files, cache_dir=outdir, use_cache=False)
    for j in range(n_per_worker):
        idx = start_index + j
        print(f"Worker {worker_id} generating sample {idx+1}/{total_n}", end='\r')
        sample = ds_worker.generate_sample()
        fn = os.path.join(outdir, f"sample_{idx:06d}.npz")
        np.savez_compressed(fn,
                            mix=sample['mix'],
                            target=sample['target'],
                            Q=sample['Q'],
                            region_azi_low=sample['region']['azi_low'],
                            region_azi_high=sample['region']['azi_high'])


def read_list(s):
    if not s:
        return []
    if os.path.exists(s) and os.path.isfile(s):
        with open(s, 'r') as f:
            return [l.strip() for l in f if l.strip()]
    # otherwise assume comma separated
    return [p.strip() for p in s.split(',') if p.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--speech_list', required=True, help='file with speech paths (one per line) or comma-separated list')
    p.add_argument('--noise_list', default='', help='file with noise paths or comma-separated list')
    p.add_argument('--outdir', required=True, help='cache directory to store npz files')
    p.add_argument('--n', type=int, default=2000, help='number of samples to precompute')
    p.add_argument('--workers', type=int, default=1, help='number of parallel worker processes')
    args = p.parse_args()

    speech_files = read_list(args.speech_list)
    noise_files = read_list(args.noise_list)

    os.makedirs(args.outdir, exist_ok=True)

    # Find existing files and start index after them
    existing = sorted([f for f in os.listdir(args.outdir) if f.endswith('.npz')])
    start_idx = len(existing)
    print(f"Starting precompute: found {start_idx} existing files, will generate up to {args.n} total.")

    def worker_task(worker_id, n_per_worker, start_index):
        # Each worker creates its own dataset instance to avoid pickling issues
        ds_worker = ReZeroOnTheFlyDataset(speech_files, noise_files, cache_dir=args.outdir, use_cache=False)
        for j in range(n_per_worker):
            idx = start_index + j
            print(f"Worker {worker_id} generating sample {idx+1}/{args.n}", end='\r')
            sample = ds_worker.generate_sample()
            fn = os.path.join(args.outdir, f"sample_{idx:06d}.npz")
            np.savez_compressed(fn,
                                mix=sample['mix'],
                                target=sample['target'],
                                Q=sample['Q'],
                                region_azi_low=sample['region']['azi_low'],
                                region_azi_high=sample['region']['azi_high'])

    if args.workers <= 1:
        # single-threaded
        for i in range(start_idx, args.n):
            print(f"Generating sample {i+1}/{args.n}", end='\r')
            ds = ReZeroOnTheFlyDataset(speech_files, noise_files, cache_dir=args.outdir, use_cache=False)
            sample = ds.generate_sample()
            fn = os.path.join(args.outdir, f"sample_{i:06d}.npz")
            np.savez_compressed(fn,
                                mix=sample['mix'],
                                target=sample['target'],
                                Q=sample['Q'],
                                region_azi_low=sample['region']['azi_low'],
                                region_azi_high=sample['region']['azi_high'])
    else:
        # parallel execution: split work evenly
        total = args.n - start_idx
        per_worker = total // args.workers
        extras = total % args.workers
        tasks = []
        cur = start_idx
        for w in range(args.workers):
            n_w = per_worker + (1 if w < extras else 0)
            if n_w <= 0:
                continue
            tasks.append((w, n_w, cur))
            cur += n_w

        # build top-level task tuples for worker_task_top
        top_tasks = []
        for (w, n_w, start_idx_w) in tasks:
            top_tasks.append((speech_files, noise_files, args.outdir, w, n_w, start_idx_w, args.n))

        with Pool(processes=args.workers) as pool:
            pool.map(worker_task_top, top_tasks)

    print('\nPrecompute finished.')


if __name__ == '__main__':
    main()
