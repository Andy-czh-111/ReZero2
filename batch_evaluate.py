#!/usr/bin/env python3
"""
Batch evaluation script for ReZero2.

Features:
- Use `ReZeroOnTheFlyDataset` if available (requires `pyroomacoustics`) or synthetic samples as fallback
- Run batched inference on device
- Use DataLoader with `num_workers` to parallelize sample generation
- Compute metrics (uses `evaluate.compute_metrics` which has fallbacks)
- Save metrics to CSV and optionally save audio files

Example:
  python scripts/batch_evaluate.py --checkpoint /path/to/ckpt.pth --num-samples 50 --batch-size 4 --output_dir ./results_eval

"""
import os
import argparse
import time
import csv
from typing import Optional

import torch
import numpy as np

# Ensure repository root is on sys.path so imports like `model` and `utils` resolve
import sys
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.insert(0, root)

from model.system import ReZeroSystem
from utils.istft import ISTFT


class SyntheticDataset(torch.utils.data.Dataset):
    """Generate synthetic mixtures and targets for smoke testing."""
    def __init__(self, num_samples=100, duration_s=1.0, fs=16000, ch=8):
        self.num_samples = num_samples
        self.duration = int(duration_s * fs)
        self.fs = fs
        self.ch = ch

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        T = self.duration
        mix = np.random.randn(T, self.ch).astype(np.float32)
        # target: random low-energy signal
        target = (np.random.randn(T) * 0.05).astype(np.float32)
        # region params as numpy scalars
        region = {
            'azi_low': np.array(-0.5, dtype=np.float32),
            'azi_high': np.array(0.5, dtype=np.float32),
            'ele_low': np.array(0.0, dtype=np.float32),
            'ele_high': np.array(1.57, dtype=np.float32),
            'dist_low': np.array(0.0, dtype=np.float32),
            'dist_high': np.array(2.0, dtype=np.float32)
        }
        # Q: random 0 or 1
        Q = np.random.choice([0, 1], p=[0.2, 0.8])
        return {'mix': torch.from_numpy(mix), 'target': torch.from_numpy(target), 'region': region, 'Q': Q}


def collate_fn(batch):
    # batch: list of dicts
    mix = torch.stack([b['mix'] for b in batch], dim=0)  # (B, T, C)
    target = torch.stack([b['target'] for b in batch], dim=0)
    Q = torch.tensor([b['Q'] for b in batch])
    region = {}
    for k in batch[0]['region']:
        region[k] = torch.stack([torch.tensor(b['region'][k]) for b in batch])
    return mix, target, region, Q


def load_model_device(device: torch.device, checkpoint: Optional[str] = None):
    # mic locations (8-mic circular array)
    R = 0.025
    MIC_LOCS = torch.from_numpy(np.array([[R * np.cos(2 * np.pi * i / 8), R * np.sin(2 * np.pi * i / 8), 0.0] for i in range(8)])).float()
    model = ReZeroSystem(mic_locations=MIC_LOCS, bsrnn_channels=48, P_dim=16, task_type='angle')
    if checkpoint is not None and os.path.exists(checkpoint):
        ck = torch.load(checkpoint, map_location='cpu')
        state = ck.get('model_state_dict', ck)
        # strip module.
        new_state = {k.replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(new_state)
    model.to(device)
    model.eval()
    return model


def run_batch_evaluate(args):
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print('Device:', device)

    # dataset selection
    if args.use_dataset:
        try:
            from data.dataset import ReZeroOnTheFlyDataset
            dataset = ReZeroOnTheFlyDataset(speech_list=args.speech_list or [], noise_list=args.noise_list or [], fs=16000)
        except Exception as e:
            print('Could not import ReZeroOnTheFlyDataset (pyroomacoustics missing?), falling back to synthetic dataset:', e)
            dataset = SyntheticDataset(num_samples=args.num_samples)
    else:
        dataset = SyntheticDataset(num_samples=args.num_samples)

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    model = load_model_device(device, args.checkpoint)
    istft = ISTFT(win_len=512, win_shift_ratio=0.25, nfft=512).to(device)

    # metrics function from evaluate (it has fallback implementations)
    try:
        from evaluate import compute_metrics
    except Exception:
        # fallback to local wrapper using evaluate's fallback
        from evaluate import compute_metrics

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, args.output_csv)
    fieldnames = ['sample_id', 'Q', 'sdr', 'stoi', 'pesq', 'decay', 'energy_out_db', 'time_s']

    writer = csv.DictWriter(open(csv_path, 'w', newline=''), fieldnames=fieldnames)
    writer.writeheader()

    sample_id = 0
    for mix, target, region, Q in loader:
        b = mix.size(0)
        # move tensors
        mix_t = mix.to(device)
        for k in region:
            region[k] = region[k].to(device)
        # forward
        t0 = time.time()
        with torch.no_grad():
            est_stft = model(mix_t, region)
            est_wav = istft(est_stft, length=mix_t.shape[1])
        t1 = time.time()

        # loop samples in batch
        for i in range(b):
            mix_np = mix[i, :, 0].cpu().numpy()
            est_np = est_wav[i].cpu().numpy()
            target_np = target[i].cpu().numpy()
            q = int(Q[i].item()) if isinstance(Q, torch.Tensor) else int(Q[i])
            metrics = compute_metrics(est_np, target_np, mix_np, q)

            row = {
                'sample_id': sample_id,
                'Q': q,
                'sdr': metrics.get('sdr', float('nan')),
                'stoi': metrics.get('stoi', float('nan')),
                'pesq': metrics.get('pesq', float('nan')),
                'decay': metrics.get('decay', float('nan')),
                'energy_out_db': metrics.get('energy_out_db', float('nan')),
                'time_s': (t1 - t0) / float(b)
            }
            writer.writerow(row)

            # optionally save audio
            if args.save_audio:
                import soundfile as sf
                prefix = f"sample{sample_id:06d}_Q{q}"
                sf.write(os.path.join(args.output_dir, prefix + '_mix.wav'), mix_np, 16000)
                sf.write(os.path.join(args.output_dir, prefix + '_est.wav'), est_np, 16000)
                sf.write(os.path.join(args.output_dir, prefix + '_ref.wav'), target_np, 16000)

            sample_id += 1
            if sample_id >= args.num_samples:
                break

        if sample_id >= args.num_samples:
            break

    print('Evaluation finished. CSV saved to', csv_path)


def _parse_args():
    p = argparse.ArgumentParser(description='Batch evaluate ReZero2')
    p.add_argument('--checkpoint', type=str, default=None)
    p.add_argument('--num-samples', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--use_dataset', action='store_true')
    p.add_argument('--speech_list', type=str, nargs='*', default=None)
    p.add_argument('--noise_list', type=str, nargs='*', default=None)
    p.add_argument('--output-dir', type=str, default='./results_eval')
    p.add_argument('--output-csv', type=str, default='metrics.csv')
    p.add_argument('--save-audio', action='store_true')
    p.add_argument('--device', type=str, default=None)
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    # Normalize arg names for code compatibility
    args.output_dir = args.output_dir
    args.output_csv = args.output_csv
    args.num_samples = args.num_samples
    args.checkpoint = args.checkpoint
    run_batch_evaluate(args)
