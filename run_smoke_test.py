#!/usr/bin/env python3
import sys, os
sys.path.insert(0, '/Project/nerual_beamforming/myproject')
sys.path.insert(0, '/Project/nerual_beamforming/myproject/ReZero2')
import torch
import numpy as np
from ReZero2.model.system import ReZeroSystem
from ReZero2.utils.istft import ISTFT
from datetime import datetime

def main():
    R=0.025
    MIC_LOCS = np.array([[R*np.cos(2*np.pi*i/8), R*np.sin(2*np.pi*i/8), 0.0] for i in range(8)])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device', device)
    model = ReZeroSystem(mic_locations=torch.from_numpy(MIC_LOCS).float(), bsrnn_channels=32, P_dim=8, task_type='angle').to(device)
    B,Ch,T = 2,8,800
    mix = torch.randn(B,Ch,T, device=device)
    region_params = {
     'azi_low': torch.zeros(B, device=device),
     'azi_high': torch.ones(B, device=device)*0.1,
     'ele_low': torch.zeros(B, device=device),
     'ele_high': torch.ones(B, device=device)*0.05,
     'dist_low': torch.ones(B, device=device)*0.5,
     'dist_high': torch.ones(B, device=device)*1.0
    }
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    est_stft = model(mix, region_params)
    istft = ISTFT(win_len=512, win_shift_ratio=0.25, nfft=512).to(device)
    try:
        est_wav = istft(est_stft, length=mix.shape[2])
    except Exception:
        est_wav = istft(est_stft)
    loss = float(torch.abs(est_wav).mean().detach().cpu().numpy())
    out_dir = os.path.join(os.getcwd(), 'ReZero2', 'results')
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, 'smoke_test_log.txt')
    with open(log_path, 'a') as f:
        f.write(f"{datetime.now().isoformat()} Smoke-test loss (mean abs of est_wav): {loss}\n")
    print('WROTE', log_path)
    print('Loss', loss)

if __name__ == '__main__':
    main()
