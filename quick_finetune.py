import os
import sys
import torch
# Ensure ReZero2 package dir is on PYTHONPATH so imports like `from model.system` resolve
REZERO2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REZERO2_DIR not in sys.path:
    sys.path.insert(0, REZERO2_DIR)
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
from model.system import ReZeroSystem
try:
    from data.dataset import ReZeroOnTheFlyDataset
except Exception:
    ReZeroOnTheFlyDataset = None
    print('⚠️ pyroomacoustics unavailable or dataset import failed; using DummyDataset fallback for quick finetune')

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, length=256, fs=16000, duration=4.0):
        self.length = length
        self.fs = fs
        self.duration = int(duration * fs)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # mix: (duration, 8), target: (duration,)
        mix = np.random.randn(self.duration, 8).astype(np.float32) * 0.01
        target = np.random.randn(self.duration).astype(np.float32) * 0.005
        region = {
            'azi_low': np.float32(-0.5), 'azi_high': np.float32(0.5),
            'ele_low': np.float32(0.0), 'ele_high': np.float32(np.pi/2),
            'dist_low': np.float32(0.0), 'dist_high': np.float32(2.0)
        }
        Q = np.int32(1)
        return {'mix': mix, 'target': target, 'region': region, 'Q': Q}
from utils.istft import ISTFT
from utils.loss import ReZeroPaperLoss

# Quick fine-tune config
NUM_EPOCHS = 5
BATCH_SIZE = 8
CHANNELS = 32
P_DIM = 8
LR = 1e-4
NUM_SAMPLES = 200  # how many generated samples to use per epoch


def collate_fn(batch):
    mix_list, target_list, Q_list = [], [], []
    region_batch = {k: [] for k in batch[0]['region'].keys()}
    for item in batch:
        mix = item['mix']
        target = item['target']
        # convert numpy arrays to tensors when DummyDataset is used
        if isinstance(mix, (np.ndarray,)):
            mix = torch.from_numpy(mix)
        if isinstance(target, (np.ndarray,)):
            target = torch.from_numpy(target)
        mix_list.append(mix)
        target_list.append(target)
        Q_list.append(item['Q'])
        for k, v in item['region'].items():
            val = v
            if not torch.is_tensor(val):
                val = torch.tensor(val)
            region_batch[k].append(val)
    mix = torch.stack(mix_list)
    target = torch.stack(target_list)
    Q = torch.tensor(Q_list)
    for k in region_batch:
        region_batch[k] = torch.stack(region_batch[k])
    return mix, target, region_batch, Q


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # small dataset
    speech_list = []
    noise_list = []
    # try to populate from repo paths if exist
    base_speech = '/Project/Separation/Data/LibriSpeech/train-clean-360'
    base_noise = '/Project/Separation/Data/Musan'
    import glob
    if os.path.exists(base_speech):
        speech_list = glob.glob(base_speech + '/**/*.flac', recursive=True)[:50]
    if os.path.exists(base_noise):
        noise_list = glob.glob(base_noise + '/**/*.wav', recursive=True)[:50]

    if len(speech_list) == 0:
        print('No speech files found, using dummy random data.')
        # generate dummy data via dataset helper
        # ReZeroOnTheFlyDataset can fallback to random signals
        speech_list = []
    if len(noise_list) == 0:
        noise_list = []

    if ReZeroOnTheFlyDataset is not None:
        ds = ReZeroOnTheFlyDataset(speech_list, noise_list, fs=16000)
    else:
        ds = DummyDataset(length=256)
    # use a subset by sampling indices
    indices = list(range(min(len(ds), NUM_SAMPLES)))
    if len(indices) == 0:
        indices = list(range(64))
    ds = Subset(ds, indices)

    loader = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True, num_workers=0)

    R = 0.025
    MIC_LOCS = np.array([[R * np.cos(2 * np.pi * i / 8), R * np.sin(2 * np.pi * i / 8), 0.0] for i in range(8)])
    model = ReZeroSystem(mic_locations=torch.from_numpy(MIC_LOCS).float(), bsrnn_channels=CHANNELS, P_dim=P_DIM, task_type='angle').to(device)

    istft = ISTFT(win_len=512, win_shift_ratio=0.25, nfft=512).to(device)
    criterion = ReZeroPaperLoss().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running = 0.0
        n = 0
        for mix, target, region, Q in loader:
            mix = mix.to(device)
            target = target.to(device)
            for k in region: region[k] = region[k].to(device)
            est_stft = model(mix, region)
            est_wav = istft(est_stft, length=mix.shape[1])
            loss = criterion(est_wav, target, est_stft, Q)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            n += 1
            if n >= 20: break
        avg = running / max(1, n)
        print(f'Epoch {epoch}/{NUM_EPOCHS} | Avg Loss (mini) = {avg:.4f}')

    # save a small checkpoint
    os.makedirs('ReZero2/checkpoints/quick_finetune', exist_ok=True)
    torch.save({'model_state_dict': model.state_dict()}, 'ReZero2/checkpoints/quick_finetune/quick.pth')
    print('Quick finetune done. checkpoint saved to ReZero2/checkpoints/quick_finetune/quick.pth')


if __name__ == '__main__':
    main()
