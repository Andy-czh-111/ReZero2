import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import glob
import random
import soundfile as sf
import time
from datetime import datetime

# 引入你的模块
from model.system import ReZeroSystem
from data.dataset import ReZeroOnTheFlyDataset
from utils.loss import ReZeroPaperLoss
from utils.istft import ISTFT

from contextlib import nullcontext
from torch.cuda.amp import GradScaler
from torch.amp import autocast


class AmpAutocast:
    def __call__(self):
        # Use CUDA autocast when available, otherwise no-op
        if torch.cuda.is_available():
            return autocast(device_type='cuda')
        else:
            return nullcontext()


class AmpScaler:
    def __call__(self):
        # Use GradScaler on CUDA, otherwise provide a dummy scaler
        if torch.cuda.is_available():
            return GradScaler()
        else:
            class DummyScaler:
                def scale(self, x):
                    return x
                def unscale_(self, optimizer):
                    return
                def step(self, optimizer):
                    optimizer.step()
                def update(self):
                    return
                def state_dict(self):
                    return {}
                def load_state_dict(self, d):
                    return
            return DummyScaler()


get_autocast = AmpAutocast()
get_scaler = AmpScaler()

# === 1. 终极配置类 ===
class Config:
    SPEECH_DIR = "/Project/Separation/Data/LibriSpeech/train-clean-360" 
    NOISE_DIR = "/Project/Separation/Data/Musan"
    EXP_NAME = "ReZero_Final_Run_12261705"
    LOG_DIR = f"./logs/{EXP_NAME}"
    CKPT_DIR = f"./checkpoints/{EXP_NAME}"
    
    BATCH_SIZE = 16 
    LR = 1e-4
    NUM_EPOCHS = 200
    VAL_SPLIT = 0.1
    NUM_WORKERS = 8 
    
    GRAD_ACCUM_STEPS = 1       
    GRAD_CLIP = 5.0            
    LR_DECAY_STEP = 2          
    LR_DECAY_GAMMA = 0.98      
    
    PATIENCE = 20             
    MIN_DELTA = 0.001          
    
    SAVE_EVERY = 5             
    SEED = 42                  

# === 2. 辅助工具类 ===
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"⚠️ EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔒 Random Seed set to {seed}")

def create_dummy_data(data_dir, num_files=10):
    if not os.path.exists(data_dir): os.makedirs(data_dir)
    files = []
    sr = 16000
    for i in range(num_files):
        y = np.random.uniform(-0.1, 0.1, size=(sr * 5,))
        path = os.path.join(data_dir, f"dummy_{i}.wav")
        sf.write(path, y, sr)
        files.append(path)
    return files

def collate_fn(batch):
    mix_list, target_list, Q_list = [], [], []
    region_batch = {k: [] for k in batch[0]['region'].keys()}
    for item in batch:
        mix_list.append(item['mix'])
        target_list.append(item['target'])
        Q_list.append(item['Q'])
        for k, v in item['region'].items():
            region_batch[k].append(v)
    
    mix = torch.stack(mix_list)
    target = torch.stack(target_list)
    Q = torch.tensor(Q_list)
    for k in region_batch:
        region_batch[k] = torch.stack(region_batch[k])
    return mix, target, region_batch, Q

def validate(model, val_loader, criterion, device, istft):
    model.eval()
    total_val_loss = 0.0
    progress = tqdm(val_loader, desc="Validating", leave=False)
    
    with torch.no_grad():
        for mix, target, region_params, Q in progress:
            mix, target, Q = mix.to(device, non_blocking=True), target.to(device, non_blocking=True), Q.to(device, non_blocking=True)
            for k in region_params: region_params[k] = region_params[k].to(device, non_blocking=True)
            
            # [修复调用]
            with get_autocast():
                est_stft = model(mix, region_params)
                est_wav = istft(est_stft, length=mix.shape[1])
                loss = criterion(est_wav, target, est_stft, Q)
                
            total_val_loss += loss.item()
            
    return total_val_loss / len(val_loader)

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, loss, path, is_best=False):
    model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    state = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(os.path.dirname(path), "best_model.pth")
        torch.save(state, best_path)
        print(f"✨ Best Model saved: {best_path} (Loss: {loss:.4f})")

# === 3. 主训练流程 ===
def train():
    set_seed(Config.SEED)
    if not os.path.exists(Config.LOG_DIR): os.makedirs(Config.LOG_DIR)
    if not os.path.exists(Config.CKPT_DIR): os.makedirs(Config.CKPT_DIR)
    
    writer = SummaryWriter(log_dir=Config.LOG_DIR)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f"🚀 Device: {device}")
    
    # 数据加载
    speech_list = glob.glob(os.path.join(Config.SPEECH_DIR, "**/*.flac"), recursive=True) + \
                  glob.glob(os.path.join(Config.SPEECH_DIR, "**/*.wav"), recursive=True)
    noise_list = glob.glob(os.path.join(Config.NOISE_DIR, "**/*.wav"), recursive=True)
    
    if len(speech_list) == 0: speech_list = create_dummy_data(Config.SPEECH_DIR)
    if len(noise_list) == 0: noise_list = create_dummy_data(Config.NOISE_DIR)

    random.shuffle(speech_list)
    split_idx = int(len(speech_list) * (1 - Config.VAL_SPLIT))
    train_speech, val_speech = speech_list[:split_idx], speech_list[split_idx:]
    
    # MIC_LOCS = np.array([
    #     [0.025, 0, 0], [0.017, 0.017, 0], [0, 0.025, 0], [-0.017, 0.017, 0],
    #     [-0.025, 0, 0], [-0.017, -0.017, 0], [0, -0.025, 0], [0.017, -0.017, 0]
    # ])
    R = 0.025
    MIC_LOCS = np.array([
    [R * np.cos(2 * np.pi * i / 8), R * np.sin(2 * np.pi * i / 8), 0.0] 
    for i in range(8)
])

    model = ReZeroSystem(
        mic_locations=torch.from_numpy(MIC_LOCS).float(),
        bsrnn_channels=48,
        P_dim=16,
        task_type='angle'
    ).to(device)

    # Enable cuDNN benchmark for fixed-size inputs to improve speed (if using CUDA)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # Try compiling the model with torch.compile if available (PyTorch 2.x)
    try:
        if hasattr(torch, 'compile'):
            model = torch.compile(model)
            print('Model compiled with torch.compile')
    except Exception as e:
        print('torch.compile failed, continuing without compile:', e)

    train_dataset = ReZeroOnTheFlyDataset(train_speech, noise_list, fs=16000)
    val_dataset = ReZeroOnTheFlyDataset(val_speech, noise_list, fs=16000)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, collate_fn=collate_fn, shuffle=True, 
                              num_workers=Config.NUM_WORKERS, pin_memory=use_cuda, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, collate_fn=collate_fn, shuffle=False, 
                            num_workers=Config.NUM_WORKERS, pin_memory=use_cuda, persistent_workers=True)

    istft = ISTFT(win_len=512, win_shift_ratio=0.25, nfft=512).to(device)
    criterion = ReZeroPaperLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=1e-5)
    
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=Config.LR_DECAY_STEP, gamma=Config.LR_DECAY_GAMMA)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=5, 
    verbose=True,
    min_lr=1e-6
)
    scaler = get_scaler()
    
    early_stopping = EarlyStopping(patience=Config.PATIENCE, min_delta=Config.MIN_DELTA)
    best_val_loss = float('inf')

    print(f"🎬 开始训练 | Log: {Config.LOG_DIR} | Patience: {Config.PATIENCE}")
    
    try:
        for epoch in range(1, Config.NUM_EPOCHS + 1):
            start_time = time.time()
            model.train()
            running_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{Config.NUM_EPOCHS}")
            optimizer.zero_grad()
            
            for step, (mix, target, region_params, Q) in enumerate(pbar):
                mix = mix.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                Q = Q.to(device, non_blocking=True)
                for k in region_params: region_params[k] = region_params[k].to(device, non_blocking=True)
                
                with get_autocast():
                    est_stft = model(mix, region_params)
                    est_wav = istft(est_stft, length=mix.shape[1])
                    loss = criterion(est_wav, target, est_stft, Q)
                    loss = loss / Config.GRAD_ACCUM_STEPS

                scaler.scale(loss).backward()
                
                if (step + 1) % Config.GRAD_ACCUM_STEPS == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                current_loss = loss.item() * Config.GRAD_ACCUM_STEPS
                running_loss += current_loss
                
                pbar.set_postfix(loss=f"{current_loss:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")
                writer.add_scalar('Loss/Train_Step', current_loss, (epoch-1)*len(train_loader)+step)

            

            avg_train_loss = running_loss / len(train_loader)
            avg_val_loss = validate(model, val_loader, criterion, device, istft)
            # scheduler.step()
            scheduler.step(avg_val_loss)
            
            epoch_time = time.time() - start_time
            
            writer.add_scalar('Loss/Train_Epoch', avg_train_loss, epoch)
            writer.add_scalar('Loss/Val_Epoch', avg_val_loss, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

            print(f"Epoch {epoch}: Train={avg_train_loss:.4f} | Val={avg_val_loss:.4f} | Time={epoch_time:.0f}s")

            is_best = avg_val_loss < best_val_loss
            if is_best: best_val_loss = avg_val_loss
            
            save_path = os.path.join(Config.CKPT_DIR, "last_model.pth")
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, avg_val_loss, save_path, is_best)
            
            if epoch % Config.SAVE_EVERY == 0:
                archive_path = os.path.join(Config.CKPT_DIR, f"epoch_{epoch}.pth")
                torch.save(torch.load(save_path), archive_path)

            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                print(f"🛑 早停触发! 验证集 Loss 已经 {Config.PATIENCE} 个 Epoch 没有显著下降。")
                break
                
    except KeyboardInterrupt:
        print("\n⛔ 训练被手动中断. 保存状态中...")
        save_path = os.path.join(Config.CKPT_DIR, "interrupted_model.pth")
        save_checkpoint(model, optimizer, scheduler, scaler, epoch, running_loss, save_path, is_best=False)
        
    print(f"✅ 训练结束. 最佳验证 Loss: {best_val_loss:.4f}")
    writer.close()

if __name__ == '__main__':
    train()

