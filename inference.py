import os
import argparse
import torch
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 引入项目模块
from model.system import ReZeroSystem
from utils.istft import ISTFT
# compute_metrics 会在需要时延迟导入（避免外部依赖在无环境下报错）

# === 配置 ===
DEFAULT_CHECKPOINT = "/Project/nerual_beamforming/myproject/ReZero2/checkpoints/ReZero_Final_Run_12261705/best_model.pth"
DEFAULT_OUTPUT = "./results"
DEFAULT_SPEECH_DIR = "/Project/Separation/Data/LibriSpeech/train-clean-360"
DEFAULT_NOISE_DIR = "/Project/Separation/Data/Musan"

def load_model(device, checkpoint_path=None):
    R = 0.025
    MIC_LOCS = np.array([
    [R * np.cos(2 * np.pi * i / 8), R * np.sin(2 * np.pi * i / 8), 0.0] 
    for i in range(8)
])
    
    # 实例化模型
    model = ReZeroSystem(
        mic_locations=torch.from_numpy(MIC_LOCS).float(),
        bsrnn_channels=48,
        task_type='angle'
    )
    
    # 加载权重
    ckpt_to_load = checkpoint_path if checkpoint_path is not None else DEFAULT_CHECKPOINT
    if os.path.exists(ckpt_to_load):
        print(f"正在加载模型: {ckpt_to_load}")
        checkpoint = torch.load(ckpt_to_load, map_location=device)
        # 兼容 DataParallel 的权重键名 (去除 'module.' 前缀)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "") 
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        print(f"⚠️ 检查点未找到: {ckpt_to_load}. 使用随机初始化模型进行推理 (仅用于 smoke-test)。")
    
    model.to(device)
    model.eval()
    return model

def inference(checkpoint_path=None, speech_dir=None, noise_dir=None, output_dir=None, use_dataset=False, device_str=None):
    device = torch.device(device_str) if device_str is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = output_dir if output_dir is not None else DEFAULT_OUTPUT
    if not os.path.exists(out_dir): os.makedirs(out_dir, exist_ok=True)

    # 1. 加载模型
    model = load_model(device, checkpoint_path=checkpoint_path)
    istft = ISTFT(win_len=512, win_shift_ratio=0.25, nfft=512).to(device)

    # 2. 准备测试数据 (使用 Dataset 生成一个样本)
    # 注意: 为了测试泛化性，最好使用训练集中没见过的文件
    import glob
    sdir = speech_dir if speech_dir is not None else DEFAULT_SPEECH_DIR
    ndir = noise_dir if noise_dir is not None else DEFAULT_NOISE_DIR
    speech_list = glob.glob(os.path.join(sdir, "**/*.flac"), recursive=True) + glob.glob(os.path.join(sdir, "**/*.wav"), recursive=True)
    noise_list = glob.glob(os.path.join(ndir, "**/*.wav"), recursive=True)

    # 尝试使用 Dataset 获取样本；如果没有数据或读取失败，回退为合成样本
    sample = None
    try:
        if len(speech_list) > 0 and len(noise_list) > 0:
            # 延迟导入以避免在没有 pyroomacoustics 的环境中直接报错
            try:
                from data.dataset import ReZeroOnTheFlyDataset
            except Exception as e:
                print(f"无法导入 ReZeroOnTheFlyDataset: {e}")
                raise

            test_dataset = ReZeroOnTheFlyDataset(speech_list[:10], noise_list[:10], fs=16000)
            print("正在生成测试样本 (来自 Dataset)...")
            sample = test_dataset[0]
    except Exception as e:
        print(f"从 Dataset 获取样本失败: {e}")

    if sample is None:
        print("使用合成样本作为回退 (随机信号)。")
        T = 16000
        Ch = 8
        mix = torch.randn(1, T, Ch).to(device)
        target = torch.randn(1, T).mul(0.05).to(device)
        Q = 1
        region_params = {
            'azi_low': torch.tensor([-0.5]).to(device),
            'azi_high': torch.tensor([0.5]).to(device),
            'ele_low': torch.tensor([0.0]).to(device),
            'ele_high': torch.tensor([1.57]).to(device),
            'dist_low': torch.tensor([0.0]).to(device),
            'dist_high': torch.tensor([2.0]).to(device)
        }
    else:
        # 增加 Batch 维度
        mix = sample['mix'].unsqueeze(0).to(device)       # (1, T, M)
        target = sample['target'].unsqueeze(0).to(device) # (1, T)
        Q = sample['Q'].item()
        region_params = sample['region']
        for k in region_params:
            region_params[k] = region_params[k].unsqueeze(0).to(device)

    # 3. 推理 (Inference)
    with torch.no_grad():
        # 这里需要用你的 AMP 包装或直接跑 (推理通常不需要 AMP)
        est_stft = model(mix, region_params)
        est_wav = istft(est_stft, length=mix.shape[1]) # (1, T)

    # 4. 保存音频
    mix_np = mix[0, :, 0].cpu().numpy() # 只保存第0个通道的混合音频
    est_np = est_wav[0].cpu().numpy()
    target_np = target[0].cpu().numpy()
    
    # 文件名信息
    azi_l = np.rad2deg(region_params['azi_low'].item())
    azi_h = np.rad2deg(region_params['azi_high'].item())
    prefix = f"Q{Q}_Azi{int(azi_l)}_{int(azi_h)}"
    
    sf.write(os.path.join(out_dir, f"{prefix}_mix.wav"), mix_np, 16000)
    sf.write(os.path.join(out_dir, f"{prefix}_est.wav"), est_np, 16000)
    sf.write(os.path.join(out_dir, f"{prefix}_ref.wav"), target_np, 16000)
    
    print(f"\n✅ 音频已保存到 {DEFAULT_OUTPUT}/")
    print(f"  - Mix: {prefix}_mix.wav (混合音频)")
    print(f"  - Est: {prefix}_est.wav (模型输出)")
    print(f"  - Ref: {prefix}_ref.wav (参考目标)")

    # 5. 计算指标（尝试导入 evaluate.compute_metrics，否则使用回退实现）
    print("\n📊 性能指标:")
    try:
        from evaluate import compute_metrics
    except Exception:
        def compute_metrics(est_wav, target_wav, mix_wav, Q, fs=16000):
            # 本地回退实现：decay / 简单 SNR
            eps = 1e-9
            metrics = {}
            if Q == 0:
                energy_in = np.sum(mix_wav**2) + eps
                energy_out = np.sum(est_wav**2) + eps
                metrics['decay'] = 10 * np.log10(energy_out / energy_in)
            else:
                # 简单 SNR 近似
                num = np.sum(target_wav**2) + eps
                den = np.sum((target_wav - est_wav)**2) + eps
                metrics['sdr'] = 10 * np.log10(num / den)
                metrics['stoi'] = 0.0
                metrics['pesq'] = 1.0
            return metrics

    try:
        metrics = compute_metrics(est_np, target_np, mix_np, Q)
        for k, v in metrics.items():
            try:
                print(f"  - {k}: {v:.4f}")
            except Exception:
                print(f"  - {k}: {v}")

        if Q > 0 and metrics.get('sdr', -99) < 5.0:
            print("\n⚠️ 警告: SDR 很低 (< 5dB)。模型可能未成功分离。")

    except Exception as e:
        print(f"无法计算指标: {e}")

    # 6. (可选) 绘制波形图
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1); plt.plot(mix_np); plt.title("Mixture (Ch0)"); plt.grid()
    plt.subplot(3, 1, 2); plt.plot(target_np); plt.title(f"Target (Q={Q})"); plt.grid()
    plt.subplot(3, 1, 3); plt.plot(est_np); plt.title("Estimated"); plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_waveform.png"))
    print(f"  - Plot: {prefix}_waveform.png")

def _parse_args():
    p = argparse.ArgumentParser(description='ReZero inference (smoke-test friendly)')
    p.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    p.add_argument('--speech_dir', type=str, default=None, help='Path to speech dataset (optional)')
    p.add_argument('--noise_dir', type=str, default=None, help='Path to noise dataset (optional)')
    p.add_argument('--output_dir', type=str, default=None, help='Output dir for wavs and plots')
    p.add_argument('--use_dataset', action='store_true', help='Try to use ReZeroOnTheFlyDataset if available')
    p.add_argument('--device', type=str, default=None, help='Device string (e.g., cpu or cuda:0)')
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    inference(checkpoint_path=args.checkpoint, speech_dir=args.speech_dir, noise_dir=args.noise_dir,
              output_dir=args.output_dir, use_dataset=args.use_dataset, device_str=args.device)