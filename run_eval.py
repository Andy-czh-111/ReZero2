import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

# 导入你的模块
from model.system import ReZeroSystem
from data.dataset import ReZeroOnTheFlyDataset
from utils.istft import ISTFT
from evaluate import compute_metrics  # <--- 核心导入

# === 配置 ===
CHECKPOINT_PATH = "/Project/nerual_beamforming/myproject/ReZero2/checkpoints/ReZero_Final_Run_12261705/best_model.pth"
# 注意：这里必须使用真实的测试数据路径，不能是伪造数据，否则指标无意义
SPEECH_DIR = "/Project/Separation/Data/LibriSpeech/dev-clean" 
NOISE_DIR = "/Project/Separation/Data/Musan"
BATCH_SIZE = 1 # 评测时建议 Batch=1 以避免 padding 对指标的微小影响
NUM_TEST_SAMPLES = 100 # 测试多少个样本 (已缩减以便快速评估)

def run_evaluation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载模型
    # 确保 mic_locations 与训练时一致
    R = 0.025
    MIC_LOCS = np.array([
    [R * np.cos(2 * np.pi * i / 8), R * np.sin(2 * np.pi * i / 8), 0.0] 
    for i in range(8)
])
    model = ReZeroSystem(
        mic_locations=torch.from_numpy(MIC_LOCS).float(),
        bsrnn_channels=48,
        task_type='angle'
    ).to(device)
    
    # 加载权重 (处理 DataParallel 的 module. 前缀)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    state_dict = ckpt['model_state_dict']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    
    istft = ISTFT(win_len=512, win_shift_ratio=0.25, nfft=512).to(device)

    # 2. 准备数据
    import glob
    import os
    speech_list = glob.glob(os.path.join(SPEECH_DIR, "**/*.flac"), recursive=True)
    noise_list = glob.glob(os.path.join(NOISE_DIR, "**/*.wav"), recursive=True)
    
    if len(speech_list) == 0:
        print("❌ 错误：未找到真实的语音文件。请检查 SPEECH_DIR 路径。")
        return

    # 使用 Dataset 生成测试数据
    test_dataset = ReZeroOnTheFlyDataset(speech_list, noise_list, fs=16000)
    # 限制测试样本数
    indices = list(range(min(len(test_dataset), NUM_TEST_SAMPLES)))
    test_dataset = torch.utils.data.Subset(test_dataset, indices)
    
    # 自定义 collate_fn (复用 train1.py 中的)
    from train import collate_fn 
    loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)

    # 3. 统计容器
    metrics_avg = {'sdr': [], 'stoi': [], 'pesq': [], 'decay': [], 'energy_out_db': []}

    print(f"🚀 开始评测 {len(loader)} 个样本...")
    
    with torch.no_grad():
        for i, (mix, target, region_params, Q) in enumerate(tqdm(loader)):
            mix = mix.to(device)
            region_params = {k: v.to(device) for k, v in region_params.items()}
            
            # --- 推理 ---
            # 你的模型可能使用了 AMP 包装，评测时可以直接调用或加上 autocast
            est_stft = model(mix, region_params)
            est_wav = istft(est_stft, length=mix.shape[1]) # (B, T)
            
            # --- 数据转换 (Tensor -> Numpy) ---
            # compute_metrics 需要一维 Numpy 数组
            est_np = est_wav[0].cpu().numpy()
            target_np = target[0].cpu().numpy()
            mix_np = mix[0, :, 0].cpu().numpy() # 取第一个麦克风作为参考混合信号
            q_val = Q.item()
            
            # --- 计算指标 ---
            # 调用 evaluate.py 中的函数
            res = compute_metrics(est_np, target_np, mix_np, q_val)
            
            # --- 记录结果 ---
            for k, v in res.items():
                if v != -np.inf: # 过滤掉计算失败的情况
                    metrics_avg[k].append(v)

    # 4. 输出平均结果
    print("\n📊 评测结果汇总:")
    if metrics_avg['decay']:
        print(f"  [Q=0] Avg Energy Decay: {np.mean(metrics_avg['decay']):.2f} dB (越低越好)")
    else:
        print("  [Q=0] 无样本")
        
    if metrics_avg['sdr']:
        print(f"  [Q>0] Avg SDR:  {np.mean(metrics_avg['sdr']):.2f} dB")
        print(f"  [Q>0] Avg STOI: {np.mean(metrics_avg['stoi']):.3f}")
        print(f"  [Q>0] Avg PESQ: {np.mean(metrics_avg['pesq']):.2f}")
    else:
        print("  [Q>0] 无样本")

if __name__ == '__main__':
    run_evaluation()