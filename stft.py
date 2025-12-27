import torch
import torch.nn as nn
import numpy as np

class STFT(nn.Module):
    def __init__(self, win_len, win_shift_ratio, nfft, win='hann'):
        """
        初始化短时傅里叶变换(STFT)的核心参数配置
        """
        super(STFT, self).__init__()
        
        self.win_len = win_len
        self.hop_len = int(win_len * win_shift_ratio)
        self.nfft = nfft
        self.win = win
        
        # [优化] 使用 register_buffer 注册窗函数
        # 这样窗函数会自动随模型移动到 GPU，且不会被视为模型参数(Parameter)
        if win == 'hann':
            self.register_buffer('window', torch.hann_window(win_len))
        else:
            self.register_buffer('window', torch.ones(win_len))

    def forward(self, signal):
        """
        前向传播
        Args:
            signal: (Batch, Channel, Time) - 注意这里适配了 System.py 的输出
        Returns:
            stft: (Batch, Freq, Time, Channel) - 适配 FeatureExtraction 的期望
        """
        # 1. 获取维度信息
        # 假设输入是 (B, C, T)
        b, c, t = signal.shape
        
        # 2. [优化] 维度展平
        # 将 (B, C, T) 变为 (B*C, T)，利用 GPU 并行计算所有通道
        signal_flat = signal.reshape(-1, t)
        
        # 3. 计算 STFT
        # center=True 是为了和 ISTFT 保持一致，避免重建错位 [重要修复]
        stft_flat = torch.stft(
            signal_flat, 
            n_fft=self.nfft, 
            hop_length=self.hop_len, 
            win_length=self.win_len, 
            window=self.window, 
            center=True,        # <--- 必须改为 True 以匹配 istft.py
            return_complex=True,
            normalized=False
        )
        # stft_flat shape: (B*C, F, T) (complex)
        
        # 4. 恢复维度
        # (B*C, F, T) -> (B, C, F, T)
        _, f, frames = stft_flat.shape
        stft = stft_flat.reshape(b, c, f, frames)
        
        # 5. 调整输出维度以匹配下游代码
        # feature_extraction.py 期望的是 (B, F, T, C)
        # 因为它后面会做 .permute(0, 3, 1, 2) 变回 (B, C, F, T)
        stft = stft.permute(0, 2, 3, 1) # (B, F, T, C)

        return stft