# utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ISTFT(nn.Module):
    def __init__(self, win_len, win_shift_ratio, nfft):
        super().__init__()
        self.win_len = win_len
        self.hop_len = int(win_len * win_shift_ratio)
        self.nfft = nfft
        self.register_buffer('window', torch.hann_window(win_len))

    def forward(self, stft_complex, length=None):
        """
        Input: (B, F, T, 2) or Complex Tensor (B, F, T)
        Output: (B, Samples)
        """
        if not torch.is_complex(stft_complex):
            if stft_complex.shape[-1] == 2:
                stft_complex = torch.view_as_complex(stft_complex)
        
        # 1. 执行 ISTFT (不传入 length 参数以避免警告)
        y = torch.istft(
            stft_complex,
            n_fft=self.nfft,
            hop_length=self.hop_len,
            win_length=self.win_len,
            window=self.window,
            center=True, # 必须与 STFT 保持一致 (默认是 True)
            return_complex=False
        )
        
        # 2. 手动对齐长度 (关键修复)
        if length is not None:
            if y.shape[-1] < length:
                # 如果短了，在末尾补零
                pad_len = length - y.shape[-1]
                y = F.pad(y, (0, pad_len))
            elif y.shape[-1] > length:
                # 如果长了，截断末尾
                y = y[..., :length]
                
        return y