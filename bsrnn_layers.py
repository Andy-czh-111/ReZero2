import torch
import torch.nn as nn
import torch.nn.utils as nn_utils

def get_band_specs(fs=16000, nfft=512):
    """
    10个100Hz宽的子带（覆盖0-1000Hz）
    12个200Hz宽的子带（覆盖1000-3400Hz）
    8个500Hz宽的子带（覆盖3400-7400Hz）
    剩余频带作为一个子带（覆盖7400-8000Hz）
    
    Args:
        fs: 采样率，默认为16000Hz
        nfft: FFT点数，默认为512
    
    Returns:
        torch.Tensor: 每个频带对应的bin数量
    """
    # 计算频率分辨率
    freq_resolution = fs / nfft     # 16000/512 = 31.25Hz/bin
    # 定义频带宽度：10个100Hz + 12个200Hz + 8个500Hz = 7400
    bands_hz = [100]*10 + [200]*12 + [500]*8   
    
    band_width_bins = []
    current_freq = 0
    total_bins = nfft // 2 + 1  # 总bin数（只考虑正频率）
    
    for bh in bands_hz:
        # 计算该带宽对应的 bin 数量
        bw_bin = int(bh / freq_resolution)
        band_width_bins.append(bw_bin)
        current_freq += bh
        
    # 处理剩余频带
    used_bins = sum(band_width_bins)
    if used_bins < total_bins:
        band_width_bins.append(total_bins - used_bins)
        
    return torch.tensor(band_width_bins, dtype=torch.long)

class BandSplit(nn.Module):
    def __init__(self, channels=128, fs=16000, nfft=512, input_dim=2):
        """
        频带分割模块 - 将频谱分割成多个子带并分别处理
        
        Args:
            channels: BSRNN 内部特征维度
            fs: 采样率
            nfft: FFT 点数
            input_dim: 输入特征的维度 (复数谱为2, 空间特征可能为1或麦克风对数)
        """
        super(BandSplit, self).__init__()
        # 动态获取频带划分
        self.band = get_band_specs(fs, nfft)
        self.subband_modules = nn.ModuleList()
        
        # 为每个子带创建独立的批归一化和全连接层
        for i, bw in enumerate(self.band):
            # 使用 LayerNorm 替代 BatchNorm1d，避免小 batch 下统计不稳定的问题
            self.subband_modules.append(nn.Sequential(
                nn.LayerNorm(int(bw) * input_dim),
                nn.Linear(int(bw) * input_dim, channels)
            ))

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (B, input_dim, F, T)
                B: batch size
                input_dim: 输入维度
                F: 频率点数
                T: 时间帧数
        
        Returns:
            torch.Tensor: 拼接后的子带特征，形状为 (B, K, T, C)
                K: 子带数量
                C: 通道数
        """
        # x: (B, input_dim, F, T) -> 需要维度重排
        b, c, f, t = x.shape
        # 重排维度为 (B, T, F, input_dim)
        x = x.permute(0, 3, 2, 1).contiguous()
        z_list = []  # 存储每个子带的输出
        curr_bin = 0  # 当前处理的频率bin起始位置
        
        # 遍历每个子带
        for i, bw in enumerate(self.band):
            # 提取当前子带的频谱数据
            out = x[:, :, curr_bin : curr_bin + bw, :].reshape(b*t, -1)
            curr_bin += bw  # 更新bin位置
            
            # 应用子带特定的批归一化和全连接层
            out = self.subband_modules[i](out)
            
            # 重塑形状并添加到列表
            z_list.append(out.view(b, t, -1).unsqueeze(1))
        
        # 沿子带维度拼接所有输出
        return torch.cat(z_list, dim=1)  # (B, K, T, C)

class MaskDecoder(nn.Module):
    def __init__(self, channels=128, fs=16000, nfft=512):
        """
        掩码解码器 - 将子带特征解码为复数掩码
        
        Args:
            channels: 输入特征维度
            fs: 采样率
            nfft: FFT点数
        """
        super(MaskDecoder, self).__init__()
        self.band = get_band_specs(fs, nfft)
        
        # 为每个子带创建独立的解码层
        for i in range(len(self.band)):
            # 组归一化层
            setattr(self, f'norm{i+1}', nn.GroupNorm(1, channels))
            # 第一个全连接层，扩展特征维度（使用 weight_norm 稳定训练）
            fc1 = nn.Linear(channels, 4*channels)
            nn_utils.weight_norm(fc1)
            setattr(self, f'fc1{i+1}', fc1)
            
            # 输出维度计算：目标为每子带输出 bw，GLU 需要双倍输入 -> bw * 2
            out_dim = int(self.band[i] * 2)
            fc2 = nn.Linear(4*channels, out_dim)
            nn_utils.weight_norm(fc2)
            setattr(self, f'fc2{i+1}', fc2)
            # GLU激活函数，用于门控机制
            setattr(self, f'glu{i+1}', nn.GLU(dim=-1))

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (B, K, T, C)
                B: batch size
                K: 子带数量
                T: 时间帧数
                C: 通道数
        
        Returns:
            torch.Tensor: 复数掩码，形状为 (B, F, T, 2)
                F: 频率点数
                2: 实部和虚部
        """
        # x: (B, K, T, C)
        nb, _, nt, _ = x.shape
        m_list = []  # 存储每个子带的掩码
        
        # 遍历每个子带
        for i in range(len(self.band)):
            # 提取当前子带的特征
            x_band = x[:, i, :, :]  # (B, T, C)
            
            # 获取当前子带的层
            norm = getattr(self, f'norm{i+1}')
            fc1 = getattr(self, f'fc1{i+1}')
            fc2 = getattr(self, f'fc2{i+1}')
            glu = getattr(self, f'glu{i+1}')
            
            # 应用组归一化（需要维度重排）
            out = norm(x_band.permute(0, 2, 1)).permute(0, 2, 1)
            # 第一个全连接层 + tanh激活
            out = torch.tanh(fc1(out))
            # 第二个全连接层 + GLU激活
            # GLU -> 得到 bw (幅度掩码维度)
            out = glu(fc2(out))  # (B, T, bw)
            # 将掩码映射到 (0, 2) 范围 (sigmoid * 2)，以允许对能量进行轻微放大
            out = torch.sigmoid(out) * 2.0
            
            m_list.append(out)
            
        # 沿频率维度拼接所有子带掩码
        m = torch.cat(m_list, dim=-1)  # (B, T, F)
        # 变形为 (B, F, T)
        m = m.reshape(nb, nt, -1).permute(0, 2, 1)  # (B, F, T)
        return m