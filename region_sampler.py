import torch
import torch.nn as nn
import numpy as np

class RegionSampler(nn.Module):
    """
    实现论文 D. Region feature sampling
    支持 Batch 维度并行采样
    """
    def __init__(self, sampling_mode='fixed_number', num_views_N=8, interval_delta_theta_deg=None):
        super().__init__()
        self.sampling_mode = sampling_mode
        self.N = num_views_N
        if interval_delta_theta_deg is not None:
            self.delta_theta = np.deg2rad(interval_delta_theta_deg)

    def forward(self, azi_low, azi_high, ele_low, ele_high, dist_low, dist_high):
        """
        输入: Tensor (Batch_Size,)
        输出: 
            sampled_azimuths: (Batch, N)
            sampled_elevations: (Batch, N)
            sampled_distances: (Batch, 2)
        """
        # 确保输入是 Tensor
        if not isinstance(azi_low, torch.Tensor):
            device = 'cpu'
            azi_low = torch.tensor(azi_low)
            azi_high = torch.tensor(azi_high)
        else:
            device = azi_low.device

        batch_size = azi_low.shape[0]

        # 1. 角度采样 (Azimuth)
        # 目标: 生成 (Batch, N) 的 grid
        
        # 计算中心仰角 (Batch,)
        center_elevation = (ele_low + ele_high) / 2.0
        
        if self.sampling_mode == 'fixed_number':
            if self.N == 1:
                # (Batch, 1)
                sampled_azimuths = (azi_low + azi_high).unsqueeze(1) / 2.0
            else:
                # 生成 0 到 1 的线性插值步长 (N,)
                steps = torch.linspace(0, 1, self.N, device=device)
                
                # 扩展维度以便广播:
                # low: (B, 1), steps: (1, N) -> (B, N)
                low = azi_low.unsqueeze(1)
                high = azi_high.unsqueeze(1)
                steps = steps.unsqueeze(0)
                
                sampled_azimuths = low + (high - low) * steps
            
        elif self.sampling_mode == 'fixed_interval':
            # 定间隔采样在 Batch 模式下较难对齐（因为不同样本可能有不同数量的view）
            # 这里为了 Batch 训练稳定性，退化为取中心值或者强制 Padding
            # 论文中提到定间隔会变长，通常训练时推荐使用 fixed_number
            # 这里简单实现为取中心
            sampled_azimuths = (azi_low + azi_high).unsqueeze(1) / 2.0

        # 扩展 Elevation 到 (Batch, N)
        # center_ele: (B,) -> (B, 1) -> (B, N)
        sampled_elevations = center_elevation.unsqueeze(1).expand_as(sampled_azimuths)
        
        # 2. 距离采样 (Batch, 2)
        # dist_low: (B,) -> (B, 2)
        sampled_distances = torch.stack([dist_low, dist_high], dim=1)
        
        return sampled_azimuths, sampled_elevations, sampled_distances