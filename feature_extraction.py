import torch
import torch.nn as nn
from utils.stft import STFT

class CalculateSpatialFeatures(nn.Module):
    """
    计算空间特征 (IPD & ILD)
    论文参考: Equation (3) 
    """
    def __init__(self, win_len, win_shift_ratio, nfft, ch_mode='MM'):
        super().__init__()
        self.stft = STFT(win_len, win_shift_ratio, nfft)
        self.ch_mode = ch_mode
        self.epsilon = 1e-6 # 增大 epsilon 提高稳定性

    def forward(self, signal):
        # 1. STFT: (B, Ch, T) -> (B, Ch, F, T) complex
        stft_complex = self.stft(signal)
        # 调整维度为 (B, Ch, F, T)
        stft_data = stft_complex.permute(0, 3, 1, 2) 
        nb, nch, nf, nt = stft_data.shape
        
        # 2. 生成麦克风对
        pairs_p1, pairs_p2 = [], []
        if self.ch_mode == 'M': 
            for i in range(1, nch):
                pairs_p1.append(stft_data[:, 0])
                pairs_p2.append(stft_data[:, i])
        elif self.ch_mode == 'MM': 
            for i in range(nch - 1):
                for j in range(i + 1, nch):
                    pairs_p1.append(stft_data[:, i])
                    pairs_p2.append(stft_data[:, j])
        
        # Stack pairs: (B*Pairs, F, T)
        Y_p1 = torch.cat(pairs_p1, dim=0)
        Y_p2 = torch.cat(pairs_p2, dim=0)
        
        # 3. IPD (Interaural Phase Difference)
        # 优化: 使用共轭乘积计算相位差，避免分别计算 angle 再相减导致的相位缠绕问题
        # IPD = angle(Y1 * conj(Y2))
        cross_spectrum = Y_p1 * torch.conj(Y_p2)
        ipd = torch.angle(cross_spectrum) # 自动限制在 (-pi, pi]
        
        # 4. ILD (Interaural Level Difference)
        # 优化: 增加 clamp 防止 log(0) 或数值爆炸
        mag_p1 = torch.abs(Y_p1)
        mag_p2 = torch.abs(Y_p2)
        log_ratio = torch.log10(mag_p1 + self.epsilon) - torch.log10(mag_p2 + self.epsilon)
        ild = 20 * log_ratio
        ild = torch.clamp(ild, min=-20, max=20) # 截断到合理范围 [-20dB, 20dB]
        
        return ipd, ild, stft_complex

class CalculateDirectionFeature(nn.Module):
    """
    计算方向特征 V(theta, phi)
    论文参考: Equation (4) [cite: 98, 101]
    """
    def __init__(self, mic_locations, nfft, fs=16000, v=343.0, ch_mode='MM'):
        super().__init__()
        self.v = v
        # 注册频率 bin 向量，避免重复创建
        self.register_buffer('freq_bins', torch.linspace(0, fs/2, nfft//2 + 1))
        
        nch = mic_locations.shape[0]
        pair_vecs = []
        # 计算麦克风对向量 d_p
        if ch_mode == 'M':
            for i in range(1, nch):
                pair_vecs.append(mic_locations[i] - mic_locations[0])
        elif ch_mode == 'MM':
            for i in range(nch - 1):
                for j in range(i + 1, nch):
                    pair_vecs.append(mic_locations[j] - mic_locations[i])
        
        # (Pairs, 3)
        self.register_buffer('pair_vectors', torch.stack(pair_vecs).float())

    def forward(self, observed_ipd, query_azi, query_ele):
        """
        Input:
            observed_ipd: (B*P, F, T) 
            query_azi: (B, N)
            query_ele: (B, N)
        Output:
            V: (B, N, F, T) 方向相似度特征
        """
        num_pairs = self.pair_vectors.shape[0]
        total_batch, nf, nt = observed_ipd.shape
        nbatch = total_batch // num_pairs
        
        # 1. 恢复 Batch 维度: (B, P, F, T)
        obs_ipd = observed_ipd.view(nbatch, num_pairs, nf, nt)
        
        # 2. 计算方向向量 r (单位向量): (B, 3, N)
        # conversion from spherical to cartesian
        r_x = torch.sin(query_ele) * torch.cos(query_azi)
        r_y = torch.sin(query_ele) * torch.sin(query_azi)
        r_z = torch.cos(query_ele)
        
        r = torch.stack([r_x, r_y, r_z], dim=1) # (B, 3, N)
        
        # 3. 计算理论时延 (TDOA): (B, P, N)
        # formula: tau = (d_p . r) / v
        # einsum: 'pc,bcn->bpn' 
        #   p: pairs, c: xyz coords, b: batch, n: views
        tdoa = torch.einsum('pc,bcn->bpn', self.pair_vectors, r) / self.v
        
        # 4. 计算 TPD (Target Phase Difference): (B, P, N, F)
        # formula: TPD = 2 * pi * f * tau
        # tdoa: (B, P, N) -> (B, P, N, 1)
        # freq: (F,) -> (1, 1, 1, F)
        tpd = 2 * torch.pi * tdoa.unsqueeze(-1) * self.freq_bins.view(1, 1, 1, -1)
        
        # 5. 计算 Cosine Similarity V
        # obs_ipd: (B, P, F, T) -> (B, P, 1, F, T)
        # tpd:     (B, P, N, F) -> (B, P, N, F, 1)
        # Broadcasting -> (B, P, N, F, T)
        
        # 这里的 diff 直接相减即可，因为 cos(a-b) 即使 a-b 超出 2pi 也是正确的
        diff = obs_ipd.unsqueeze(2) - tpd.unsqueeze(-1)
        similarity = torch.cos(diff)
        
        # 6. 对麦克风对求和 (Sum over Pairs) [cite: 98]
        # (B, P, N, F, T) -> (B, N, F, T)
        V = torch.sum(similarity, dim=1)
        
        return V