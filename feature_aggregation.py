import torch
import torch.nn as nn
from .bsrnn_layers import get_band_specs

class RegionFeatureAggregator(nn.Module):
    """
    Region Feature Aggregation: RNN-Loop
    论文参考: Section IV-E, Fig 5(e) [cite: 201]
    """
    def __init__(self, hidden_dim_P, fs=16000, nfft=512):
        super().__init__()
        self.P = hidden_dim_P
        self.band_specs = get_band_specs(fs, nfft)
        
        # Subband Projection: Map different bandwidths to fixed dimension P
        self.subband_projs = nn.ModuleList([
            nn.Linear(int(bw), hidden_dim_P) for bw in self.band_specs
        ])
        
        # Shared LSTM for all subbands
        # Input: P, Hidden: P
        self.lstm = nn.LSTM(input_size=hidden_dim_P, hidden_size=hidden_dim_P, batch_first=True)

    def forward(self, V):
        """
        Input: V (Batch, N_views, Freq, Time)
        Output: V_agg (Batch, K_subbands, Time, 2*P)
        """
        nb, N, nf, nt = V.shape
        
        # Permute to (B, T, N, F) for easier slicing
        V = V.permute(0, 3, 1, 2) 
        
        curr_freq = 0
        agg_features = []
        
        # Iterate over subbands
        for k, bw in enumerate(self.band_specs):
            bw = int(bw)
            
            # 1. Slice: (B, T, N, BW_k)
            v_band = V[..., curr_freq : curr_freq + bw]
            curr_freq += bw
            
            # 2. Project: (B, T, N, BW_k) -> (B, T, N, P)
            v_proj = self.subband_projs[k](v_band)
            
            # 3. Prepare for RNN: Merge Batch & Time -> (B*T, N, P)
            v_flat = v_proj.reshape(nb * nt, N, -1)
            
            # 4. RNN-Loop Strategy [cite: 201]
            # 取序列的第一个元素，拼接到末尾形成闭环
            # v_flat[:, 0:1, :] shape is (B*T, 1, P)
            v_loop = torch.cat([v_flat, v_flat[:, 0:1, :]], dim=1) # (B*T, N+1, P)
            
            # 5. LSTM Forward
            self.lstm.flatten_parameters() # 内存连续化，加速
            out, _ = self.lstm(v_loop) # (B*T, N+1, P)
            
            # 6. Aggregation: Concatenate last 2 steps [cite: 202]
            # out[:, -2:, :] -> (B*T, 2, P)
            v_aggregated = out[:, -2:, :].reshape(nb * nt, -1) # (B*T, 2P)
            
            # 7. Reshape back: (B, T, 2P)
            v_agg_k = v_aggregated.reshape(nb, nt, -1)
            agg_features.append(v_agg_k.unsqueeze(1)) # (B, 1, T, 2P)
            
        # Concat subbands: (B, K, T, 2P)
        V_agg = torch.cat(agg_features, dim=1)
        
        return V_agg