import torch
import torch.nn as nn
from .bsrnn_layers import BandSplit, MaskDecoder, get_band_specs

class SubbandDEG(nn.Module):
    """
    D-ReZero 专用: 距离嵌入生成器 (Distance Embedding Generator)
    论文参考: Section IV-C-2 [cite: 139]
    """
    def __init__(self, num_subbands, embedding_dim=128):
        super().__init__()
        # 为每个子带创建独立的 MLP，不共享参数
        self.degs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, embedding_dim), nn.Tanh(),
                nn.Linear(embedding_dim, embedding_dim), nn.Tanh(),
                nn.Linear(embedding_dim, embedding_dim)
            ) for _ in range(num_subbands)
        ])

    def forward(self, d):
        # d: (B,) or (B, 1)
        if d.dim() == 1: d = d.unsqueeze(-1)
        
        embeddings = []
        for deg in self.degs:
            # (B, 1) -> (B, Emb_Dim) -> (B, 1, Emb_Dim)
            e = deg(d).unsqueeze(1)
            embeddings.append(e)
            
        return torch.cat(embeddings, dim=1) # (B, K, Emb_Dim)

class ReZeroBSRNN(nn.Module):
    """
    ReZero BSRNN 主架构
    论文参考: Fig 6 & Fig 7 [cite: 208, 279]
    """
    def __init__(self, 
                 num_channel=128, 
                 num_layer=8, 
                 condition_type='angle', # 'angle' or 'distance'
                 spatial_dim=28,         # MM mode: C(8,2)=28 pairs
                 condition_dim=32,       # Region descriptor dim (2*P)
                 fs=16000, 
                 nfft=512,
                 rnn_type='lstm',        # 'lstm' or 'gru'
                 rnn_dropout=0.0):
        super(ReZeroBSRNN, self).__init__()
        
        self.num_layer = num_layer
        self.condition_type = condition_type
        
        # 1. 频带划分
        self.band_specs = get_band_specs(fs, nfft)
        num_subbands = len(self.band_specs)
        
        # 2. 输入特征 BandSplit
        # Stream 1: Spectral (Real, Imag) -> input_dim=2
        self.bs_spec = BandSplit(channels=num_channel, fs=fs, nfft=nfft, input_dim=2)
        
        # Stream 2: Spatial (IPD or ILD) -> input_dim=spatial_dim
        self.bs_spatial = BandSplit(channels=num_channel, fs=fs, nfft=nfft, input_dim=spatial_dim)
        
        # Stream 3: Condition (Region Features)
        if condition_type == 'distance':
            # D-ReZero: DEG -> Expand
            self.subband_deg = SubbandDEG(num_subbands, embedding_dim=num_channel)
        elif condition_type == 'angle':
            # A-ReZero: Region Feature Projection
            # 优化: 使用 Conv1d(kernel=1) 代替 Linear，减少 forward 中的 permute 操作
            # 论文: Norm + FC [cite: 252]
            # 使用 Conv1d 作为子带条件投影，同时使用 LayerNorm 做通道归一化以稳定小 batch
            self.cond_proj = nn.ModuleList([
                nn.Conv1d(condition_dim, num_channel, kernel_size=1) for _ in range(num_subbands)
            ])
            self.cond_ln = nn.ModuleList([nn.LayerNorm(condition_dim) for _ in range(num_subbands)])

        # 3. BSRNN Layers
        for i in range(self.num_layer):
            # Time Axis: Uni-directional RNN (LSTM or GRU)
            if rnn_type.lower() == 'gru':
                setattr(self, f'lstm_t{i+1}', nn.GRU(num_channel, num_channel, batch_first=True, bidirectional=False))
            else:
                setattr(self, f'lstm_t{i+1}', nn.LSTM(num_channel, num_channel, batch_first=True, bidirectional=False))
            setattr(self, f'norm_t{i+1}', nn.LayerNorm(num_channel))
            setattr(self, f'fc_t{i+1}', nn.Linear(num_channel, num_channel))
            # ReZero scaling for time residual
            setattr(self, f'alpha_t{i+1}', nn.Parameter(torch.zeros(1)))
            # optional dropout after time RNN
            setattr(self, f'drop_t{i+1}', nn.Dropout(rnn_dropout))
            
            # Frequency Axis: Bi-directional LSTM
            if rnn_type.lower() == 'gru':
                setattr(self, f'lstm_k{i+1}', nn.GRU(num_channel, num_channel, batch_first=True, bidirectional=True))
            else:
                setattr(self, f'lstm_k{i+1}', nn.LSTM(num_channel, num_channel, batch_first=True, bidirectional=True))
            setattr(self, f'norm_k{i+1}', nn.LayerNorm(num_channel))
            setattr(self, f'fc_k{i+1}', nn.Linear(num_channel * 2, num_channel))
            # ReZero scaling for frequency residual
            setattr(self, f'alpha_k{i+1}', nn.Parameter(torch.zeros(1)))
            setattr(self, f'drop_k{i+1}', nn.Dropout(rnn_dropout))

        # 4. Mask Decoder
        self.mask_decoder = MaskDecoder(channels=num_channel, fs=fs, nfft=nfft)

    def forward(self, x_audio_complex, spatial_feature, condition_input):
        """
        x_audio_complex: (B, F, T)
        spatial_feature: (B, P, F, T) [IPD for A-ReZero, ILD for D-ReZero]
        condition_input: 
            - angle mode: (B, K, T, Cond_Dim) [aggregated region features]
            - distance mode: (B,) or (B, 1) [distance scalar]
        """
        # (B, F, T) -> (B, 2, F, T) -> BandSplit 需要 (B, In, F, T)
        x_real = torch.view_as_real(x_audio_complex).permute(0, 3, 1, 2)
        
        # --- Stream 1: Spectral ---
        z_spec = self.bs_spec(x_real) # (B, K, T, C)
        B, K, T, C = z_spec.shape
        
        # --- Stream 2: Spatial ---
        z_spatial = self.bs_spatial(spatial_feature) # (B, K, T, C)
        
        # --- Stream 3: Condition ---
        z_cond = 0
        if self.condition_type == 'distance':
            # 1. Generate Embeddings: (B, K, C)
            deg_out = self.subband_deg(condition_input)
            # 2. Expand to Time: (B, K, 1, C) -> (B, K, T, C)
            z_cond = deg_out.unsqueeze(2).expand(-1, -1, T, -1)
            
        elif self.condition_type == 'angle':
            # condition_input: (B, K, T, Cond_Dim)
            z_cond_list = []
            for k in range(len(self.band_specs)):
                # 取出第 k 个子带特征: (B, T, Cond_Dim)
                c_in = condition_input[:, k, :, :]
                # 先做 LayerNorm 在时间轴上 (B, T, Cond_Dim)
                c_ln = self.cond_ln[k](c_in)
                # 变换为 (B, Cond_Dim, T) 以适配 Conv1d
                c_for_conv = c_ln.permute(0, 2, 1)

                # Projection: (B, C, T)
                out = self.cond_proj[k](c_for_conv)

                # 恢复维度: (B, 1, T, C)
                z_cond_list.append(out.permute(0, 2, 1).unsqueeze(1))
            
            z_cond = torch.cat(z_cond_list, dim=1) # (B, K, T, C)
            
        # --- Fusion ---
        z = z_spec + z_spatial + z_cond # Element-wise Sum [cite: 208]

        # --- BSRNN Processing ---
        # Reshape helper: (B, K, T, C)
        for i in range(self.num_layer):
            # 1. Time Block
            residual = z
            # Merge Batch & Band: (B*K, T, C)
            z_t_in = z.view(B * K, T, C)
            # BN: (B*K, C, T) -> need permute for BatchNorm1d if using 1d input
            # Code uses BatchNorm1d(num_channel). Input should be (N, C) or (N, C, L)
            # Efficient implementation: (B*K*T, C) -> Norm -> Reshape
            z_t_norm = getattr(self, f'norm_t{i+1}')(z_t_in.reshape(-1, C)).reshape(B * K, T, C)

            z_t_out, _ = getattr(self, f'lstm_t{i+1}')(z_t_norm)
            z_t_proj = getattr(self, f'fc_t{i+1}')(z_t_out)
            alpha_t = getattr(self, f'alpha_t{i+1}')
            z = residual + alpha_t * z_t_proj.view(B, K, T, C)
            
            # 2. Frequency Block
            residual = z
            # Merge Batch & Time: (B*T, K, C)
            z_k_in = z.permute(0, 2, 1, 3).contiguous().view(B * T, K, C)
            
            z_k_norm = getattr(self, f'norm_k{i+1}')(z_k_in.reshape(-1, C)).reshape(B * T, K, C)

            z_k_out, _ = getattr(self, f'lstm_k{i+1}')(z_k_norm)
            z_k_proj = getattr(self, f'fc_k{i+1}')(z_k_out)

            # Restore: (B*T, K, C) -> (B, T, K, C) -> (B, K, T, C)
            alpha_k = getattr(self, f'alpha_k{i+1}')
            z = residual + alpha_k * z_k_proj.view(B, T, K, C).permute(0, 2, 1, 3)
            
        # --- Mask Estimation ---
        # MaskDecoder 现在输出实值幅度掩码 m_mag: (B, F, T)
        m_mag = self.mask_decoder(z)  # (B, F, T)

        # 直接用幅度掩码乘以复数谱（保留参考相位）
        # est_stft = m_mag * x_audio_complex
        # ensure broadcasting works: m_mag (B,F,T) -> (B,F,T)
        return m_mag * x_audio_complex