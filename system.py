import torch
import torch.nn as nn
# ISTFT 在训练/验证脚本中由外部调用，这里不在模块内部执行重建。

from .feature_extraction import CalculateSpatialFeatures, CalculateDirectionFeature
from .feature_aggregation import RegionFeatureAggregator
from .region_sampler import RegionSampler
from .bsrnn import ReZeroBSRNN

class ReZeroSystem(nn.Module):
    """
    ReZero 完整系统实现
    论文参考: 
    - Fig 2 (Pipeline Overview)
    - Section IV-C & IV-D (Distance/Direction Logic)
    - Section VI-C (Cascade: A-ReZero -> D-ReZero)
    """
    def __init__(self, 
                 mic_locations, 
                 win_len=512, 
                 win_shift_ratio=0.25, 
                 nfft=512, 
                 bsrnn_channels=64, 
                 P_dim=16, 
                 task_type='cascade'):
        """
        Args:
            mic_locations: (M, 3) 麦克风坐标
            win_len: STFT 窗长
            win_shift_ratio: 帧移比例 (0.25 -> hop_len = win_len/4)
            nfft: FFT 点数
            bsrnn_channels: BSRNN 隐藏层维度
            P_dim: 区域特征聚合后的维度 (2*P_dim 会作为 A-ReZero 的条件维度)
            task_type: 'angle', 'distance', 或 'cascade' (默认)
        """
        super().__init__()

        self.task_type = task_type.lower()
        num_mics = mic_locations.shape[0]
        # 计算麦克风对数 (MM mode: C(M, 2))
        self.num_pairs = num_mics * (num_mics - 1) // 2
        self.nfft = nfft
        
        # 1. 基础特征提取器 (Spatial & STFT)
        self.spatial_extractor = CalculateSpatialFeatures(
            win_len, win_shift_ratio, nfft, ch_mode='MM'
        )
        
        # 注意: ISTFT 不在此模块内部执行，训练/验证代码会调用 ISTFT。

        # 3. 区域采样器 (所有模式都需要)
        self.sampler = RegionSampler(sampling_mode='fixed_number', num_views_N=8)

        # 4. 初始化 A-ReZero 组件 (Angle or Cascade)
        if self.task_type in ['angle', 'cascade']:
            self.direction_extractor = CalculateDirectionFeature(
                mic_locations, nfft, ch_mode='MM'
            )
            self.aggregator = RegionFeatureAggregator(
                hidden_dim_P=P_dim, fs=16000, nfft=nfft
            )
            # A-ReZero BSRNN
            self.separator_angle = ReZeroBSRNN(
                num_channel=bsrnn_channels, 
                num_layer=6, 
                condition_type='angle',
                spatial_dim=self.num_pairs,    
                condition_dim=2 * P_dim, # RNN-Loop 输出维度是 2*P      
                nfft=nfft,
                rnn_dropout=0.1
            )
        
        # 5. 初始化 D-ReZero 组件 (Distance or Cascade)
        if self.task_type in ['distance', 'cascade']:
            # D-ReZero BSRNN
            self.separator_dist = ReZeroBSRNN(
                num_channel=bsrnn_channels, 
                num_layer=6, 
                condition_type='distance',
                spatial_dim=self.num_pairs,   
                condition_dim=0, # Distance 模式内部处理维度，这里传 0 即可
                nfft=nfft,
                rnn_dropout=0.1
            )

    def forward(self, audio_multichannel, region_params):
        """
        前向传播
        
        Args:
            audio_multichannel: (Batch, Time, Channels) 或 (Batch, Channels, Time)
            region_params: 字典，包含 'azi_low', 'azi_high', 'dist_low', 'dist_high' 等 Tensor
            
        Returns:
            est_stft: (Batch, F, T) 估计的复数谱（caller 负责 ISTFT 重建）
        """
        # 0. 输入维度预处理 (统一转为 Batch, Ch, Time)
        # CalculateSpatialFeatures 期望输入的 dim=1 是 Channel 还是 Time?
        # 查看 feature_extraction.py，它调用 STFT，通常 STFT 期望 (B, Time) 或 (B, Ch, Time)
        # 假设 audio_multichannel 是 (B, Time, Ch) -> 需要 permute
        if audio_multichannel.dim() == 3 and audio_multichannel.shape[-1] < audio_multichannel.shape[-2]:
             # 简单的启发式判断：如果最后一维是通道数 (通常 4-8)，倒数第二维是时间
            audio_multichannel = audio_multichannel.permute(0, 2, 1)
        
        # 记录原始长度用于 ISTFT 对齐
        original_len = audio_multichannel.shape[-1]

        # A. 提取基础特征
        # ipd_flat: (B*Pairs, F, T)
        # ild_flat: (B*Pairs, F, T)
        # stft_complex: (B, Ch, F, T)
        ipd_flat, ild_flat, stft_complex = self.spatial_extractor(audio_multichannel)
        
        stft_complex = stft_complex.permute(0, 3, 1, 2).contiguous()
        
        # 取第 0 通道作为 Reference STFT
        ref_stft = stft_complex[:, 0, :, :] 
        nb, nf, nt = ref_stft.shape
        
        # Reshape 为网络输入格式: (B, Pairs, F, T)
        ipd = ipd_flat.reshape(nb, self.num_pairs, nf, nt)
        ild = ild_flat.reshape(nb, self.num_pairs, nf, nt)

        # B. 区域参数采样
        # s_azi, s_ele: (B, N)
        # s_dist: (B, 2) -> [low, high]
        s_azi, s_ele, s_dist = self.sampler(
            region_params['azi_low'], region_params['azi_high'],
            region_params['ele_low'], region_params['ele_high'],
            region_params['dist_low'], region_params['dist_high']
        )
        
        # 初始化估计谱 (Cascade 模式会覆盖)
        est_stft = ref_stft

        # === Stage 1: Angle Separation (A-ReZero) ===
        if self.task_type in ['angle', 'cascade']:
            # 1. 计算方向特征图 V: (B, N, F, T)
            V = self.direction_extractor(ipd_flat, s_azi, s_ele)
            
            # 2. 聚合特征: (B, K, T, 2P)
            # 使用 RNN-Loop 将 N 个视角的特征聚合
            V_agg = self.aggregator(V)
            
            # 3. 执行分离
            # 输入: Ref STFT, IPD, Region Features
            est_stft = self.separator_angle(
                ref_stft, 
                spatial_feature=ipd, 
                condition_input=V_agg
            )

        # === Stage 2: Distance Separation (D-ReZero) ===
        if self.task_type in ['distance', 'cascade']:
            # D-ReZero 逻辑: Output = Mask(High) - Mask(Low)
            # 输入信号: 如果是 Cascade，基于 Stage 1 的输出；否则基于 Ref STFT
            input_stft = est_stft 
            
            # 优化：并行计算 Batch 内的 Low 和 High 两个条件的 Mask
            
            # 1. 准备距离条件
            d_low = s_dist[:, 0]  # (B,)
            d_high = s_dist[:, 1] # (B,)
            # 拼接: (2B,) -> [Batch_Low..., Batch_High...]
            d_concat = torch.cat([d_low, d_high], dim=0)
            
            # 2. 准备输入特征 (在 Batch 维复制)
            # input_stft: (B, F, T) -> (2B, F, T)
            input_stft_concat = torch.cat([input_stft, input_stft], dim=0)
            # spatial (ILD): (B, P, F, T) -> (2B, P, F, T)
            ild_concat = torch.cat([ild, ild], dim=0)
            
            # 3. 并行推理 (一次 Forward 处理 2*B 个样本)
            est_dist_concat = self.separator_dist(
                input_stft_concat, 
                spatial_feature=ild_concat, 
                condition_input=d_concat
            )
            
            # 4. 拆分结果
            est_low, est_high = torch.split(est_dist_concat, nb, dim=0)
            
            # 5. 减法操作 (Section IV-D)
            # 目标区域信号 = 范围[0, High] - 范围[0, Low]
            est_stft = est_high - est_low

        # 返回估计的复数谱，caller（训练/验证脚本）负责调用 ISTFT 以获得时域波形
        return est_stft