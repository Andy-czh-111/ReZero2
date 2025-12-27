import torch
import torch.nn as nn

class ReZeroPaperLoss(nn.Module):
    """
    实现论文 Equation (5):
    - 当 Q=0 时: L1 Loss (Magnitude Spectrum) -> 迫使输出静音
    - 当 Q>0 时: SNR Loss (Time Domain) -> 迫使输出匹配目标
    """
    def __init__(self, lambda_silence=0.01, epsilon=1e-8):
        super().__init__()
        self.lambda_silence = lambda_silence # 论文提及 empirically set to 0.01 [cite: 357]
        self.epsilon = epsilon

    def si_snr(self, est, ref):
        # Scale-Invariant SNR 计算
        # est, ref: (B, T)
        # Zero-mean normalization (important for SI-SNR)
        est_z = est - est.mean(dim=-1, keepdim=True)
        ref_z = ref - ref.mean(dim=-1, keepdim=True)

        # Projection of est on ref
        proj = (torch.sum(est_z * ref_z, dim=-1, keepdim=True) /
            (torch.sum(ref_z ** 2, dim=-1, keepdim=True) + self.epsilon)) * ref_z
        e_noise = est_z - proj

        # SI-SNR in dB
        si_snr_val = 10.0 * torch.log10((torch.sum(proj ** 2, dim=-1) + self.epsilon) /
                        (torch.sum(e_noise ** 2, dim=-1) + self.epsilon))
        return si_snr_val
    
    def forward(self, est_wav, target_wav, est_stft, Q):
        """
        est_wav: (B, T) 估计的时域波形
        target_wav: (B, T) 目标时域波形 (Q=0时应全为0)
        est_stft: (B, F, T) 估计的复数谱 (用于Q=0的约束)
        Q: (B,) 区域内目标数量
        """
        batch_size = est_wav.shape[0]

        # Prepare per-sample loss vector
        device = est_wav.device
        loss_per_sample = torch.zeros(batch_size, device=device)

        mask_silence = (Q == 0)
        mask_active = (Q > 0)

        # Active samples: -SI-SNR (we want to maximize SI-SNR -> minimize -SI-SNR)
        if mask_active.any():
            est_active = est_wav[mask_active]
            ref_active = target_wav[mask_active]
            snr_val = self.si_snr(est_active, ref_active)  # (n_active,)
            loss_per_sample[mask_active] = -snr_val

        # Silence samples: spectral L1 (mean over elements per sample)
        if mask_silence.any():
            est_stft_silence = est_stft[mask_silence]  # (n_sil, F, T) complex
            # Use magnitude (abs of complex) and L1 mean per-sample
            mag = torch.abs(est_stft_silence)
            per_sample_mean = mag.reshape(mag.size(0), -1).mean(dim=1)
            loss_per_sample[mask_silence] = self.lambda_silence * per_sample_mean

        # Final loss: mean over batch (gives proper weighting by sample counts)
        total_loss = torch.mean(loss_per_sample)
        return total_loss