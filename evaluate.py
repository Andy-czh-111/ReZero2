import numpy as np
import math

# 尝试导入高级指标库；如果不可用，使用回退实现
_HAS_MIR = False
_HAS_STOI = False
_HAS_PESQ = False
try:
    from mir_eval.separation import bss_eval_sources
    _HAS_MIR = True
except Exception:
    _HAS_MIR = False

try:
    from pystoi import stoi
    _HAS_STOI = True
except Exception:
    _HAS_STOI = False

try:
    from pesq import pesq
    _HAS_PESQ = True
except Exception:
    _HAS_PESQ = False


def _fallback_sdr(est, ref):
    """简单的 SNR 近似（每帧或整段）作为 SDR 回退：10*log10(sum(ref^2) / sum((ref-est)^2))."""
    eps = 1e-9
    num = np.sum(ref ** 2) + eps
    den = np.sum((ref - est) ** 2) + eps
    return 10.0 * np.log10(num / den)


def _fallback_stoi(est, ref, fs):
    # 无法精确实现 STOI，这里返回 0.0 表示不可用
    return 0.0


def _fallback_pesq(est, ref, fs):
    # 无法精确实现 PESQ，这里返回 1.0 (最低) 作为回退
    return 1.0


def compute_metrics(est_wav, target_wav, mix_wav, Q, fs=16000):
    """Compute metrics with graceful fallback when external libs are missing.

    All inputs expected as 1-D numpy arrays (samples,).
    """
    metrics = {}
    epsilon = 1e-9

    if Q == 0:
        energy_in = np.sum(mix_wav ** 2) + epsilon
        energy_out = np.sum(est_wav ** 2) + epsilon
        decay = 10 * np.log10(energy_out / energy_in)
        metrics['decay'] = float(decay)
        # also provide simple output energy
        metrics['energy_out_db'] = 10 * np.log10(energy_out + epsilon)

    else:
        # SDR
        if _HAS_MIR:
            try:
                sdr, _, _, _ = bss_eval_sources(target_wav[None, :], est_wav[None, :], compute_permutation=False)
                metrics['sdr'] = float(sdr[0])
            except Exception:
                metrics['sdr'] = float(_fallback_sdr(est_wav, target_wav))
        else:
            metrics['sdr'] = float(_fallback_sdr(est_wav, target_wav))

        # STOI
        if _HAS_STOI:
            try:
                metrics['stoi'] = float(stoi(target_wav, est_wav, fs, extended=False))
            except Exception:
                metrics['stoi'] = _fallback_stoi(est_wav, target_wav, fs)
        else:
            metrics['stoi'] = _fallback_stoi(est_wav, target_wav, fs)

        # PESQ
        if _HAS_PESQ:
            try:
                metrics['pesq'] = float(pesq(fs, target_wav, est_wav, 'wb'))
            except Exception:
                metrics['pesq'] = _fallback_pesq(est_wav, target_wav, fs)
        else:
            metrics['pesq'] = _fallback_pesq(est_wav, target_wav, fs)

    return metrics