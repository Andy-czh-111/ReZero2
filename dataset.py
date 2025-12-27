import torch
from torch.utils.data import Dataset
import numpy as np
import pyroomacoustics as pra
import random
import librosa
import os

class ReZeroOnTheFlyDataset(Dataset):
    def __init__(self, speech_list, noise_list, fs=16000, duration=4.0,
                 cache_dir=None, use_cache=False):
        """
        初始化ReZero实时数据集生成器
        
        Args:
            speech_list: 语音文件路径列表，用于生成目标语音信号
            noise_list: 噪声文件路径列表，用于生成干扰噪声信号
            fs: 采样率，默认16000Hz
            duration: 音频时长，默认4.0秒
        """
        # 音频参数设置
        self.fs = fs  # 采样率 (Hz)
        self.duration = int(duration * fs)  # 音频样本点数 = 时长(秒) × 采样率
        
        # 音频文件列表
        self.speech_files = speech_list  # 语音文件路径列表
        self.noise_files = noise_list    # 噪声文件路径列表

        # 缓存/预计算支持
        self.cache_dir = cache_dir
        self.use_cache = bool(use_cache and cache_dir is not None)
        self.cached_files = []
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            # 查找已有的 npz 缓存文件
            for fn in sorted(os.listdir(self.cache_dir)):
                if fn.endswith('.npz'):
                    self.cached_files.append(os.path.join(self.cache_dir, fn))
        
        # 麦克风阵列几何配置
        # 8麦克风圆阵，半径 0.025m 
        self.R = 0.025  # 麦克风阵列半径 (米)
        
        # PyRoomAcoustics 坐标系: (3, M)
        # 生成8个麦克风的圆形阵列位置坐标
        # 每个麦克风位置: [R*cos(2πi/8), R*sin(2πi/8), 0.0]
        # 转置后形状为 (3, 8)，符合PyRoomAcoustics的坐标格式要求
        self.mic_pos = np.array([[self.R*np.cos(2*np.pi*i/8), 
                                  self.R*np.sin(2*np.pi*i/8), 
                                  0.0] for i in range(8)]).T

    def __len__(self):
        """
        返回数据集的样本数量
        
        注意：这是一个实时生成的数据集，每次调用__getitem__都会动态生成新的样本。
        返回的2000表示每一轮训练中生成的样本数量，而不是预先存储的样本总数。
        
        Returns:
            int: 数据集大小，固定为2000个样本
        """
        if self.use_cache and len(self.cached_files) > 0:
            return len(self.cached_files)
        return 2000  # 每一轮生成的样本数


    def generate_sample(self):
        """Generate one sample and return numpy arrays (mix, target, region, Q).

        This extracts the core of __getitem__ so it can be reused by a
        precompute script or by multiprocessing workers.
        """
        # We reuse the logic from __getitem__ but return numpy arrays.
        while True:
            try:
                # === copy of generation logic from __getitem__ ===
                p = random.random()
                if p < 0.27: Q_target = 0
                elif p < 0.92: Q_target = 1
                else: Q_target = 2

                L = np.random.uniform(3, 10)
                W = np.random.uniform(3, 8)
                H = np.random.uniform(2.5, 4)
                rt60 = np.random.uniform(0.05, 0.25)
                e_absorption, max_order = pra.inverse_sabine(rt60, [L, W, H])
                room = pra.ShoeBox([L, W, H], fs=self.fs,
                                  materials=pra.Material(e_absorption),
                                  max_order=min(max_order, 10))
                center_x = np.random.uniform(0.5, L-0.5)
                center_y = np.random.uniform(0.5, W-0.5)
                mic_absolute_pos = self.mic_pos + np.array([[center_x], [center_y], [0.7]])
                room.add_microphone_array(mic_absolute_pos)

                azi_center = np.random.uniform(-np.pi, np.pi)
                azi_width = np.deg2rad(np.random.uniform(30, 90))
                def wrap_angle(a): return (a + np.pi) % (2 * np.pi) - np.pi
                region_low = wrap_angle(azi_center - azi_width/2)
                region_high = wrap_angle(azi_center + azi_width/2)
                def is_in_region(ang, low, high):
                    ang = wrap_angle(ang)
                    diff = wrap_angle(ang - azi_center)
                    return abs(diff) <= azi_width / 2

                sources_added = 0
                target_signals_early = []

                total_speech = 2 if Q_target == 2 else random.randint(1, 2)
                speech_audios = self.load_audio(self.speech_files, total_speech)
                current_q = 0
                retry_limit = 50
                for i_sig, sig in enumerate(speech_audios):
                    placed = False
                    for _ in range(retry_limit):
                        src_x = np.random.uniform(0.5, L-0.5)
                        src_y = np.random.uniform(0.5, W-0.5)
                        src_z = np.random.uniform(1.0, 1.8)
                        dx = src_x - center_x
                        dy = src_y - center_y
                        azi = np.arctan2(dy, dx)
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist < 0.2 or dist > 2.0: continue
                        in_region = is_in_region(azi, region_low, region_high)
                        should_be_target = (current_q < Q_target)
                        if should_be_target and in_region:
                            room.add_source([src_x, src_y, src_z], signal=sig)
                            target_signals_early.append((len(room.sources)-1, sig))
                            current_q += 1
                            placed = True
                            break
                        elif not should_be_target and not in_region:
                            room.add_source([src_x, src_y, src_z], signal=sig)
                            placed = True
                            break
                    if not placed:
                        raise ValueError("Failed to place sources satisfying Q constraint")

                if current_q != Q_target:
                     raise ValueError(f"Failed to meet Q target: got {current_q}, expected {Q_target}")

                num_noise = random.randint(1, 2)
                if len(self.noise_files) > 0:
                    noise_audios = self.load_audio(self.noise_files, num_noise)
                    for sig in noise_audios:
                        src_x = np.random.uniform(0.5, L-0.5)
                        src_y = np.random.uniform(0.5, W-0.5)
                        src_z = np.random.uniform(0.5, 2.0)
                        room.add_source([src_x, src_y, src_z], signal=sig)

                room.compute_rir()

                target_wav = np.zeros(self.duration)
                if Q_target > 0:
                    for src_idx, sig in target_signals_early:
                        rir = room.rir[0][src_idx]
                        peak_idx = np.argmax(np.abs(rir))
                        start = max(0, peak_idx - 96)
                        end = min(len(rir), peak_idx + 800)
                        rir_early = np.zeros_like(rir)
                        rir_early[start:end] = rir[start:end]
                        conv = np.convolve(sig, rir_early)[:self.duration]
                        target_wav += conv

                room.simulate()
                mix_multichannel = room.mic_array.signals.T
                mix_multichannel = mix_multichannel[:self.duration, :]
                max_val = np.max(np.abs(mix_multichannel)) + 1e-8
                mix_multichannel /= max_val
                target_wav /= max_val

                region_params = {
                    'azi_low': float(region_low),
                    'azi_high': float(region_high),
                    'ele_low': 0.0,
                    'ele_high': float(np.pi/2),
                    'dist_low': 0.0,
                    'dist_high': 2.0
                }

                return {
                    'mix': mix_multichannel.astype(np.float32),
                    'target': target_wav.astype(np.float32),
                    'region': region_params,
                    'Q': int(Q_target)
                }

            except ValueError:
                continue
            except Exception as e:
                print(f"Unexpected error in generation: {e}")
                continue

    def load_audio(self, files, count):
        """
        从文件列表中加载指定数量的音频文件，并进行预处理
        
        Args:
            files: 音频文件路径列表
            count: 需要加载的音频文件数量
            
        Returns:
            list: 处理后的音频信号列表，每个信号长度为self.duration
        """
        signals = []  # 存储处理后的音频信号
        
        # 防止文件不够：根据文件数量选择合适的采样策略
        if len(files) < count:
            # 文件数量不足时，允许重复选择
            selected = random.choices(files, k=count)  # 允许重复
        else:
            # 文件数量足够时，随机选择不重复的文件
            selected = random.sample(files, k=count)
            
        # 逐个加载和处理音频文件
        for f in selected:
            try:
                # 加载音频文件，使用指定的采样率
                y, _ = librosa.load(f, sr=self.fs)
                
                # 归一化音量，防止过大或过小
                if np.max(np.abs(y)) > 0:
                    y = y / np.max(np.abs(y))  # 归一化到[-1, 1]范围
                    
                # 音频长度处理
                if len(y) > self.duration:
                    # 音频过长：随机截取指定长度的片段
                    start = random.randint(0, len(y) - self.duration)
                    y = y[start:start+self.duration]
                else:
                    # 音频过短：用零填充到指定长度
                    y = np.pad(y, (0, self.duration - len(y)))
                    
                signals.append(y)  # 将处理后的音频添加到结果列表
                
            except Exception as e:
                # 异常处理：如果文件加载失败，生成替代音频防止训练崩溃
                print(f"Error loading {f}: {e}")
                # 生成一个小的随机噪声作为替代，避免完全静音
                signals.append(np.random.uniform(-0.01, 0.01, size=self.duration))
                
        return signals  # 返回处理后的音频信号列表

    def __getitem__(self, idx):
        """
        动态生成一个训练样本，包含完整的房间声学仿真过程
        
        Args:
            idx: 样本索引（在实时生成数据集中仅用于接口兼容性）
            
        Returns:
            dict: 包含以下键的字典：
                - 'mix': 多通道混合信号 (duration, 8)
                - 'target': 目标信号 (duration,)
                - 'region': 查询区域参数
                - 'Q': 目标语音数量标签
        """
        # 如果启用了缓存且有缓存文件，直接从缓存加载
        if self.use_cache and len(self.cached_files) > 0:
            fn = self.cached_files[idx]
            data = np.load(fn)
            mix = data['mix']
            target = data['target']
            region = {
                'azi_low': torch.tensor(float(data['region_azi_low'])).float(),
                'azi_high': torch.tensor(float(data['region_azi_high'])).float(),
                'ele_low': torch.tensor(0.0).float(),
                'ele_high': torch.tensor(np.pi/2).float(),
                'dist_low': torch.tensor(0.0).float(),
                'dist_high': torch.tensor(2.0).float()
            }
            return {
                'mix': torch.from_numpy(mix).float(),
                'target': torch.from_numpy(target).float(),
                'region': region,
                'Q': torch.tensor(int(data['Q']))
            }

        # 否则在线生成样本
        sample = self.generate_sample()
        return {
            'mix': torch.from_numpy(sample['mix']).float(),
            'target': torch.from_numpy(sample['target']).float(),
            'region': {
                'azi_low': torch.tensor(sample['region']['azi_low']).float(),
                'azi_high': torch.tensor(sample['region']['azi_high']).float(),
                'ele_low': torch.tensor(sample['region']['ele_low']).float(),
                'ele_high': torch.tensor(sample['region']['ele_high']).float(),
                'dist_low': torch.tensor(sample['region']['dist_low']).float(),
                'dist_high': torch.tensor(sample['region']['dist_high']).float()
            },
            'Q': torch.tensor(int(sample['Q']))
        }