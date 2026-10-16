# -*- coding: utf-8 -*-
"""
ComfyUI-DJ_AudioPad · 音频加载 + 补齐整数秒节点

关键约束 & 踩坑记录：
1. 用传统 ComfyUI 节点框架（NODE_CLASS_MAPPINGS），和 DJ_Batch 一致。
   IO.ComfyNode 框架的节点在自定义插件里注册有问题，拖不进工作流。
2. 文件选择下拉 + 上传 + 播放条 UI 由前端自动提供：
   - COMBO 类型输入 → 下拉选择
   - upload 参数 → 上传按钮
   - AUDIO 输出 → 播放条预览
3. 核心逻辑：在音频尾部追加静音，把总时长凑成 ceil(原时长) 整数秒。
   - 2.3s → 补 0.7s → 输出 3s
   - 5.01s → 补 0.99s → 输出 6s
   - 5.0s（已是整数）→ 不补，原样输出 5s
   **不是截断、不是抹零，是加法补齐！**
4. 硬指标：输出时长必须是整数秒。万一浮点误差导致 ceil 后仍非整数，
   直接按 int(target_seconds) * sr 截断采样点数，确保输出严格整数秒。
5. AUDIO 格式：{"waveform": [B, C, S] float32 tensor, "sample_rate": int}
   - batch 维固定为 1（单个文件）
   - key 用 "sample_rate"（SoundFlow 等下游节点检查的拼写）
6. 解码用 PyAV（import av），和 ComfyUI 官方 LoadAudio 一致。
7. 原音频内容一字不动，只在尾部追加零值采样。
8. 采样率、声道数全部继承原音频，只动时长。
"""

import math
import os
import hashlib

import torch
import av
import folder_paths


def f32_pcm(wav: torch.Tensor) -> torch.Tensor:
    """Convert audio to float 32 bits PCM format.（复制自官方 nodes_audio.py）"""
    if wav.dtype.is_floating_point:
        return wav
    elif wav.dtype == torch.int16:
        return wav.float() / (2 ** 15)
    elif wav.dtype == torch.int32:
        return wav.float() / (2 ** 31)
    raise ValueError(f"Unsupported wav dtype: {wav.dtype}")


def load_audio(filepath: str) -> tuple[torch.Tensor, int]:
    """
    用 PyAV 解码音频文件（复制自官方 nodes_audio.py 的 load 函数）。
    返回 (waveform, sample_rate)：
      waveform: [channels, samples] float32 tensor
      sample_rate: int
    """
    with av.open(filepath) as af:
        if not af.streams.audio:
            raise ValueError("No audio stream found in the file.")

        stream = af.streams.audio[0]
        sr = stream.codec_context.sample_rate
        n_channels = stream.channels

        frames = []
        for frame in af.decode(streams=stream.index):
            buf = torch.from_numpy(frame.to_ndarray())
            if buf.shape[0] != n_channels:
                buf = buf.view(-1, n_channels).t()
            frames.append(buf)

        if not frames:
            raise ValueError("No audio frames decoded.")

        wav = torch.cat(frames, dim=1)
        wav = f32_pcm(wav)
        return wav, sr


def pad_to_integer_seconds(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """
    核心补齐逻辑：在尾部追加静音，把总时长凑成 ceil(原时长) 整数秒。

    参数：
      waveform: [B, C, S] float32 tensor
      sample_rate: int

    返回：
      padded_waveform: [B, C, S'] float32 tensor，其中 S' 是整数秒对应的采样点数
    """
    total_samples = waveform.shape[-1]

    # 计算目标秒数：向上取整到整数秒
    # 用 math.ceil 避免浮点误差（比如 5.0000001 → 6 而不是 5）
    target_seconds = math.ceil(total_samples / sample_rate)

    # 目标采样点数
    target_samples = target_seconds * sample_rate

    # 需要补齐的采样点数
    pad_samples = target_samples - total_samples

    if pad_samples > 0:
        # 创建零值 tensor，shape 和 waveform 一致，只有最后一维是 pad_samples
        pad_shape = list(waveform.shape)
        pad_shape[-1] = pad_samples
        padding = torch.zeros(pad_shape, dtype=waveform.dtype, device=waveform.device)

        # 在尾部追加静音
        padded_waveform = torch.cat([waveform, padding], dim=-1)
    else:
        # 已是整数秒（或负数，理论上不会），原样输出
        padded_waveform = waveform

    # 硬指标兜底：确保输出时长严格是整数秒
    # 如果浮点误差导致 padded_waveform.shape[-1] / sr 仍非整数，截断到最近整数秒
    final_samples = padded_waveform.shape[-1]
    final_seconds_exact = final_samples / sample_rate
    final_seconds_int = int(round(final_seconds_exact))

    if abs(final_seconds_exact - final_seconds_int) > 1e-6:
        # 有浮点误差，截断到整数秒对应的采样点数
        truncated_samples = final_seconds_int * sample_rate
        if truncated_samples < final_samples:
            padded_waveform = padded_waveform[:, :, :truncated_samples]

    return padded_waveform


class AudioPad_DJ:
    """
    音频加载 + 补齐整数秒节点。
    传统 ComfyUI 节点框架，自带文件选择/上传/播放条 UI。
    在尾部追加静音，把总时长凑成 ceil(原时长) 整数秒。
    """

    @classmethod
    def INPUT_TYPES(cls):
        # 扫描 input 目录的音频文件（和官方 LoadAudio 一致）
        input_dir = folder_paths.get_input_directory()
        os.makedirs(input_dir, exist_ok=True)
        files = folder_paths.filter_files_content_types(os.listdir(input_dir), ["audio", "video"])

        return {
            "required": {
                # 文件选择下拉（COMBO 类型，前端自动提供上传按钮）
                "audio": (sorted(files), {
                    "audio_upload": True,  # 启用音频上传（和 LoadAudioUI 一致）
                }),
            },
            "optional": {
                # 上游 AUDIO 输入（普适性：任何输出 AUDIO 的节点都可以连进来）
                # 如果连接了上游节点，优先使用上游的音频，忽略文件选择
                "audio_input": ("AUDIO",),
                # 播放条 UI（前端看到 AUDIO_UI 类型会自动渲染播放条）
                "audioUI": ("AUDIO_UI",)
            }
        }

    RETURN_TYPES = ("AUDIO", "FLOAT")
    RETURN_NAMES = ("音频", "时长(秒)")
    FUNCTION = "run"
    CATEGORY = "DJ/AudioPad"

    DESCRIPTION = (
        "音频加载 + 补齐整数秒：在尾部追加静音，把总时长凑成 ceil(原时长) 整数秒。\n"
        "原音频内容不做任何修改，采样率和声道数继承原音频。\n"
        "硬指标：输出时长必须是整数秒。"
    )

    def run(self, audio, audio_input=None, **kwargs):
        """
        执行逻辑：
        1. 如果有上游 AUDIO 输入，优先使用（普适性：任何输出 AUDIO 的节点都可以连进来）
        2. 否则从文件选择下拉加载音频
        3. 补齐整数秒
        4. 返回 AUDIO dict
        播放条由前端 AUDIO_UI 输入自动提供，不需要后端保存临时文件。
        """
        # 优先使用上游 AUDIO 输入（如果连接了）
        if audio_input is not None:
            waveform = audio_input["waveform"]
            sample_rate = audio_input["sample_rate"]
            # 确保有 batch 维：[C, S] → [1, C, S]
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(0)
        else:
            # 从文件选择下拉加载音频
            audio_path = folder_paths.get_annotated_filepath(audio)
            waveform, sample_rate = load_audio(audio_path)
            # 加 batch 维：[C, S] → [1, C, S]
            waveform = waveform.unsqueeze(0)

        # 补齐整数秒
        padded_waveform = pad_to_integer_seconds(waveform, sample_rate)

        # 计算时长（秒，精确到小数点后3位）
        duration = round(padded_waveform.shape[-1] / sample_rate, 3)

        # 返回 ComfyUI AUDIO dict + 时长
        result = {"waveform": padded_waveform, "sample_rate": sample_rate}
        return (result, duration)

    @classmethod
    def IS_CHANGED(cls, audio, audio_input=None, **kwargs):
        """
        缓存指纹：
        - 如果有上游 AUDIO 输入，返回 NaN（强制每次重新执行，因为上游音频可能变了）
        - 否则用文件内容的 SHA256 做指纹（和官方 LoadAudio 一致）
        """
        if audio_input is not None:
            return float('nan')  # 强制重新执行
        
        audio_path = folder_paths.get_annotated_filepath(audio)
        m = hashlib.sha256()
        with open(audio_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, audio, audio_input=None, **kwargs):
        """
        验证输入：
        - 如果有上游 AUDIO 输入，直接通过
        - 否则验证文件是否存在（和官方 LoadAudio 一致）
        """
        if audio_input is not None:
            return True
        
        if not folder_paths.exists_annotated_filepath(audio):
            return "Invalid audio file: {}".format(audio)
        return True
