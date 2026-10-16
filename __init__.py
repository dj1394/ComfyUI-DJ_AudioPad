# -*- coding: utf-8 -*-
"""
ComfyUI-DJ_AudioPad · 音频加载 + 补齐整数秒节点

单节点包：Node ID = 文件夹名，显示名 = 文件夹名 + _中文功能
用传统 ComfyUI 节点框架（NODE_CLASS_MAPPINGS），和 DJ_Batch 一致。
"""

from .audio_pad import AudioPad_DJ

NODE_CLASS_MAPPINGS = {
    "ComfyUI-DJ_AudioPad": AudioPad_DJ,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyUI-DJ_AudioPad": "ComfyUI-DJ_AudioPad_音频补齐整数秒",
}
