# -*- coding: utf-8 -*-
"""
LTX Director，为服务端做的节点
"""
import json
import logging
from .ltx_director import KLLTXDirector, GuideData

log = logging.getLogger(__name__)


def parse_user_config(raw_text: str) -> dict:
    """
    宽容解析 JSON：去除 BOM，提取第一个完整 JSON 对象，忽略尾部多余内容。
    """
    text = raw_text.lstrip('\ufeff').strip()
    if not text:
        return {}
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(text)
        return obj
    except json.JSONDecodeError:
        start = text.find('{')
        if start == -1:
            raise ValueError("No JSON object found in input.")
        try:
            obj, _ = decoder.raw_decode(text[start:])
            return obj
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse JSON. Received (first 200 chars):\n{text[start:start+200]}") from e


class KLLTXDirectorWrapper:
    """
    Simplified LTX Director.
    Supports "type": "image" (requires url) or "type": "text" (prompt only).
    支持的输入参数格式：
    {
  "images": [
    {
      "url": "https://example.com/cat.jpg",
      "start": 0.0,
      "duration": 3.0,
      "prompt": "a cute cat sitting on a sofa",
      "type": "image",
      "strength": 1.0
    },
    {
      "start": 3.0,
      "duration": 2.0,
      "prompt": "camera slowly zooms out, cinematic transition",
      "type": "text"
    },
    {
      "url": "https://example.com/dog.jpg",
      "start": 5.0,
      "duration": 4.0,
      "prompt": "a dog running in the park",
      "type": "image",
      "strength": 0.8
    }
  ],
  "audio": {
    "url": "https://example.com/background_music.mp3",
    "start": 0.0,
    "duration": 9.0
  },
  "global_prompt": "4k, highly detailed, cinematic lighting, masterpiece",
  "frame_rate": 24,
  "width": 768,
  "height": 512,
  "resize_method": "maintain aspect ratio",
  "divisible_by": 32,
  "img_compression": 18,
  "epsilon": 0.001
}
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "user_config": ("STRING", {
                    "multiline": True,
                    "default": '{"images": [], "audio": null, "global_prompt": "", "frame_rate": 24, "width": 768, "height": 512}'
                }),
            },
            "optional": {
                "audio_vae": ("VAE",),
                "optional_latent": ("LATENT",),
            }
        }

    CATEGORY = "KL WhatDreamsCost"
    RETURN_TYPES = ("MODEL", "CONDITIONING", "LATENT", "LATENT", "GUIDE_DATA", "FLOAT", "AUDIO")
    RETURN_NAMES = ("model", "positive", "video_latent", "audio_latent", "guide_data", "frame_rate", "combined_audio")
    FUNCTION = "execute"

    def execute(self, model, clip, user_config, audio_vae=None, optional_latent=None):
        # 1. 解析用户配置（宽容模式）
        try:
            config = parse_user_config(user_config)
        except ValueError as e:
            log.error(f"Failed to parse user_config: {e}")
            raise e

        images = config.get("images", [])
        audio = config.get("audio")
        global_prompt = config.get("global_prompt", "")
        frame_rate = config.get("frame_rate", 24)
        width = config.get("width", 0)
        height = config.get("height", 0)
        resize_method = config.get("resize_method", "maintain aspect ratio")
        divisible_by = config.get("divisible_by", 32)
        img_compression = config.get("img_compression", 18)
        epsilon = config.get("epsilon", 0.001)

        # 2. 构建原始段列表（未排序）
        raw_segments = []
        audio_segments = []
        total_frames = 0
        image_count = 0

        for idx, item in enumerate(images):
            start_sec = item.get("start")
            duration_sec = item.get("duration", 3.0)
            length_frames = int(duration_sec * frame_rate)

            if start_sec is None:
                start_frames = total_frames
            else:
                start_frames = int(start_sec * frame_rate)

            prompt = item.get("prompt", "")
            seg_type = item.get("type", "image")

            if seg_type == "text":
                seg = {
                    "id": f"seg_{idx}_{id(item)}",
                    "start": start_frames,
                    "length": length_frames,
                    "prompt": prompt,
                    "type": "text",
                }
                raw_segments.append(seg)
                total_frames = max(total_frames, start_frames + length_frames)
                continue

            # 图片段：必须有 url
            url = item.get("url", "")
            if not url:
                log.warning(f"Image segment {idx} has no URL, skipping.")
                continue

            strength = item.get("strength", 1.0)
            strength = max(0.0, min(1.0, strength))

            # ⭐ 关键修复：同时提供 imageB64（值为 URL）以通过原始节点的过滤条件
            # 原始节点过滤时检查 (imageFile or imageB64)，但 _load_image_tensor 优先使用 imageUrl
            seg = {
                "id": f"seg_{idx}_{id(item)}",
                "start": start_frames,
                "length": length_frames,
                "prompt": prompt,
                "type": "image",
                "imageUrl": url,
                "imageB64": url,           # ⭐ 关键：用于通过过滤条件
                "guideStrength": strength,
            }
            raw_segments.append(seg)
            image_count += 1
            total_frames = max(total_frames, start_frames + length_frames)

        # 检查是否有任何图片段
        if image_count == 0:
            raise ValueError(
                "No valid image segments found in user_config. "
                "At least one image with a valid 'url' is required for quality generation."
            )

        # 3. 按 start 排序所有段（确保与原始节点内部排序一致）
        sorted_segments = sorted(raw_segments, key=lambda s: s["start"])
        total_frames = 0
        for seg in sorted_segments:
            total_frames = max(total_frames, seg["start"] + seg["length"])

        # 4. 处理音频
        if audio:
            audio_start_sec = audio.get("start", 0)
            audio_duration_sec = audio.get("duration", 0)
            audio_url = audio.get("url", "")
            if audio_url and audio_duration_sec > 0:
                audio_start_frames = int(audio_start_sec * frame_rate)
                audio_length_frames = int(audio_duration_sec * frame_rate)
                audio_seg = {
                    "id": f"audio_{id(audio)}",
                    "start": audio_start_frames,
                    "length": audio_length_frames,
                    "trimStart": 0,
                    "audioDurationFrames": audio_length_frames,
                    "audioUrl": audio_url,
                    "fileName": audio_url.split("/")[-1],
                    "waveformPeaks": [],
                }
                audio_segments.append(audio_seg)
                total_frames = max(total_frames, audio_start_frames + audio_length_frames)

        if total_frames <= 0:
            total_frames = 24

        # 5. 构建 timeline_data（使用排序后的段）
        timeline_data = {
            "segments": sorted_segments,
            "audioSegments": audio_segments,
        }
        timeline_json = json.dumps(timeline_data)

        # 6. 构建辅助字符串（均基于排序后的段）
        local_prompts = " | ".join([seg.get("prompt", "") for seg in sorted_segments])
        segment_lengths = ",".join([str(seg.get("length", 24)) for seg in sorted_segments])

        # guide_strength 只包含图片段，按排序后的顺序
        sorted_image_segments = [s for s in sorted_segments if s.get("type") == "image"]
        guide_strength = ",".join([str(s.get("guideStrength", 1.0)) for s in sorted_image_segments])

        log.info(f"[LTX Director Wrapper] Total frames: {total_frames}, frame_rate: {frame_rate}")
        log.info(f"[LTX Director Wrapper] Image segments: {len(sorted_image_segments)}, guide_strength: {guide_strength}")

        # 7. 委托给原始 KLLTXDirector
        result = KLLTXDirector.execute(
            model=model,
            clip=clip,
            global_prompt=global_prompt,
            duration_frames=total_frames,
            duration_seconds=total_frames / float(frame_rate),
            timeline_data=timeline_json,
            local_prompts=local_prompts,
            segment_lengths=segment_lengths,
            guide_strength=guide_strength,
            epsilon=epsilon,
            frame_rate=frame_rate,
            display_mode="frames",
            custom_width=width,
            custom_height=height,
            resize_method=resize_method,
            divisible_by=divisible_by,
            img_compression=img_compression,
            audio_vae=audio_vae,
            optional_latent=optional_latent,
            use_custom_audio=bool(audio)
        )

        return result
