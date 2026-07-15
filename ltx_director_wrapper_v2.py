# -*- coding: utf-8 -*-
"""
LTX Director Wrapper V2 – 支持 IC-LoRA、Retake 模式、视频段等新特性。
用户通过一个 JSON 配置驱动，内部自动构造完整 timeline_data 并调用原始 LTXDirector。
"""

import json
import logging
from .ltx_director import KLLTXDirector, GuideData, MotionGuideData

log = logging.getLogger(__name__)


class KLLTXDirectorWrapperV2:
    """
    简化版 LTX Director V2，支持新特性：
    - 图片/文本/视频段 (images)
    - IC-LoRA 参考段 (motion)
    - 音频段 (audio)
    - Retake 模式 (retake)
    所有全局参数均为节点控件。
    """

    @staticmethod
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

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "user_config": ("STRING", {
                    "multiline": True,
                    "default": '{"images": [], "motion": [], "audio": null, "retake": null}'
                }),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 16.0, "max": 99.0, "step": 1}),
                "width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "global_prompt": ("STRING", {"default": "", "multiline": True}),
                "resize_method": (["maintain aspect ratio", "stretch to fit", "pad", "pad green", "crop"], {"default": "maintain aspect ratio"}),
                "divisible_by": ("INT", {"default": 32, "min": 1, "max": 256, "step": 1}),
                "img_compression": ("INT", {"default": 18, "min": 0, "max": 100, "step": 1}),
                "epsilon": ("FLOAT", {"default": 0.001, "min": 0.0001, "max": 0.99, "step": 0.0001}),
                "use_custom_audio": ("BOOLEAN", {"default": True}),
                "inpaint_audio": ("BOOLEAN", {"default": True}),
                "use_custom_motion": ("BOOLEAN", {"default": True}),
                "override_audio": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "audio_vae": ("VAE",),
                "optional_latent": ("LATENT",),
            }
        }

    CATEGORY = "KL WhatDreamsCost"
    RETURN_TYPES = ("MODEL", "CONDITIONING", "LATENT", "LATENT", "GUIDE_DATA", "MOTION_GUIDE_DATA", "FLOAT", "AUDIO")
    RETURN_NAMES = ("model", "positive", "video_latent", "audio_latent", "guide_data", "motion_guide_data", "frame_rate", "combined_audio")
    FUNCTION = "execute"

    def execute(self, model, clip, user_config,
                frame_rate, width, height,
                global_prompt, resize_method, divisible_by,
                img_compression, epsilon,
                use_custom_audio, inpaint_audio,
                use_custom_motion, override_audio,
                audio_vae=None, optional_latent=None):
        # 1. 解析配置（仅包含动态数据）
        try:
            config = self.parse_user_config(user_config)
        except ValueError as e:
            log.error(f"Failed to parse user_config: {e}")
            raise e

        images = config.get("images", [])
        motion = config.get("motion", [])
        audio = config.get("audio")
        retake = config.get("retake")

        # 2. 构建 segments (主轨道)
        segments = []
        total_frames = 0
        start_sec = 0.0   # 用于自动计算 start 的累计

        for idx, item in enumerate(images):
            seg_type = item.get("type", "image")
            url = item.get("url", "")
            start = item.get("start")
            duration = item.get("duration", 3.0)
            prompt = item.get("prompt", "")
            strength = item.get("strength", 1.0)
            if start is None:
                start = start_sec
            start_frames = int(start * frame_rate)
            length_frames = int(duration * frame_rate)

            seg = {
                "id": f"seg_{idx}_{id(item)}",
                "start": start_frames,
                "length": length_frames,
                "prompt": prompt,
            }

            if seg_type == "text":
                seg["type"] = "text"
            elif seg_type == "image":
                if not url:
                    log.warning(f"Image segment {idx} missing url, skipping.")
                    continue
                seg["type"] = "image"
                seg["imageUrl"] = url
                seg["imageB64"] = url   # 用于通过过滤
                seg["guideStrength"] = strength
            elif seg_type == "video":
                if not url:
                    log.warning(f"Video segment {idx} missing url, skipping.")
                    continue
                seg["type"] = "video"
                seg["imageUrl"] = url
                seg["imageB64"] = url
                seg["guideStrength"] = strength
                seg["trimStart"] = 0
                seg["videoDurationFrames"] = length_frames
            else:
                log.warning(f"Unknown segment type '{seg_type}' at index {idx}, skipping.")
                continue

            segments.append(seg)
            total_frames = max(total_frames, start_frames + length_frames)
            start_sec = start + duration

        if not segments:
            segments.append({
                "id": "placeholder",
                "start": 0,
                "length": max(24, int(1 * frame_rate)),
                "prompt": "empty",
                "type": "text",
            })
            total_frames = max(24, total_frames)

        # 3. 构建 motionSegments (IC-LoRA 轨道)
        motion_segments = []
        for idx, item in enumerate(motion):
            url = item.get("url", "")
            if not url:
                continue
            start = item.get("start", 0.0)
            duration = item.get("duration", 5.0)
            video_strength = item.get("videoStrength", 1.0)
            video_attention_strength = item.get("videoAttentionStrength", 0.65)
            is_image = item.get("type") == "image" or url.lower().endswith(('.jpg','.jpeg','.png','.webp'))
            start_frames = int(start * frame_rate)
            length_frames = int(duration * frame_rate)

            seg = {
                "id": f"motion_{idx}_{id(item)}",
                "type": "motion_video",
                "start": start_frames,
                "length": length_frames,
                "trimStart": 0,
                "videoDurationFrames": length_frames,
                "videoFile": "",
                "videoUrl": url,
                "fileName": url.split("/")[-1],
                "videoStrength": video_strength,
                "videoAttentionStrength": video_attention_strength,
                "isStaticImage": is_image,
                "resampleMode": "nearest",
            }
            if is_image:
                seg["imageUrl"] = url
                seg["imageB64"] = url
            motion_segments.append(seg)
            total_frames = max(total_frames, start_frames + length_frames)

        # 4. 构建 audioSegments (音频轨道)
        audio_segments = []
        if audio:
            audio_start = audio.get("start", 0.0)
            audio_duration = audio.get("duration", 0.0)
            audio_url = audio.get("url", "")
            if audio_url and audio_duration > 0:
                audio_start_frames = int(audio_start * frame_rate)
                audio_length_frames = int(audio_duration * frame_rate)
                seg = {
                    "id": f"audio_{id(audio)}",
                    "type": "audio",
                    "start": audio_start_frames,
                    "length": audio_length_frames,
                    "trimStart": 0,
                    "audioDurationFrames": audio_length_frames,
                    "audioUrl": audio_url,
                    "audioFile": "",
                    "fileName": audio_url.split("/")[-1],
                    "waveformPeaks": [],
                }
                audio_segments.append(seg)
                total_frames = max(total_frames, audio_start_frames + audio_length_frames)

        if total_frames <= 0:
            total_frames = 24

        # 5. Retake 模式
        retake_mode = False
        retake_video = None
        retake_start = 0
        retake_length = total_frames
        retake_prompt = ""
        retake_strength = 1.0
        if retake:
            retake_mode = True
            video_url = retake.get("video_url", "")
            if video_url:
                retake_start = int(retake.get("start", 0.0) * frame_rate)
                retake_length = int(retake.get("length", 5.0) * frame_rate)
                retake_prompt = retake.get("prompt", "")
                retake_strength = retake.get("strength", 1.0)
                retake_video = {
                    "fileName": video_url.split("/")[-1],
                    "imageFile": "",
                    "videoDurationFrames": retake_length,
                    "fileSize": 0,
                }
                log.warning("Retake mode with remote URL may not work unless file is already in input directory.")

        # 6. 构建完整的 timeline_data
        timeline_data = {
            "segments": segments,
            "motionSegments": motion_segments,
            "audioSegments": audio_segments,
            "global_prompt": global_prompt,
            "retake_global_prompt": global_prompt if retake_mode else "",
            "mainTrackEnabled": True,
            "audioTrackEnabled": use_custom_audio,
            "motionTrackEnabled": use_custom_motion,
            "showFilenames": True,
            "overrideAudio": override_audio,
            "inpaint_audio": inpaint_audio,
            "retakeMode": retake_mode,
            "retakeStart": retake_start,
            "retakeLength": retake_length,
            "retakePrompt": retake_prompt,
            "retakeStrength": retake_strength,
            "retakeVideo": retake_video,
            "normalStartFrame": 0,
            "normalDurationFrames": total_frames,
        }
        timeline_json = json.dumps(timeline_data)

        # 7. 辅助字段
        sorted_segments = sorted(segments, key=lambda s: s["start"])
        local_prompts = " | ".join([s.get("prompt", "") for s in sorted_segments])
        segment_lengths = ",".join([str(s.get("length", 24)) for s in sorted_segments])
        sorted_image_segments = [s for s in sorted_segments if s.get("type") in ("image", "video")]
        guide_strength = ",".join([str(s.get("guideStrength", 1.0)) for s in sorted_image_segments])

        start_frame = 0
        end_frame = total_frames
        duration_frames = total_frames
        start_second = 0.0
        end_second = duration_frames / float(frame_rate)
        duration_seconds = duration_frames / float(frame_rate)

        # 8. 调用原始 LTXDirector
        result = KLLTXDirector.execute(
            model=model,
            clip=clip,
            start_second=start_second,
            end_second=end_second,
            duration_seconds=duration_seconds,
            start_frame=start_frame,
            end_frame=end_frame,
            duration_frames=duration_frames,
            timeline_data=timeline_json,
            local_prompts=local_prompts,
            segment_lengths=segment_lengths,
            global_prompt=global_prompt,
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
            use_custom_audio=use_custom_audio,
            inpaint_audio=inpaint_audio,
            use_custom_motion=use_custom_motion,
            override_audio=override_audio,
        )

        return result
