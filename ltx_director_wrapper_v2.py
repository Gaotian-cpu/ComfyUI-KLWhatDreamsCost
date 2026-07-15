# -*- coding: utf-8 -*-
"""
LTX Director Wrapper V2 – 支持 IC-LoRA、Retake 模式、视频段等新特性。
用户通过一个 JSON 配置驱动，内部自动构造完整 timeline_data 并调用原始 LTXDirector。
"""

import json
import logging
from .ltx_director import KLLTXDirector, GuideData, MotionGuideData

log = logging.getLogger(__name__)


def parse_user_config(raw_text: str) -> dict:
    """宽容解析 JSON：去除 BOM，提取第一个完整 JSON 对象，忽略尾部多余内容。"""
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


class KLLTXDirectorWrapperV2:
    """
    简化版 LTX Director V2，支持新特性：
    - 图片/文本/视频段 (images)
    - IC-LoRA 参考段 (motion)
    - 音频段 (audio)
    - Retake 模式 (retake)
    - 全局参数
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "user_config": ("STRING", {
                    "multiline": True,
                    "default": '{"images": [], "motion": [], "audio": null, "retake": null, "global_prompt": "", '
                               '"frame_rate": 24, "width": 768, "height": 512} '
                }),
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

    def execute(self, model, clip, user_config, audio_vae=None, optional_latent=None):
        # 1. 解析配置
        try:
            config = parse_user_config(user_config)
        except ValueError as e:
            log.error(f"Failed to parse user_config: {e}")
            raise e

        images = config.get("images", [])
        motion = config.get("motion", [])
        audio = config.get("audio")
        retake = config.get("retake")
        global_prompt = config.get("global_prompt", "")
        frame_rate = config.get("frame_rate", 24)
        width = config.get("width", 0)          # 0 表示自适应
        height = config.get("height", 0)
        resize_method = config.get("resize_method", "maintain aspect ratio")
        divisible_by = config.get("divisible_by", 32)
        img_compression = config.get("img_compression", 18)
        epsilon = config.get("epsilon", 0.001)
        use_custom_audio = config.get("use_custom_audio", True)
        inpaint_audio = config.get("inpaint_audio", True)
        use_custom_motion = config.get("use_custom_motion", True)
        override_audio = config.get("override_audio", False)

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
            # 处理 start 未指定情况：自动接续
            if start is None:
                start = start_sec
            start_frames = int(start * frame_rate)
            length_frames = int(duration * frame_rate)

            # 构建段对象
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
                seg["videoDurationFrames"] = length_frames   # 暂定，后续可从实际视频获取
            else:
                log.warning(f"Unknown segment type '{seg_type}' at index {idx}, skipping.")
                continue

            segments.append(seg)
            total_frames = max(total_frames, start_frames + length_frames)
            start_sec = start + duration   # 更新累计起始时间

        # 如果没有有效段，创建占位文本段
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
            # 判断是否为静态图片（通过扩展名或type）
            is_image = item.get("type") == "image" or url.lower().endswith(('.jpg','.jpeg','.png','.webp'))
            seg_type = "motion_video" if not is_image else "motion_video"  # 统一为motion_video，isStaticImage标记
            start_frames = int(start * frame_rate)
            length_frames = int(duration * frame_rate)

            seg = {
                "id": f"motion_{idx}_{id(item)}",
                "type": seg_type,
                "start": start_frames,
                "length": length_frames,
                "trimStart": 0,
                "videoDurationFrames": length_frames,
                "videoFile": "",   # 上传后填充
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
                    "audioFile": "",   # 上传后填充
                    "fileName": audio_url.split("/")[-1],
                    "waveformPeaks": [],
                }
                audio_segments.append(seg)
                total_frames = max(total_frames, audio_start_frames + audio_length_frames)

        if total_frames <= 0:
            total_frames = 24

        # 5. 处理 Retake 模式
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
                    "imageFile": "",   # 上传后填充
                    "videoDurationFrames": retake_length,
                    "fileSize": 0,
                }
                # 注意：retake 视频的 URL 需要能通过 _load_video_tensor 加载，
                # 但该函数依赖本地文件，因此这里仅作占位，实际需要上传或提供本地路径。
                # 为简化，我们暂时将 URL 存入 imageUrl 字段供前端预览，但后端需处理。
                # 更好的做法是要求用户上传到 ComfyUI input 目录。
                # 由于 wrapper 是服务端调用，我们可以接受 URL 并在 execute 中临时下载？
                # 这里先保留，后续可扩展。
                log.warning("Retake mode with remote URL may not work unless file is already in input directory.")
                # 为了支持远程 URL，我们可以在 timeline_data 中放入 videoUrl 字段，但原始节点只认 imageFile。
                # 因此建议用户将视频放入 input 目录。
                # 此处我们仅构造基本结构，供用户后续手动处理。

        # 6. 构建完整的 timeline_data（包含所有轨道和设置）
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

        # 7. 生成辅助字段 (local_prompts, segment_lengths, guide_strength)
        # 排序 segments 按 start
        sorted_segments = sorted(segments, key=lambda s: s["start"])
        local_prompts = " | ".join([s.get("prompt", "") for s in sorted_segments])
        segment_lengths = ",".join([str(s.get("length", 24)) for s in sorted_segments])
        # guide_strength 只针对 type image/video
        sorted_image_segments = [s for s in sorted_segments if s.get("type") in ("image", "video")]
        guide_strength = ",".join([str(s.get("guideStrength", 1.0)) for s in sorted_image_segments])

        # 8. 计算 start_frame, end_frame, duration_frames
        # 在 V2 中，start_frame 和 end_frame 用于定义生成范围，我们设置为 0 和 total_frames
        start_frame = 0
        end_frame = total_frames
        duration_frames = total_frames
        start_second = 0.0
        end_second = duration_frames / float(frame_rate)
        duration_seconds = duration_frames / float(frame_rate)

        # 9. 调用原始 LTXDirector
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
            display_mode="frames",   # 内部使用帧
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
