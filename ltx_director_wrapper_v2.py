# -*- coding: utf-8 -*-
"""
LTX Director Wrapper V2 – 支持 IC-LoRA、Retake 模式、视频段等新特性。
用户通过一个 JSON 配置驱动，内部自动构造完整 timeline_data 并调用原始 LTXDirector。
"""

import json
import logging
import os
import time
import urllib.request
from urllib.parse import urlparse
import folder_paths

from .ltx_director import KLLTXDirector, GuideData, MotionGuideData

log = logging.getLogger(__name__)


class KLLTXDirectorWrapperV2:
    # 类级别的缓存，URL -> 本地相对路径
    _media_cache = {}

    @staticmethod
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

    @staticmethod
    def _download_media(url: str, subfolder: str = "whatdreamscost") -> str:
        """
        下载远程媒体文件到 ComfyUI 的 input 目录，返回相对于 input 目录的路径。
        如果文件已存在或 URL 为本地路径，则直接返回。
        """
        if not url:
            return ""

        # 检查缓存
        if url in KLLTXDirectorWrapperV2._media_cache:
            return KLLTXDirectorWrapperV2._media_cache[url]

        input_dir = folder_paths.get_input_directory()
        target_dir = os.path.join(input_dir, subfolder)
        os.makedirs(target_dir, exist_ok=True)

        # 如果已经是本地绝对路径，直接返回相对路径
        if os.path.exists(url):
            rel_path = os.path.relpath(url, input_dir)
            KLLTXDirectorWrapperV2._media_cache[url] = rel_path
            return rel_path

        # 尝试解析为相对路径（可能已经在 input 下）
        possible_path = os.path.join(input_dir, url)
        if os.path.exists(possible_path):
            rel_path = os.path.relpath(possible_path, input_dir)
            KLLTXDirectorWrapperV2._media_cache[url] = rel_path
            return rel_path

        # 远程 URL，进行下载
        try:
            parsed = urlparse(url)
            # 提取文件名，若没有则用时间戳
            filename = os.path.basename(parsed.path)
            if not filename:
                filename = f"downloaded_{int(time.time())}.tmp"
            # 添加时间戳防止重复冲突
            name, ext = os.path.splitext(filename)
            safe_filename = f"{name}_{int(time.time())}{ext}"
            local_path = os.path.join(target_dir, safe_filename)

            log.info(f"[LTXDirectorWrapperV2] Downloading {url} -> {local_path}")
            urllib.request.urlretrieve(url, local_path)

            rel_path = os.path.relpath(local_path, input_dir)
            KLLTXDirectorWrapperV2._media_cache[url] = rel_path
            return rel_path

        except Exception as e:
            log.error(f"[LTXDirectorWrapperV2] Failed to download {url}: {e}")
            raise RuntimeError(f"Failed to download media from {url}: {e}")

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
                "frame_rate": ("INT", {"default": 24, "min": 1, "max": 240, "step": 1}),
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
        # 1. 解析配置
        try:
            config = self.parse_user_config(user_config)
        except ValueError as e:
            log.error(f"Failed to parse user_config: {e}")
            raise e

        images = config.get("images", [])
        motion = config.get("motion", [])
        audio = config.get("audio")
        retake = config.get("retake")

        # 2. 构建 segments (主轨道) —— 下载图片/视频
        segments = []
        total_frames = 0
        start_sec = 0.0

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

            # 下载媒体文件
            local_path = ""
            if url:
                try:
                    local_path = self._download_media(url)
                except Exception as e:
                    log.error(f"Segment {idx} download failed: {e}")
                    continue

            seg = {
                "id": f"seg_{idx}_{id(item)}",
                "start": start_frames,
                "length": length_frames,
                "prompt": prompt,
            }

            if seg_type == "text":
                seg["type"] = "text"
            elif seg_type in ("image", "video"):
                if not local_path:
                    log.warning(f"Segment {idx} has no valid local file, skipping.")
                    continue
                seg["type"] = seg_type
                seg["imageFile"] = local_path   # 关键：使用本地文件路径
                seg["guideStrength"] = strength
                if seg_type == "video":
                    seg["trimStart"] = 0
                    seg["videoDurationFrames"] = length_frames
                # 保留 imageUrl 作为参考（但不会被加载）
                seg["imageUrl"] = url
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

        # 3. 构建 motionSegments (IC-LoRA 轨道) —— 下载视频/图片
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

            # 下载媒体
            try:
                local_path = self._download_media(url)
            except Exception as e:
                log.error(f"Motion segment {idx} download failed: {e}")
                continue

            start_frames = int(start * frame_rate)
            length_frames = int(duration * frame_rate)

            seg = {
                "id": f"motion_{idx}_{id(item)}",
                "type": "motion_video",
                "start": start_frames,
                "length": length_frames,
                "trimStart": 0,
                "videoDurationFrames": length_frames,
                "videoFile": local_path,   # 关键：本地文件路径
                "fileName": os.path.basename(local_path),
                "videoStrength": video_strength,
                "videoAttentionStrength": video_attention_strength,
                "isStaticImage": is_image,
                "resampleMode": "nearest",
                "videoUrl": url,   # 保留参考
            }
            if is_image:
                seg["imageFile"] = local_path
            motion_segments.append(seg)
            total_frames = max(total_frames, start_frames + length_frames)

        # 4. 构建 audioSegments (音频轨道) —— 下载音频
        audio_segments = []
        if audio:
            audio_start = audio.get("start", 0.0)
            audio_duration = audio.get("duration", 0.0)
            audio_url = audio.get("url", "")
            if audio_url and audio_duration > 0:
                try:
                    local_audio_path = self._download_media(audio_url)
                except Exception as e:
                    log.error(f"Audio download failed: {e}")
                    # 不阻塞，继续
                    local_audio_path = ""

                if local_audio_path:
                    audio_start_frames = int(audio_start * frame_rate)
                    audio_length_frames = int(audio_duration * frame_rate)
                    seg = {
                        "id": f"audio_{id(audio)}",
                        "type": "audio",
                        "start": audio_start_frames,
                        "length": audio_length_frames,
                        "trimStart": 0,
                        "audioDurationFrames": audio_length_frames,
                        "audioFile": local_audio_path,
                        "fileName": os.path.basename(local_audio_path),
                        "waveformPeaks": [],
                        "audioUrl": audio_url,   # 保留参考
                    }
                    audio_segments.append(seg)
                    total_frames = max(total_frames, audio_start_frames + audio_length_frames)

        if total_frames <= 0:
            total_frames = 24

        # 5. Retake 模式 —— 下载 base video
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
                try:
                    local_video_path = self._download_media(video_url)
                except Exception as e:
                    log.error(f"Retake video download failed: {e}")
                    local_video_path = ""

                if local_video_path:
                    retake_start = int(retake.get("start", 0.0) * frame_rate)
                    retake_length = int(retake.get("length", 5.0) * frame_rate)
                    retake_prompt = retake.get("prompt", "")
                    retake_strength = retake.get("strength", 1.0)
                    retake_video = {
                        "fileName": os.path.basename(local_video_path),
                        "imageFile": local_video_path,   # 关键：本地路径
                        "videoDurationFrames": retake_length,
                        "fileSize": 0,
                    }

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
