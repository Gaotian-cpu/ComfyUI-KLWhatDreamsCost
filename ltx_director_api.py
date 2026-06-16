# -*- coding: utf-8 -*-
"""
LTX Director，为服务端做的节点
"""
import json
import logging
from .ltx_director import KLLTXDirector, GuideData

log = logging.getLogger(__name__)


class KLLTXDirectorWrapper:
    """
    Simplified LTX Director (Legacy API compatible).
    Accepts a JSON config and delegates to the original Director.
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
        # 1. Parse user JSON
        try:
            config = json.loads(user_config)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid user_config JSON: {e}")

        images = config.get("images", [])
        audio = config.get("audio")
        global_prompt = config.get("global_prompt", "")
        frame_rate = config.get("frame_rate", 24)
        width = config.get("width", 768)
        height = config.get("height", 512)
        resize_method = config.get("resize_method", "maintain aspect ratio")
        divisible_by = config.get("divisible_by", 32)
        img_compression = config.get("img_compression", 18)
        epsilon = config.get("epsilon", 0.001)

        segments = []
        audio_segments = []
        total_frames = 0

        # 2. Build image segments
        for idx, img in enumerate(images):
            start = img.get("start", total_frames)
            duration_sec = img.get("duration", 3.0)
            length_frames = int(duration_sec * frame_rate)
            prompt = img.get("prompt", "")
            strength = img.get("strength", 1.0)
            url = img.get("url", "")
            if not url:
                log.warning(f"Image {idx} has no URL, skipping.")
                continue
            seg = {
                "id": f"seg_{idx}_{id(img)}",
                "start": start,
                "length": length_frames,
                "prompt": prompt,
                "type": "image",
                "imageUrl": url,
                "guideStrength": strength,
            }
            segments.append(seg)
            total_frames = max(total_frames, start + length_frames)

        # Fallback if no images
        if not segments:
            segments.append({
                "id": "placeholder",
                "start": 0,
                "length": max(24, int(1 * frame_rate)),
                "prompt": "empty",
                "type": "text",
            })
            total_frames = max(24, total_frames)

        # 3. Build audio segment if provided
        if audio:
            audio_start = audio.get("start", 0)
            audio_duration_sec = audio.get("duration", 0)
            audio_url = audio.get("url", "")
            if audio_url and audio_duration_sec > 0:
                audio_length_frames = int(audio_duration_sec * frame_rate)
                audio_seg = {
                    "id": f"audio_{id(audio)}",
                    "start": audio_start,
                    "length": audio_length_frames,
                    "trimStart": 0,
                    "audioDurationFrames": audio_length_frames,
                    "audioUrl": audio_url,
                    "fileName": audio_url.split("/")[-1],
                    "waveformPeaks": [],
                }
                audio_segments.append(audio_seg)
                total_frames = max(total_frames, audio_start + audio_length_frames)

        if total_frames <= 0:
            total_frames = 24

        # 4. Build timeline_data and auxiliary strings
        timeline_data = {
            "segments": segments,
            "audioSegments": audio_segments,
        }
        timeline_json = json.dumps(timeline_data)

        local_prompts = " | ".join([seg.get("prompt", "") for seg in segments])
        segment_lengths = ",".join([str(seg.get("length", 24)) for seg in segments])
        guide_strength = ",".join([
            str(seg.get("guideStrength", 1.0)) if seg.get("type") == "image" else "0.0"
            for seg in segments
        ])

        # 5. Delegate to the original KLLTXDirector (Class Method)
        # Note: KLLTXDirector.execute is a @classmethod, so we call it directly.
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

        # result is a tuple in the order defined by KLLTXDirector's outputs
        return result
