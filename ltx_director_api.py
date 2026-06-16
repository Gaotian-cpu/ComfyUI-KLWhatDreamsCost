# -*- coding: utf-8 -*-
"""
LTX Director，为服务端做的节点
"""
import json
import logging
from comfy_api.latest import io
from .ltx_director import KLLTXDirector, GuideData

log = logging.getLogger(__name__)


class KLLTXDirectorWrapper(io.ComfyNode):
    """
    Automated LTX Director – accepts a simple JSON config,
    builds the timeline_data internally, and delegates to the original Director.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KL LTXDirectorWrapper",
            display_name="KL LTX Director (Auto)",
            category="KL WhatDreamsCost",
            description=(
                "Simplified LTX Director. Provide a JSON with image/audio configs, "
                "and this node will automatically construct the timeline_data and execute."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Vae.Input("audio_vae", optional=True),
                io.Latent.Input("optional_latent", optional=True),
                io.String.Input(
                    "user_config",
                    multiline=True,
                    default='{"images": [], "audio": null, "global_prompt": "", "frame_rate": 24, "width": 768, "height": 512}',
                    tooltip="JSON config with fields: images (list of {url, start, duration, prompt, strength}), audio (optional {url, start, duration}), global_prompt, frame_rate, width, height."
                ),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(display_name="video_latent"),
                io.Latent.Output(display_name="audio_latent"),
                GuideData.Output(display_name="guide_data"),
                io.Float.Output(display_name="frame_rate"),
                io.Audio.Output(display_name="combined_audio"),
            ],
        )

    @classmethod
    def execute(cls, model, clip, user_config, audio_vae=None, optional_latent=None):
        # 1. Parse user JSON
        try:
            config = json.loads(user_config)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid user_config JSON: {e}")

        images = config.get("images", [])
        audio = config.get("audio")  # dict or None
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

        # 5. Delegate to the original KLLTXDirector
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
            use_custom_audio=bool(audio)  # enable custom audio if audio config is present
        )

        return result
