from .ltx_keyframer import KLLTXKeyframer
from .multi_image_loader import KLMultiImageLoader
from .ltx_sequencer import KLLTXSequencer
from .speech_length_calculator import KLSpeechLengthCalculator
from .load_audio_ui import KLLoadAudioUI
from .load_video_ui import KLLoadVideoUI
from .ltx_director import KLLTXDirector
from .ltx_director_guide import KLLTXDirectorGuide, KLLTXDirectorCropGuides
from .ltx_director_wrapper_v2 import KLLTXDirectorWrapperV2
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override


class PromptRelay(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            KLLTXDirector,
            KLLTXDirectorGuide
        ]


async def comfy_entrypoint() -> PromptRelay:
    return PromptRelay()
    
NODE_CLASS_MAPPINGS = {
    "KLLTXKeyframer": KLLTXKeyframer,
    "KLMultiImageLoader": KLMultiImageLoader,
    "KLLTXSequencer": KLLTXSequencer,
    "KLSpeechLengthCalculator": KLSpeechLengthCalculator,
    "KLLoadAudioUI": KLLoadAudioUI,
    "KLLoadVideoUI": KLLoadVideoUI,
    "KLLTXDirector": KLLTXDirector,
    "KLLTXDirectorGuide": KLLTXDirectorGuide,
    "KLLTXDirectorCropGuides": KLLTXDirectorCropGuides,
    'KLLTXDirectorWrapperV2': KLLTXDirectorWrapperV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KLLTXKeyframer": "KL LTX Keyframer",
    "KLMultiImageLoader": "KL Multi Image Loader",
    "KLLTXSequencer": "KL LTX Sequencer",
    "KLSpeechLengthCalculator": "KL Speech Length Calculator",
    "KLLoadAudioUI": "KL Load Audio UI",
    "KLLoadVideoUI": "KL Load Video UI",
    "KLLTXDirector": "KL LTX Director",
    "KLLTXDirectorGuide": "KL LTX Director Guide",
    "KLLTXDirectorCropGuides": "KL LTX Director Crop Guides",
    'KLLTXDirectorWrapperV2': 'KL LTXDirector Wrapper V2',
}

WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
