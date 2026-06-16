from .ltx_keyframer import KLLTXKeyframer
from .multi_image_loader import MultiImageLoader
from .ltx_sequencer import KLLTXSequencer
from .speech_length_calculator import KLSpeechLengthCalculator
from .load_audio_ui import KLLoadAudioUI
from .load_video_ui import KLLoadVideoUI
from .ltx_director import KLLTXDirector
from .ltx_director_guide import LTXDirectorGuide
from .ltx_director_api import KLLTXDirectorWrapper
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override


class PromptRelay(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            KLLTXDirector,
            LTXDirectorGuide
        ]


async def comfy_entrypoint() -> PromptRelay:
    return PromptRelay()
    
NODE_CLASS_MAPPINGS = {
    "KL LTXKeyframer": KLLTXKeyframer,
    "KL MultiImageLoader": MultiImageLoader,
    "KL LTXSequencer": KLLTXSequencer,
    "KL SpeechLengthCalculator": KLSpeechLengthCalculator,
    "KL LoadAudioUI": KLLoadAudioUI,
    "KL LoadVideoUI": KLLoadVideoUI,
    "KL LTXDirector": KLLTXDirector,
    "KL LTXDirectorGuide": LTXDirectorGuide,
    'KL LTX Director (Auto)': KLLTXDirectorWrapper
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KL LTXKeyframer": "KL LTX Keyframer",
    "KL MultiImageLoader": "KL Multi Image Loader",
    "KL LTXSequencer": "KL LTX Sequencer",
    "KL SpeechLengthCalculator": "KL Speech Length Calculator",
    "KL LoadAudioUI": "KL Load Audio UI",
    "KL LoadVideoUI": "KL Load Video UI",
    "KL LTXDirector": "KL LTX Director",
    "KL LTXDirectorGuide": "KL LTX Director Guide",
    'KL LTX Director (Auto)': 'KL LTX Director (Auto)'
}

WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
