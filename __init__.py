"""
ComfyUI-OpenFlam Custom Node
----------------------------------------------------
OpenFLAM integration for ComfyUI enabling audio-text similarity analysis
and sound event detection.

Based on: https://arxiv.org/abs/2505.05335
Original Repository: https://github.com/adobe-research/openflam
"""

from .nodes.openflam_nodes import (
    OpenFlamLoadModel,
    OpenFlamGlobalSimilarity,
    OpenFlamLocalSimilarity,
    OpenFlamLoadAudio,
    OpenFlamVisualizeSpectrogram
)

NODE_CLASS_MAPPINGS = {
    "OpenFlamLoadModel": OpenFlamLoadModel,
    "OpenFlamGlobalSimilarity": OpenFlamGlobalSimilarity,
    "OpenFlamLocalSimilarity": OpenFlamLocalSimilarity,
    "OpenFlamLoadAudio": OpenFlamLoadAudio,
    "OpenFlamVisualizeSpectrogram": OpenFlamVisualizeSpectrogram,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenFlamLoadModel": "OpenFLAM Load Model",
    "OpenFlamGlobalSimilarity": "OpenFLAM Global Similarity",
    "OpenFlamLocalSimilarity": "OpenFLAM Local Similarity",
    "OpenFlamLoadAudio": "OpenFLAM Load Audio",
    "OpenFlamVisualizeSpectrogram": "OpenFLAM Visualize Spectrogram",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
