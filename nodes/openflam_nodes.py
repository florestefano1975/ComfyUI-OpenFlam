"""
OpenFLAM Nodes for ComfyUI
----------------------------------------------------
Custom nodes implementing OpenFLAM functionality for audio-text analysis
"""

import os
import sys
import torch
import numpy as np
import librosa
import folder_paths
from pathlib import Path

# Add openflam to path
dir_path = os.path.dirname(os.path.abspath(__file__))
openflam_path = os.path.join(dir_path, "..", "openflam", "src")
if openflam_path not in sys.path:
    sys.path.insert(0, openflam_path)

import openflam


class OpenFlamLoadModel:
    """
    Load OpenFLAM model for audio-text similarity analysis
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (["v1-base"],),
                "cache_dir": ("STRING", {
                    "default": "openflam",
                    "multiline": False
                }),
            },
        }
    
    RETURN_TYPES = ("OPENFLAM_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "audio/openflam"
    
    def load_model(self, model_name, cache_dir):
        """Load OpenFLAM model"""
        # Create cache directory if not exists
        cache_path = os.path.join(folder_paths.models_dir, cache_dir)
        os.makedirs(cache_path, exist_ok=True)
        
        # Detect device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model
        model = openflam.OpenFLAM(
            model_name=model_name,
            default_ckpt_path=cache_path
        ).to(device)
        
        return (model,)


class OpenFlamLoadAudio:
    """
    Load and preprocess audio file for OpenFLAM (requires 48kHz, 10 seconds)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get audio directory from ComfyUI
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) 
                if os.path.isfile(os.path.join(input_dir, f)) and 
                f.split('.')[-1].lower() in ['wav', 'mp3', 'ogg', 'flac', 'm4a', 'aac']]
        
        return {
            "required": {
                "audio": (sorted(files),),
            },
            "optional": {
                "duration": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.1,
                    "max": 60.0,
                    "step": 0.1
                }),
            }
        }
    
    RETURN_TYPES = ("AUDIO_TENSOR", "AUDIO_DATA")
    RETURN_NAMES = ("audio", "audio_data")
    FUNCTION = "load_audio"
    CATEGORY = "audio/openflam"
    
    def load_audio(self, audio, duration=10.0):
        """Load and preprocess audio file"""
        # Get audio file path
        audio_path = folder_paths.get_annotated_filepath(audio)
        
        # OpenFLAM requires 48kHz sample rate
        SR = 48000
        
        # Load audio
        audio_data, sr = librosa.load(audio_path, sr=SR)
        
        # Trim or pad to specified duration
        target_length = int(duration * SR)
        if len(audio_data) > target_length:
            audio_data = audio_data[:target_length]
        else:
            audio_data = np.pad(
                audio_data, 
                (0, target_length - len(audio_data)), 
                mode='constant'
            )
        
        # Convert to tensor
        audio_tensor = torch.tensor(audio_data).unsqueeze(0)  # [1, samples]
        
        # Store audio data info for visualization
        audio_info = {
            "data": audio_data,
            "sample_rate": SR,
            "duration": duration
        }
        
        return (audio_tensor, audio_info)


class OpenFlamGlobalSimilarity:
    """
    Compute global (clip-wise) similarity between audio and text using OpenFLAM
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("OPENFLAM_MODEL",),
                "audio": ("AUDIO_TENSOR",),
                "text_prompts": ("STRING", {
                    "default": "breaking bones, metallic creak, tennis ball",
                    "multiline": True
                }),
            },
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("results", "best_match")
    FUNCTION = "compute_similarity"
    CATEGORY = "audio/openflam"
    OUTPUT_NODE = True
    
    def compute_similarity(self, model, audio, text_prompts):
        """Compute global similarity between audio and text"""
        # Parse text prompts (comma separated or newline separated)
        text_samples = [t.strip() for t in text_prompts.replace('\n', ',').split(',') if t.strip()]
        
        if not text_samples:
            return ("No text prompts provided", "")
        
        # Move audio to device
        device = next(model.parameters()).device
        audio = audio.to(device)
        
        # Get features
        with torch.no_grad():
            audio_global_feature = model.get_global_audio_features(audio)
            text_feature = model.get_text_features(text_samples)
            
            # Calculate similarity (dot product)
            global_similarities = (text_feature @ audio_global_feature.T).squeeze(1)
        
        # Format results
        results = ["Global Audio-Text Similarities:\n" + "="*40]
        scores = []
        
        for text, score in zip(text_samples, global_similarities):
            score_val = score.item()
            scores.append((text, score_val))
            results.append(f"{text}: {score_val:.4f}")
        
        # Find best match
        best_text, best_score = max(scores, key=lambda x: x[1])
        best_match = f"Best Match: {best_text} ({best_score:.4f})"
        
        results.append("="*40)
        results.append(best_match)
        
        result_str = "\n".join(results)
        
        return (result_str, best_match)


class OpenFlamLocalSimilarity:
    """
    Compute local (frame-wise) similarity for sound event detection using OpenFLAM
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("OPENFLAM_MODEL",),
                "audio": ("AUDIO_TENSOR",),
                "text_prompts": ("STRING", {
                    "default": "breaking bones, metallic creak, tennis ball",
                    "multiline": True
                }),
                "method": (["unbiased", "approximate"],),
                "cross_product": ("BOOLEAN", {"default": True}),
                "median_filter_size": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 11,
                    "step": 2
                }),
            },
        }
    
    RETURN_TYPES = ("STRING", "DICT")
    RETURN_NAMES = ("summary", "similarity_maps")
    FUNCTION = "compute_local_similarity"
    CATEGORY = "audio/openflam"
    OUTPUT_NODE = True
    
    def compute_local_similarity(self, model, audio, text_prompts, method, 
                                 cross_product, median_filter_size):
        """Compute local similarity for sound event detection"""
        import scipy.ndimage
        
        # Parse text prompts
        text_samples = [t.strip() for t in text_prompts.replace('\n', ',').split(',') if t.strip()]
        
        if not text_samples:
            return ("No text prompts provided", {})
        
        # Move audio to device
        device = next(model.parameters()).device
        audio = audio.to(device)
        
        # Get local similarity
        with torch.no_grad():
            act_map_cross = model.get_local_similarity(
                audio,
                text_samples,
                method=method,
                cross_product=cross_product,
            ).cpu().numpy()
        
        # Apply median filtering for smoother results
        act_map_filter = []
        for i in range(act_map_cross.shape[0]):
            filtered = scipy.ndimage.median_filter(
                act_map_cross[i], 
                (1, median_filter_size)
            )
            act_map_filter.append(filtered)
        act_map_filter = np.array(act_map_filter)
        
        # Prepare similarity dictionary
        similarity_dict = {}
        for i, text in enumerate(text_samples):
            similarity_dict[text] = act_map_filter[0][i]
        
        # Create summary
        summary_lines = ["Local Similarity Analysis:\n" + "="*40]
        summary_lines.append(f"Method: {method}")
        summary_lines.append(f"Cross Product: {cross_product}")
        summary_lines.append(f"Text Prompts: {len(text_samples)}")
        summary_lines.append(f"Time Frames: {act_map_filter.shape[2]}")
        
        # Add statistics for each text prompt
        summary_lines.append("\nStatistics per text prompt:")
        for text, sim_map in similarity_dict.items():
            max_val = np.max(sim_map)
            mean_val = np.mean(sim_map)
            summary_lines.append(f"  {text}:")
            summary_lines.append(f"    Max: {max_val:.4f}, Mean: {mean_val:.4f}")
        
        summary_lines.append("="*40)
        summary = "\n".join(summary_lines)
        
        return (summary, similarity_dict)


class OpenFlamVisualizeSpectrogram:
    """
    Generate spectrogram visualization with FLAM similarity heatmap
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_data": ("AUDIO_DATA",),
                "similarity_maps": ("DICT",),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "visualize_spectrogram"
    CATEGORY = "audio/openflam"
    
    def visualize_spectrogram(self, audio_data, similarity_maps):
        """Generate spectrogram and heatmap visualization"""
        import matplotlib.pyplot as plt
        import matplotlib
        import io
        from PIL import Image
        
        # Use non-interactive backend for matplotlib
        matplotlib.use('Agg')
        
        # Get audio numpy array and info
        audio_np = audio_data["data"]
        sample_rate = audio_data["sample_rate"]
        duration = audio_data["duration"]
        
        # Generate visualization in memory
        from openflam.module.plot_utils import plot_spec
        from matplotlib.colors import LinearSegmentedColormap
        
        # Create figure with 2 subplots
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 8), dpi=150)
        
        # Plot spectrogram
        plt.sca(axes[0])
        plot_spec(audio_np, sample_rate, title="Audio spectrogram")
        
        # Plot heatmap
        labels_list = list(similarity_maps.keys())
        data_matrix = np.vstack([similarity_maps[lbl] for lbl in labels_list])
        n_labels, n_frames = data_matrix.shape
        
        im = axes[1].imshow(
            data_matrix,
            aspect="auto",
            origin="upper",
            extent=[0, duration, 0, n_labels],
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        
        # Draw horizontal white lines between rows
        for i in range(1, n_labels):
            axes[1].axhline(i, color="white", linewidth=0.8)
        
        axes[1].set_title("FLAM Output", fontsize=14)
        axes[1].set_yticks(np.arange(n_labels) + 0.5)
        axes[1].set_yticklabels(labels_list[::-1], fontsize=15)
        axes[1].set_xlabel("Time (s)", fontsize=10)
        
        # Add colorbar
        plt.tight_layout()
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.14, 0.015, 0.32])
        fig.colorbar(im, cax=cbar_ax)
        
        # Save figure to memory buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        
        # Load from buffer and convert to ComfyUI IMAGE format
        img = Image.open(buf).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np)[None, ...]  # Add batch dimension
        
        buf.close()
        
        return (img_tensor,)


# Node class mappings for export (used by __init__.py)
__all__ = [
    'OpenFlamLoadModel',
    'OpenFlamGlobalSimilarity', 
    'OpenFlamLocalSimilarity',
    'OpenFlamLoadAudio',
    'OpenFlamVisualizeSpectrogram'
]
