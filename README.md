# ComfyUI-OpenFlam

Custom nodes for ComfyUI that integrate [OpenFLAM](https://github.com/adobe-research/openflam) - an advanced audio-text analysis model supporting both zero-shot sound event detection and large-scale audio retrieval via free-form text.

## Description

OpenFLAM (Framewise Language-Audio Modeling) is a cutting-edge model based on contrastive language-audio pretraining (CLAP) that enhances frame-wise event localization capabilities through learnable text and audio biases and scales.

This custom node brings OpenFLAM functionality directly into ComfyUI, enabling:
- **Global similarity analysis**: Compute clip-wise similarity between audio and text
- **Local event detection**: Identify and localize specific sound events over time
- **Zero-shot audio retrieval**: Find audio content based on textual descriptions

## Installation

1. Navigate to ComfyUI's custom nodes directory:
```bash
cd ComfyUI/custom_nodes
```

2. Clone this repository:
```bash
git clone https://github.com/yourusername/ComfyUI-OpenFlam.git
```

3. Install dependencies:
```bash
cd ComfyUI-OpenFlam
pip install -r requirements.txt
```

**Note**: OpenFLAM requires **PyTorch 2.6.0+** and supports CUDA GPUs. The model will automatically download pretrained checkpoints on first use.

## Available Nodes

### 1. OpenFLAM Load Model
Load the pretrained OpenFLAM model.

**Inputs:**
- `model_name`: Model name (currently supported: "v1-base")
- `cache_dir`: Directory to save model checkpoints (default: "models/openflam")

**Outputs:**
- `model`: OpenFLAM model instance

### 2. OpenFLAM Load Audio
Load and preprocess an audio file for OpenFLAM.

**Inputs:**
- `audio`: Audio file to load (supports: WAV, MP3, OGG, FLAC, M4A, AAC)
- `duration`: Audio clip duration in seconds (default: 10.0)

**Outputs:**
- `audio`: Preprocessed audio tensor at 48kHz

**Note**: OpenFLAM requires 48kHz audio. The node automatically handles resampling.

### 3. OpenFLAM Global Similarity
Compute global (clip-wise) similarity between audio and text.

**Inputs:**
- `model`: Loaded OpenFLAM model
- `audio`: Audio tensor to analyze
- `text_prompts`: Text prompts separated by comma or newline

**Outputs:**
- `results`: Formatted results with all similarity scores
- `best_match`: Text prompt with the highest score

**Usage example:**
Text prompts: `"breaking bones, metallic creak, tennis ball, female speaker"`

### 4. OpenFLAM Local Similarity
Compute local (frame-wise) similarity for sound event detection.

**Inputs:**
- `model`: Loaded OpenFLAM model
- `audio`: Audio tensor to analyze
- `text_prompts`: Text prompts separated by comma or newline
- `method`: Computation method ("unbiased" or "approximate")
- `cross_product`: Compute similarity between all audio and all texts
- `median_filter_size`: Median filter size for smoother results (default: 3)

**Outputs:**
- `summary`: Analysis summary with statistics
- `similarity_maps`: Dictionary with similarity maps for each text prompt

**Methods:**
- `unbiased`: Implements Eq. 9 from the paper (more accurate)
- `approximate`: Implements Eq. 8 from the paper (faster, valid when bias is small)

### 5. OpenFLAM Visualize Spectrogram
Generate a visual representation with audio spectrogram and FLAM similarity heatmap.

**Inputs:**
- `audio_data`: Audio data from Load Audio node
- `similarity_maps`: Similarity maps dictionary from Local Similarity node

**Outputs:**
- `image`: Image tensor compatible with ComfyUI (connect to Preview Image or Save Image nodes)

**Note**: The visualization includes:
- Top panel: Audio spectrogram (log-magnitude STFT)
- Bottom panel: FLAM output heatmap showing similarity over time for each text prompt

## Example Workflows

### Simple Global Analysis
1. `OpenFLAM Load Model` → load the model
2. `OpenFLAM Load Audio` → load audio file
3. `OpenFLAM Global Similarity` → compute similarity with prompts
4. View results in output

### Sound Event Detection
1. `OpenFLAM Load Model` → load the model
2. `OpenFLAM Load Audio` → load audio file
3. `OpenFLAM Local Similarity` → detect events over time
4. Use similarity maps for further processing

### Spectrogram Visualization
1. `OpenFLAM Load Model` → load the model
2. `OpenFLAM Load Audio` → load audio file (outputs audio, audio_data)
3. `OpenFLAM Local Similarity` → detect events (outputs similarity_maps)
4. `OpenFLAM Visualize Spectrogram` → generate visualization (outputs image)
5. `Preview Image` (native ComfyUI) → view spectrogram + heatmap
6. Optional: `Save Image` (native ComfyUI) → save to disk

## System Requirements

- **Python**: 3.10+
- **PyTorch**: 2.6.0 - 2.8.0
- **GPU**: CUDA recommended (also works on CPU but slower)
- **RAM**: Minimum 8GB (16GB recommended)
- **Disk Space**: ~2GB for pretrained model

## Technical Details

### Audio Configuration
- **Sample Rate**: 48kHz (required)
- **Standard Duration**: 10 seconds
- **Input Format**: Mono audio

### Embeddings
- **Global Dimension**: 512
- **Local Dimension**: Frame-wise with temporal resolution of ~0.1Hz

### Performance
- **GPU (CUDA)**: ~1-2 seconds per 10-second clip
- **CPU**: ~10-20 seconds per 10-second clip

## Citation

If you use OpenFLAM in your work, please cite the original paper:

```bibtex
@inproceedings{flam2025,
    title={{FLAM}: Frame-Wise Language-Audio Modeling},
    author={Yusong Wu and Christos Tsirigotis and Ke Chen and Cheng-Zhi Anna Huang and Aaron Courville and Oriol Nieto and Prem Seetharaman and Justin Salamon},
    booktitle={Forty-second International Conference on Machine Learning (ICML)},
    year={2025},
    url={https://openreview.net/forum?id=7fQohcFrxG}
}
```

## License

Both code and models for OpenFLAM are released under a non-commercial [Adobe Research License](./openflam/LICENSE). Please review it carefully before using this technology.

## Useful Links

- [OpenFLAM Paper](https://arxiv.org/abs/2505.05335)
- [Original Repository](https://github.com/adobe-research/openflam)
- [FLAM Website](https://flam-model.github.io/)
- [HuggingFace Checkpoint](https://huggingface.co/kechenadobe/OpenFLAM)

## Known Issues

- The model requires PyTorch 2.6.0+, ensure you have the correct version installed
- First run will automatically download the checkpoint (~2GB) - this may take time
- MPS (Apple Silicon) is not currently supported by OpenFLAM

## Support

For issues specific to this custom node, please open an issue on this repository.
For OpenFLAM-related issues, please refer to the [original repository](https://github.com/adobe-research/openflam).

## Credits

- **OpenFLAM**: Ke Chen, Yusong Wu, Oriol Nieto, Prem Seetharaman @ Adobe Research
- **ComfyUI Integration**: [Your name]
