# Installation Guide - ComfyUI-OpenFlam

This guide provides detailed instructions for installing and configuring ComfyUI-OpenFlam.

## System Requirements

### Minimum Hardware
- **CPU**: Modern multi-core processor
- **RAM**: 8GB (16GB recommended)
- **Disk Space**: 5GB available
  - 2GB for OpenFLAM model
  - 1-2GB for PyTorch dependencies
  - Additional space for audio files

### Recommended Hardware
- **GPU**: NVIDIA with 6GB+ VRAM and CUDA support
- **RAM**: 16GB or higher
- **SSD**: For faster loading

### Software
- **Operating System**: Windows 10/11, Linux, macOS
- **Python**: 3.10 or higher
- **ComfyUI**: Latest version installed and working
- **PyTorch**: 2.6.0 - 2.8.0

## Step-by-Step Installation

### Method 1: Standard Installation (Recommended)

#### 1. Navigate to custom nodes directory
```bash
cd ComfyUI/custom_nodes
```

#### 2. Clone the repository
```bash
git clone https://github.com/yourusername/ComfyUI-OpenFlam.git
```

#### 3. Enter the directory
```bash
cd ComfyUI-OpenFlam
```

#### 4. Install dependencies

**On Windows (with ComfyUI virtual environment):**
```bash
..\..\..\python_embeded\python.exe -m pip install -r requirements.txt
```

**On Linux/macOS (with ComfyUI virtual environment):**
```bash
pip install -r requirements.txt
```

**In separate virtual environment:**
```bash
# Activate your venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

#### 5. Restart ComfyUI
Close and reopen ComfyUI to load the new nodes.

### Method 2: Manual Installation

If you prefer to install each component manually:

#### 1. Install PyTorch
Visit [pytorch.org](https://pytorch.org/get-started/locally/) and install the correct version for your system.

**Example for CUDA 11.8:**
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

**Example for CPU only:**
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

#### 2. Install other dependencies
```bash
pip install numpy librosa soundfile torchlibrosa lightning transformers==4.56.1 matplotlib scipy huggingface-hub
```

#### 3. Verify installation
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Installation Verification

### 1. Check that nodes are loaded
Start ComfyUI and look for the "audio/openflam" category in the node list. You should see:
- OpenFLAM Load Model
- OpenFLAM Load Audio
- OpenFLAM Global Similarity
- OpenFLAM Local Similarity

### 2. Quick test
1. Add the `OpenFLAM Load Model` node
2. Click "Queue Prompt" (without connecting other nodes)
3. The model should start downloading the checkpoint (~2GB)

### 3. Check logs
Look for any errors in ComfyUI logs:
```
[ComfyUI-OpenFlam] Loading OpenFLAM model...
[ComfyUI-OpenFlam] Model loaded successfully
```

## Common Troubleshooting

### Problem: "No module named 'openflam'"

**Solution**: The path is not configured correctly.

Verify the folder structure is:
```
ComfyUI/
└── custom_nodes/
    └── ComfyUI-OpenFlam/
        ├── __init__.py
        ├── nodes/
        │   └── openflam_nodes.py
        └── openflam/
            └── src/
                └── openflam/
```

### Problem: "CUDA out of memory"

**Solutions**:
1. Reduce audio duration to 5-7 seconds instead of 10
2. Close other applications using the GPU
3. Use CPU instead of GPU (slower but works)
4. Restart ComfyUI to free memory

### Problem: "torch version mismatch"

**Solution**: Uninstall and reinstall PyTorch with the correct version.
```bash
pip uninstall torch torchvision torchaudio
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
```

### Problem: Nodes don't appear in ComfyUI

**Solutions**:
1. Verify that `__init__.py` is in the main directory
2. Check ComfyUI logs for import errors
3. Ensure all dependencies are installed
4. Completely restart ComfyUI

### Problem: Slow model download

**Explanation**: The OpenFLAM checkpoint is ~2GB and downloads from HuggingFace.

**Solutions**:
1. Be patient, this is normal on first run
2. Check your internet connection
3. Model will be saved in `models/openflam/` for future use
4. Can manually download from [HuggingFace](https://huggingface.co/kechenadobe/OpenFLAM)

### Problem: "RuntimeError: MPS backend not supported"

**Explanation**: OpenFLAM doesn't currently support MPS (Apple Silicon).

**Solution**: The node will automatically use CPU on Mac with Apple Silicon.

### Problem: Errors with librosa or soundfile

**Solution**: Install required system dependencies.

**On Ubuntu/Debian:**
```bash
sudo apt-get install libsndfile1
```

**On macOS:**
```bash
brew install libsndfile
```

**On Windows**: Usually not necessary, but if there are issues:
```bash
pip install --upgrade soundfile librosa
```

## Advanced Configuration

### Changing Model Directory

Modify the `cache_dir` parameter in the `OpenFLAM Load Model` node to change where the checkpoint is saved:

```
Default: models/openflam
Custom: /path/to/your/models/openflam
```

### Optimizing for Multiple GPUs

OpenFLAM automatically uses the available GPU. To select a specific GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py  # Use GPU 0
CUDA_VISIBLE_DEVICES=1 python main.py  # Use GPU 1
```

### Configuring GPU Memory

If you have memory issues, you can limit VRAM usage:

```python
# Add at the beginning of openflam_nodes.py
import torch
torch.cuda.set_per_process_memory_fraction(0.8, 0)  # Use max 80% VRAM
```

## Updating

To update ComfyUI-OpenFlam to the latest version:

```bash
cd ComfyUI/custom_nodes/ComfyUI-OpenFlam
git pull
pip install -r requirements.txt --upgrade
```

Restart ComfyUI after updating.

## Uninstallation

To completely remove ComfyUI-OpenFlam:

```bash
# 1. Remove the directory
cd ComfyUI/custom_nodes
rm -rf ComfyUI-OpenFlam  # Linux/macOS
# or
rd /s /q ComfyUI-OpenFlam  # Windows

# 2. (Optional) Remove downloaded model
rm -rf ComfyUI/models/openflam
```

## Support and Resources

- **Issues**: [GitHub Issues](https://github.com/yourusername/ComfyUI-OpenFlam/issues)
- **OpenFLAM Documentation**: [GitHub](https://github.com/adobe-research/openflam)
- **ComfyUI**: [GitHub](https://github.com/comfyanonymous/ComfyUI)
- **Discussions**: [ComfyUI Matrix](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org)

## Next Steps

After successful installation:

1. Read [EXAMPLES.md](EXAMPLES.md) for practical examples
2. Check [README.md](README.md) for complete documentation
3. Experiment with your audio files!
