import torch
from PIL import Image
import os

# GPU info
print("="*50)
print("GPU STATUS")
print("="*50)
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")

# Check a sample image
print("\n" + "="*50)
print("SAMPLE IMAGE INFO")
print("="*50)
img_path = "c:/Users/94775/Desktop/Component_1/cardio_image_384/train/positive"
sample = os.listdir(img_path)[0]
img = Image.open(os.path.join(img_path, sample))
print(f"File: {sample}")
print(f"Size: {img.size}")
print(f"Mode: {img.mode}")
print(f"Format: {img.format}")

# Check if timm is installed
try:
    import timm
    print(f"\ntimm version: {timm.__version__}")
    models = timm.list_models('convnext_base*')
    print(f"ConvNext-Base models available: {models}")
except ImportError:
    print("\ntimm NOT installed")

# Check sklearn
try:
    import sklearn
    print(f"scikit-learn version: {sklearn.__version__}")
except ImportError:
    print("scikit-learn NOT installed")
