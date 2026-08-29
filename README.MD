# FDLD-UNet

FDLD-UNet is a frequency-domain and spatial-domain U-Net framework for medical image semantic segmentation. The project supports both 2D and 3D scenarios and includes several comparison models. The current main models are `D2_FDLD.py` and `D3_FDLD.py`.

## 1. Model Overview

FDLD-UNet integrates frequency-domain channel filtering and adaptive spatial filtering into a U-Net encoder-decoder architecture:

```text
Input image / volume
        ↓
Frequency-domain channel filtering
        ↓
Adaptive spatial filtering
        ↓
U-Net encoder
        ↓
Bottleneck feature enhancement
        ↓
U-Net decoder with skip connections
        ↓
Segmentation prediction
```

### 2D Main Model

File: `models/D2_FDLD.py`

Model class: `FDLD_UNet`

- Input: `[B, C, H, W]`
- Output: `[B, num_classes, H, W]`
- Uses 2D FFT and 2D deformable convolution
- Main components: `ChannelFilter`, `A_SpatialFilter`, `DoubleConv`, `Down`, and `Up`

### 3D Main Model

File: `models/D3_FDLD.py`

Model class: `FDLD`

- Input: `[B, C, D, H, W]`
- Output: `[B, num_classes, D, H, W]`
- Uses 3D FFT, 3D deformable sampling, and 3D convolution
- Main components: `ChannelFilter3D_ChannelFFT`, `SFAC`, `DeformConv3d`, `CFAE`, `Down3D`, and `Up3D`
- The encoder uses three stages with `down_factor=4`; input dimensions should be compatible with repeated 4x downsampling and upsampling

## 2. Comparison Models

The `D2_` and `D3_` prefixes identify 2D and 3D models respectively.

| Category | 2D Model | 3D Model | Description |
|---|---|---|---|
| Baseline U-Net | `D2_Unet.py` | `D3_Unet.py` | Standard U-Net |
| UNet++ | `D2_unetpp.py` | `D3_unetpp.py` | Nested skip connections |
| TransUNet | `D2_transunet.py` | `D3_transunet.py` | U-Net with Transformer/ViT features |
| UNeXt | `D2_Unext.py` | `D3_Unext.py` | Shift-MLP feature mixing |
| VM-UNet | `D2_vmunet.py` | `D3_vmunet.py` | Visual Mamba encoder |
| SwinUNETR | `D2_SwinUNETR.py` | `D3_SwinUNETR.py` | MONAI Swin Transformer |
| DynUNet | `D2_DynUNet.py` | `D3_DynUNet.py` | MONAI DynUNet |
| FlexUNet | `D2_FlexUNet.py` | `D3_FlexUNet.py` | MONAI FlexUNet |
| MHorUNet | `D2_MHorUnet.py` | `D3_MHorUnet.py` | Multi-order convolution and attention bridge |
| HFE-RWKV | — | `D3_HFE_RWKV.py` | High/low-frequency features with RWKV |
| ZIZ/RiR | `D2_ZIZ.py` | `D3_ZIZ.py` | RiR hierarchy and attention |
| VoxelMorph | `D2_voxelmorph.py` | `D3_voxelmorph.py` | Image registration baseline |
| Mamba-VoxelMorph | `D2_Mamba_VoxelMorph.py` | `D3_Mamba_VoxelMorph.py` | Mamba and VoxelMorph registration model |

## 3. Project Structure

```text
FDLD-UNet-master/
├── models/
│   ├── D2_FDLD.py              # 2D main model
│   ├── D3_FDLD.py              # 3D main model
│   ├── D2_*.py                 # 2D comparison models
│   ├── D3_*.py                 # 3D comparison models
│   ├── vmamba.py               # VMamba/VSSM components
│   ├── fdld_unet.py            # Original FDLD 2D implementation
│   ├── basicunet.py            # MONAI BasicUNet wrapper
│   └── __init__.py             # Model factory and exports
├── dataloaders/
│   └── folder_loader.py        # Folder-based image/mask loader
├── scripts/
│   └── train.sh                # Example training script
├── train.py                    # Training entry point
├── requirements.txt            # Python dependencies
└── README.MD
```

## 4. Dataset Organization

The current `FolderLoader` expects the following structure:

```text
data/<dataset_name>/
├── training/
│   ├── images/
│   └── masks/
├── validation/
│   ├── images/
│   └── masks/
└── testing/                    # Optional
    ├── images/
    └── masks/
```

Each image and mask must have the same filename. The 2D loader returns images with shape `[C, H, W]` and masks with shape `[1, H, W]`. A separate volume loader is required for 3D data, using the `[C, D, H, W]` layout for each sample.

## 5. Model Usage

### 2D FDLD

```python
import torch
from models.D2_FDLD import FDLD_UNet

model = FDLD_UNet(
    in_channels=3,
    num_classes=2,
    img_size=256,
    bilinear=True,
)

x = torch.randn(2, 3, 256, 256)
y = model(x)
print(y.shape)  # [2, 2, 256, 256]
```

### 3D FDLD

```python
import torch
from models.D3_FDLD import FDLD

model = FDLD(
    n_channels=1,
    n_classes=2,
    bilinear=True,
)

x = torch.randn(1, 1, 128, 128, 128)
y = model(x)
print(y.shape)  # [1, 2, 128, 128, 128]
```

## 6. Training

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the current 2D training pipeline:

```bash
python train.py \
  --input ./data/isic2017 \
  --output ./checkpoints/isic2017 \
  --model fdld_unet \
  --spatial_dims 2 \
  --in_channels 3 \
  --num_classes 2 \
  --img_size 256 \
  --batch_size 1 \
  --epochs 20 \
  --opt RMSprop \
  --lr 1e-4 \
  --momentum 0.9 \
  --weight_decay 1e-4 \
  --sched cosine
```

Alternatively:

```bash
bash scripts/train.sh
```

The current `train.py` uses the 2D folder loader and is configured for 2D RGB binary segmentation by default. Training `D3_FDLD.py` requires a dedicated 3D volume loader and training entry point.

## 7. Important Notes

- The models return logits. Apply softmax only during inference when class probabilities are required.
- The 2D frequency filters contain learnable parameters tied to `img_size`; the configured image size should match the actual input size.
- The 3D model has a high memory requirement. Start with a small batch size and smaller volumes when debugging.
- Use `[B, C, D, H, W]` consistently for 3D tensors.
- The training script automatically uses CUDA when available and falls back to CPU otherwise.
- `requirements.txt` must use the package name `torchvision`, not `tochvision`.

## 8. Main Files

- 2D main model: [`models/D2_FDLD.py`](models/D2_FDLD.py)
- 3D main model: [`models/D3_FDLD.py`](models/D3_FDLD.py)
- Training entry point: [`train.py`](train.py)
- Data loader: [`dataloaders/folder_loader.py`](dataloaders/folder_loader.py)
- Training script: [`scripts/train.sh`](scripts/train.sh)
