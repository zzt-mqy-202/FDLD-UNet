# FDLD_UNet

## File Structure
```
├── data
│   ├── isic2017
│   │   ├── training
│   │   │   ├── images
│   │   │   │   ├── a.png
│   │   │   │   ├── b.png
│   │   │   │   ...
│   │   │   ├── masks
│   │   │   │   ├── a.png
│   │   │   │   ├── b.png
│   │   │   │   ...
│   │   ├── validation
│   │   └── testing
│   ...
├── checkpoints
│   ├── isic2017
│   │   ├── fdld_unet
│   │   │   ├── train.pth
│   │   │   ├── epoch_000.pth
│   │   │   ├── epoch_050.pth
│   │   │   ...
│   │   ...
│   ...
├── dataloaders
├── models
├── scripts
└── train.py
```
