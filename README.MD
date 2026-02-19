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
│   │   │   │   ├── train.pth
│   │   │   │   ...
├── dataloaders
├── models
├── script
├── train.py
```