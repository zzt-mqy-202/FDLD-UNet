from pathlib import Path

import monai.transforms as MT
import numpy as np
import torch
from einops import rearrange
from monai.utils import InterpolateMode
from PIL import Image
from torch.utils.data import DataLoader, Dataset

__all__ = ['FolderDataset', 'FolderLoader',]


class FolderDataset(Dataset):
    def __init__(
        self,
        root: Path | str,
        img_size: int,
    ) -> None:
        super().__init__()
        self.root = Path(root).expanduser()
        self.img_size = img_size

        self.data = []
        for img_path in Path(self.root, 'images').iterdir():
            msk_path = self.root / 'masks' / img_path.name
            if not msk_path.exists():
                continue
            self.data.append({
                'img': img_path,
                'msk': msk_path,
            })

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> None:
        sample = self.data[index]

        img = Image.open(sample['img'])
        img = np.asarray(img) / 255
        img = torch.tensor(img).float()
        img = rearrange(img, 'h w c -> c h w')

        msk = Image.open(sample['msk']).convert('1')
        msk = np.asarray(msk)
        msk = torch.tensor(msk).float()
        msk = rearrange(msk, 'h w -> 1 h w')

        sample = {
            'img': img,
            'msk': msk,
        }

        spatial_size = (self.img_size, self.img_size)
        transformer = MT.Compose([
            MT.Resized(('img', 'msk'),
                       spatial_size=spatial_size,
                       mode=(InterpolateMode.AREA, InterpolateMode.NEAREST_EXACT)),
        ])
        sample = transformer(sample)

        return sample['img'].float(), sample['msk'].long()


class FolderLoader(DataLoader):
    def __init__(
        self,
        root,
        img_size: int = 256,
        **kwargs
    ) -> None:
        dataset = FolderDataset(root, img_size)
        super().__init__(dataset, **kwargs)
