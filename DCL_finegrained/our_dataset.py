import torch as t
from torch.utils import data
from PIL import Image
import os


class CUB(data.Dataset):
    def __init__(self, root, transforms=None):
        self.imgs = []
        self.transforms = transforms

        for cls in os.listdir(root):
            for img_name in os.listdir(os.path.join(root, cls)):
                img_path = os.path.join(root, cls, img_name)
                self.imgs.append(img_path)


    def __getitem__(self, index):
        imgs = self.imgs[index]
        data = Image.open(imgs).convert('RGB')
        if self.transforms:
            data = self.transforms(data)

        return data

    def __len__(self):
        return len(self.imgs)