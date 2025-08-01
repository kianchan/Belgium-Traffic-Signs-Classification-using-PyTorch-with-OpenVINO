from PIL import Image
import numpy as np
import torch

class Resize:
    def __init__(self, size):
        self.size = size
    def __call__(self, img):
        return img.resize(self.size, Image.BILINEAR)

class ToTensor:
    def __call__(self, img):
        arr = np.array(img, dtype=np.float32) / 255.0
        if len(arr.shape) == 2:  # grayscale
            arr = np.expand_dims(arr, axis=-1)
        arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
        return torch.tensor(arr, dtype=torch.float32)

class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
    def __call__(self, tensor):
        return (tensor - self.mean) / self.std

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

def get_transforms():
    return Compose([
        Resize((64, 64)),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
