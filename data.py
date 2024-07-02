from abc import ABC, abstractmethod
from PIL import Image
import torchvision.transforms as transforms
import torch
from pathlib import Path

__DATASET__ = {}


def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls

    return wrapper


def get_dataset(name: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __DATASET__[name](**kwargs)


class DiffusionData(ABC):
    @abstractmethod
    def get_shape(self):
        pass

    @abstractmethod
    def get_data(self, size=16, sigma=2e-3):
        pass

    def get_random(self, size=16, sigma=2e-3):
        shape = (size, *self.get_shape())
        return torch.randn(shape) * sigma


@register_dataset('image')
class ImageDataset(DiffusionData):
    """
        A concrete class for handling image datasets, inherits from DiffusionData.

        This class is responsible for loading images from a specified directory,
        applying transformations to center crop the squared images of given resolution.

        Supported extension : ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
        Output data range   : [-1, 1]
    """

    def __init__(self, root='dataset/demo', resolution=256, device='cuda', start_id=None, end_id=None):
        # Define the file extensions to search for
        extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
        self.data = [file for ext in extensions for file in Path(root).rglob(ext)]
        self.data = sorted(self.data)

        # Subset the dataset
        self.data = self.data[start_id: end_id]
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution)
        ])
        self.res = resolution
        self.device = device

    def load_data(self, i):
        return (self.trans(Image.open(self.data[i])) * 2 - 1).to(self.device)

    def get_shape(self):
        return (3, self.res, self.res)

    def get_data(self, size=16, sigma=2e-3):
        data = torch.stack([self.load_data(i) for i in range(size)], dim=0)
        return data + torch.randn_like(data) * sigma

    def __len__(self):
        return len(self.data)

@register_dataset('empty')
class Empty(DiffusionData):
    """
        A concrete class for unknown ground truth images.

        Output data range   : [-1, 1]
    """

    def __init__(self, shape, device='cuda'):
        self.shape = shape
        self.device = device

    def load_data(self, i):
        return torch.zeros(self.shape).cuda()

    def get_shape(self):
        return self.shape

    def get_data(self, size=16, sigma=2e-3):
        data = torch.stack([self.load_data(i) for i in range(size)], dim=0)
        return data + torch.randn_like(data) * sigma

    def __len__(self):
        return 1
