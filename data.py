from abc import ABC, abstractmethod
import PIL
import torchvision.transforms as transforms
import torch
from pathlib import Path
import numpy as np

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


@register_dataset('ffhq')
class FFHQ(DiffusionData):
    def __init__(self, root='dataset/ffhq256', resolution=256, device='cuda', start_id=None, end_id=None):
        self.data = sorted(list(Path(root).glob('*.png')))
        self.data = self.data[start_id: end_id]
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resolution)
        ])
        self.res = resolution
        self.device = device

    def load_data(self, i):
        return (self.trans(PIL.Image.open(self.data[i])) * 2 - 1).to(self.device)

    def get_shape(self):
        return (3, self.res, self.res)

    def get_data(self, size=16, sigma=2e-3):
        data = torch.stack([self.load_data(i) for i in range(size)], dim=0)
        return data + torch.randn_like(data) * sigma


@register_dataset('imagenet')
class ImageNet(DiffusionData):
    def __init__(self, root='dataset/imagenet256', resolution=256, device='cuda', start_id=None, end_id=None):
        self.data = sorted(list(Path(root).glob('*.JPEG')))
        self.data = self.data[start_id: end_id]
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution)
        ])
        self.res = resolution
        self.device = device

    def load_data(self, i):
        return (self.trans(PIL.Image.open(self.data[i])) * 2 - 1).to(self.device)

    def get_shape(self):
        return (3, self.res, self.res)

    def get_data(self, size=16, sigma=2e-3):
        data = torch.stack([self.load_data(i) for i in range(size)], dim=0)
        return data + torch.randn_like(data) * sigma


@register_dataset('imagenet-filtered')
class ImageNetFiltered(DiffusionData):
    def __init__(self, root='dataset/imagenet256', meta_file='dataset/imagenet256/meta.txt', resolution=256, device='cuda', start_id=None, end_id=None):
        with open(meta_file, 'r') as file:
            lines = file.readlines()
        subset_path_name = [line.split(' ')[0] for line in lines]
        self.data = [Path(root) / path for path in subset_path_name]

        self.data = self.data[start_id: end_id]
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution)
        ])
        self.res = resolution
        self.device = device

    def load_data(self, i):
        return (self.trans(PIL.Image.open(self.data[i])) * 2 - 1).to(self.device)

    def get_shape(self):
        return (3, self.res, self.res)

    def get_data(self, size=16, sigma=2e-3):
        data = torch.stack([self.load_data(i) for i in range(size)], dim=0)
        return data + torch.randn_like(data) * sigma


@register_dataset('lsun_bedroom')
class LSUNBedroom(DiffusionData):
    def __init__(self, root='dataset/lsun_bedroom256', resolution=256, device='cuda', start_id=None, end_id=None):
        self.data = sorted(list((Path(root)).rglob('*.jpg')))
        self.data = self.data[start_id: end_id]
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution)
        ])
        self.res = resolution
        self.device = device

    def load_data(self, i):
        return (self.trans(PIL.Image.open(self.data[i])) * 2 - 1).to(self.device)

    def get_shape(self):
        return (3, self.res, self.res)

    def get_data(self, size=16, sigma=2e-3):
        data = torch.stack([self.load_data(i) for i in range(size)], dim=0)
        return data + torch.randn_like(data) * sigma


@register_dataset('celebA')
class CelebA(DiffusionData):
    def __init__(self, root='dataset/celebA256', resolution=256, device='cuda', start_id=None, end_id=None):
        self.data = sorted(list((Path(root)).rglob('*.jpg')))
        self.data = self.data[start_id: end_id]
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution)
        ])
        self.res = resolution
        self.device = device

    def load_data(self, i):
        return (self.trans(PIL.Image.open(self.data[i])) * 2 - 1).to(self.device)

    def get_shape(self):
        return (3, self.res, self.res)

    def get_data(self, size=16, sigma=2e-3):
        data = torch.stack([self.load_data(i) for i in range(size)], dim=0)
        return data + torch.randn_like(data) * sigma

@register_dataset('empty')
class Empty(DiffusionData):
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
