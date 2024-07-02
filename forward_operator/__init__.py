from abc import ABC, abstractmethod
from .resizer import Resizer
import torch.nn.functional as F
from .fastmri_utils import fft2c_new
from .motionblur.motionblur import Kernel

import torch
import torch.nn as nn
import scipy
import numpy as np
import yaml

__OPERATOR__ = {}


def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls

    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class Operator(ABC):
    def __init__(self, sigma=0.05):
        self.sigma = sigma

    @abstractmethod
    def __call__(self, x):
        pass

    def measure(self, x):
        y0 = self(x)
        return y0 + self.sigma * torch.randn_like(y0)

    def error(self, x, y):
        return ((self(x) - y) ** 2).flatten(1).sum(-1)

    def log_likelihood(self, x, y):
        return -self.error(x, y) / 2 / self.sigma ** 2

    def likelihood(self, x, y):
        return torch.exp(self.log_likelihood(x, y))


# Linear Operator
@register_operator(name='down_sampling')
class DownSampling(Operator):
    def __init__(self, resolution=256, scale_factor=4, device='cuda', sigma=0.05):
        super().__init__(sigma)
        in_shape = [1, 3, resolution, resolution]
        self.down_sample = Resizer(in_shape, 1 / scale_factor).to(device)

    def __call__(self, x):
        return self.down_sample(x)


def random_sq_bbox(img, mask_shape, image_size=256, margin=(16, 16)):
    """Generate a random sqaure mask for inpainting
    """
    B, C, H, W = img.shape
    h, w = mask_shape
    margin_height, margin_width = margin
    maxt = image_size - margin_height - h
    maxl = image_size - margin_width - w

    # bb
    t = np.random.randint(margin_height, maxt)
    l = np.random.randint(margin_width, maxl)

    # make mask
    mask = torch.ones([B, C, H, W], device=img.device)
    mask[..., t:t + h, l:l + w] = 0

    return mask, t, t + h, l, l + w


class mask_generator:
    def __init__(self, mask_type, mask_len_range=None, mask_prob_range=None,
                 image_size=256, margin=(32, 32)):
        """
        (mask_len_range): given in (min, max) tuple.
        Specifies the range of box size in each dimension
        (mask_prob_range): for the case of random masking,
        specify the probability of individual pixels being masked
        """
        assert mask_type in ['box', 'random', 'both', 'extreme']
        self.mask_type = mask_type
        self.mask_len_range = mask_len_range
        self.mask_prob_range = mask_prob_range
        self.image_size = image_size
        self.margin = margin

    def _retrieve_box(self, img):
        l, h = self.mask_len_range
        l, h = int(l), int(h)
        mask_h = np.random.randint(l, h)
        mask_w = np.random.randint(l, h)
        mask, t, tl, w, wh = random_sq_bbox(img,
                                            mask_shape=(mask_h, mask_w),
                                            image_size=self.image_size,
                                            margin=self.margin)
        return mask, t, tl, w, wh

    def _retrieve_random(self, img):
        total = self.image_size ** 2
        # random pixel sampling
        l, h = self.mask_prob_range
        prob = np.random.uniform(l, h)
        mask_vec = torch.ones([1, self.image_size * self.image_size])
        samples = np.random.choice(self.image_size * self.image_size, int(total * prob), replace=False)
        mask_vec[:, samples] = 0
        mask_b = mask_vec.view(1, self.image_size, self.image_size)
        mask_b = mask_b.repeat(3, 1, 1)
        mask = torch.ones_like(img, device=img.device)
        mask[:, ...] = mask_b
        return mask

    def __call__(self, img):
        if self.mask_type == 'random':
            mask = self._retrieve_random(img)
            return mask
        elif self.mask_type == 'box':
            mask, t, th, w, wl = self._retrieve_box(img)
            return mask
        elif self.mask_type == 'extreme':
            mask, t, th, w, wl = self._retrieve_box(img)
            mask = 1. - mask
            return mask


@register_operator(name='inpainting')
class Inpainting(Operator):
    def __init__(self, mask_type, mask_len_range=None, mask_prob_range=None, resolution=256, device='cuda',
                 sigma=0.05):
        super().__init__(sigma)
        self.mask_gen = mask_generator(mask_type, mask_len_range, mask_prob_range, resolution)
        self.mask = None  # [B, 1, H, W]

    def __call__(self, x):
        if self.mask is None:
            self.mask = self.mask_gen(x)
            self.mask = self.mask[0:1, 0:1, :, :]
        return x * self.mask


class Blurkernel(nn.Module):
    def __init__(self, blur_type='gaussian', kernel_size=31, std=3.0, device=None):
        super().__init__()
        self.blur_type = blur_type
        self.kernel_size = kernel_size
        self.std = std
        self.device = device
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(self.kernel_size // 2),
            nn.Conv2d(3, 3, self.kernel_size, stride=1, padding=0, bias=False, groups=3)
        )

        self.weights_init()

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        if self.blur_type == "gaussian":
            n = np.zeros((self.kernel_size, self.kernel_size))
            n[self.kernel_size // 2, self.kernel_size // 2] = 1
            k = scipy.ndimage.gaussian_filter(n, sigma=self.std)
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)
        elif self.blur_type == "motion":
            k = Kernel(size=(self.kernel_size, self.kernel_size), intensity=self.std).kernelMatrix
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)

    def update_weights(self, k):
        if not torch.is_tensor(k):
            k = torch.from_numpy(k).to(self.device)
        for name, f in self.named_parameters():
            f.data.copy_(k)

    def get_kernel(self):
        return self.k


@register_operator(name='gaussian_blur')
class GaussianBlur(Operator):
    def __init__(self, kernel_size, intensity, device='cuda', sigma=0.05):
        super().__init__(sigma)
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='gaussian',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))
        self.conv.requires_grad_(False)

    def __call__(self, data):
        return self.conv(data)


@register_operator(name='motion_blur')
class MotionBlur(Operator):
    def __init__(self, kernel_size, intensity, device='cuda', sigma=0.05):
        super().__init__(sigma)
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='motion',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)  # should we keep this device term?

        self.kernel = Kernel(size=(kernel_size, kernel_size), intensity=intensity)
        kernel = torch.tensor(self.kernel.kernelMatrix, dtype=torch.float32)
        self.conv.update_weights(kernel)
        self.conv.requires_grad_(False)

    def __call__(self, data):
        # A^T * A
        return self.conv(data)


# Non-linear Operator
@register_operator(name='phase_retrieval')
class PhaseRetrieval(Operator):
    def __init__(self, oversample=0.0, resolution=256, sigma=0.05):
        super().__init__(sigma)
        self.pad = int((oversample / 8.0) * resolution)

    def __call__(self, x):
        x = x * 0.5 + 0.5  # [-1, 1] -> [0, 1]
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad))
        if not torch.is_complex(x):
            x = x.type(torch.complex64)
        fft2_m = torch.view_as_complex(fft2c_new(torch.view_as_real(x)))
        amplitude = fft2_m.abs()
        # amplitude = (amplitude - amplitude.min()) / (amplitude.max() - amplitude.min())
        return amplitude


@register_operator(name='nonlinear_blur')
class NonlinearBlur(Operator):
    def __init__(self, opt_yml_path, device='cuda', sigma=0.05):
        super().__init__(sigma)
        self.device = device
        self.blur_model = self.prepare_nonlinear_blur_model(opt_yml_path)
        self.blur_model.requires_grad_(False)

        np.random.seed(0)
        kernel_np = np.random.randn(1, 512, 2, 2) * 1.2
        random_kernel = (torch.from_numpy(kernel_np)).float().to(self.device)
        self.random_kernel = random_kernel

    def prepare_nonlinear_blur_model(self, opt_yml_path):
        from .bkse.models.kernel_encoding.kernel_wizard import KernelWizard

        with open(opt_yml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = opt["pretrained"]
        blur_model = KernelWizard(opt)
        blur_model.eval()
        blur_model.load_state_dict(torch.load(model_path))
        blur_model = blur_model.to(self.device)
        self.random_kernel = torch.randn(1, 512, 2, 2).to(self.device) * 1.2
        return blur_model

    def call_old(self, data):
        # random_kernel = torch.randn(1, 512, 2, 2).to(self.device) * 1.2
        data = (data + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        blurred = []
        for i in range(data.shape[0]):
            single_blurred = self.blur_model.adaptKernel(data[i:i + 1], kernel=self.random_kernel)
            blurred.append(single_blurred)
        blurred = torch.cat(blurred, dim=0)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1)  # [0, 1] -> [-1, 1]
        return blurred

    def __call__(self, data):
        data = (data + 1.0) / 2.0  # [-1, 1] -> [0, 1]

        random_kernel = self.random_kernel.repeat(data.shape[0], 1, 1, 1)
        blurred = self.blur_model.adaptKernel(data, kernel=random_kernel)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1)  # [0, 1] -> [-1, 1]

        # blurred = []
        # for i in range(data.shape[0]):
        #     single_blurred = self.blur_model.adaptKernel(data[i:i + 1], kernel=self.random_kernel)
        #     blurred.append(single_blurred)
        # blurred = torch.cat(blurred, dim=0)
        # blurred = (blurred * 2.0 - 1.0).clamp(-1, 1)  # [0, 1] -> [-1, 1]
        return blurred


@register_operator(name='high_dynamic_range')
class HighDynamicRange(Operator):
    def __init__(self, device='cuda', scale=2, sigma=0.05):
        super().__init__(sigma)
        self.device = device
        self.scale = scale

    def __call__(self, data):
        return torch.clip((data * self.scale), -1, 1)
