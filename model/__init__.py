from .edm import dnnlib
import pickle
import torch
import torch.nn as nn
from .ddpm.unet import create_model
from omegaconf import OmegaConf
import importlib
from abc import abstractmethod
from .precond import VPPrecond, LDM
import sys

__MODEL__ = {}


def register_model(name: str):
    def wrapper(cls):
        if __MODEL__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __MODEL__[name] = cls
        return cls

    return wrapper


def get_model(name: str, **kwargs):
    if __MODEL__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __MODEL__[name](**kwargs)




class DiffusionModel(nn.Module):
    """
    A class representing a diffusion model.
    Methods:
        score(x, sigma): Calculates the score of the diffusion model given the input `x` and standard deviation `sigma`.
        tweedie(x, sigma): Calculates the Tweedie distribution given the input `x` and standard deviation `sigma`.
        Must overload either `score` or `tweedie` method.
    """

    def __init__(self):
        super(DiffusionModel, self).__init__()
        # Check if either `score` or `tweedie` is overridden
        if (self.score.__func__ is DiffusionModel.score and
            self.tweedie.__func__ is DiffusionModel.tweedie):
            raise NotImplementedError(
                "Either `score` or `tweedie` method must be implemented."
            )

    def score(self, x, sigma):
        d = self.tweedie(x, sigma)
        return (d - x) / sigma ** 2

    def tweedie(self, x, sigma):
        return x + self.score(x, sigma) * sigma**2


class LatentDiffusionModel(nn.Module):
    """
    A class representing a latent diffusion model.
    Methods:
        encode(x0): Encodes the input `x0` into latent space.
        decode(z0): Decodes the latent variable `z0` into the output space.
        score(z, sigma): Calculates the score of the latent diffusion model given the latent variable `z` and standard deviation `sigma`.
        tweedie(z, sigma): Calculates the Tweedie distribution given the latent variable `z` and standard deviation `sigma`.
        Must overload either `score` or `tweedie` method.
    """ 
    def __init__(self):
        super(LatentDiffusionModel, self).__init__()
        # Check if either `score` or `tweedie` is overridden
        if (self.score.__func__ is LatentDiffusionModel.score and
            self.tweedie.__func__ is LatentDiffusionModel.tweedie):
            raise NotImplementedError(
                "Either `score` or `tweedie` method must be implemented."
            )
    
    @abstractmethod
    def encode(self, x0):
        pass

    @abstractmethod
    def decode(self, z0):
        pass

    def score(self, z, sigma):
        d = self.tweedie(z, sigma)
        return (d - z) / sigma ** 2

    def tweedie(self, z, sigma):
        return z + self.score(z, sigma) * sigma**2

@register_model(name='ddpm')
class DDPM(DiffusionModel):
    """
    DDPM (Diffusion Denoising Probabilistic Model)
    Attributes:
        model (VPPrecond): The neural network used for denoising.

    Methods:
        __init__(self, model_config, device='cuda'): Initializes the DDPM object.
        tweedie(self, x, sigma=2e-3): Applies the DDPM model to denoise the input, using VP preconditioning from EDM.
    """

    def __init__(self, model_config, device='cuda'):
        super().__init__()
        self.model = VPPrecond(model=create_model(**model_config),learn_sigma=model_config['learn_sigma'],conditional=model_config['class_cond']).to(device)
        self.model.eval()

    def tweedie(self, x, sigma=2e-3):
        return self.model(x, torch.as_tensor(sigma).to(x.device))


@register_model(name='edm')
class EDM(DiffusionModel):
    """
    Diffusion models from EDM (Elucidating the Design Space of Diffusion-Based Generative Models).
    """

    def __init__(self, model_config, device='cuda'):
        super().__init__()
        self.model = self.load_pretrained_model(model_config['model_path'],device=device)

    def load_pretrained_model(self, url, device='cuda'):
        with dnnlib.util.open_url(url) as f:
            sys.path.append('model/edm')
            model = pickle.load(f)['ema'].to(device)
        return model

    def tweedie(self, x, sigma=2e-3):
        return self.model(x, torch.as_tensor(sigma).to(x.device))


@register_model(name='ldm_ddpm')
class LatentDDPM(LatentDiffusionModel):
    """
    Latent Diffusion Models (High-Resolution Image Synthesis with Latent Diffusion Models).
    """
    def __init__(self, ldm_config, diffusion_path, device='cuda'):
        super().__init__()
        config = OmegaConf.load(ldm_config)
        net = LDM(load_model_from_config(config, diffusion_path)).to(device)
        self.model = VPPrecond(model=net).to(device)
        self.model.requires_grad_(False)

    def encode(self, x0):
        return self.model.model.encode(x0)

    def decode(self, z0):
        return self.model.model.decode(z0)

    def tweedie(self, x, sigma=2e-3):
        return self.model(x, torch.as_tensor(sigma).to(x.device))




def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def load_model_from_config(config, ckpt, train=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)  # , map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    _, _ = model.load_state_dict(sd, strict=False)

    model.cuda()

    if train:
        model.train()
    else:
        model.eval()

    return model
