from .edm import dnnlib
import pickle
import torch
import numpy as np
import torch.nn as nn
import math
import yaml
from .ddpm.unet import create_model
from omegaconf import OmegaConf
import importlib

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
    def score(self, x, sigma):
        pass

    def tweedie(self, x, sigma):
        pass


class LatentDiffusionModel(nn.Module):
    def encode(self, x0):
        pass

    def decode(self, z0):
        pass

    def score(self, z, sigma):
        pass

    def tweedie(self, z, sigma):
        pass


# @register_model(name='sdv1.5')
# class StableDiffusion(LatentDiffusionModel):
#     def __init__(self, device='cuda'):
#         super().__init__()
#         pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
#         self.tokenizer = pipeline.tokenizer
#         self.text_encoder = pipeline.text_encoder
#         self.vae = pipeline.vae
#         self.scheduler = pipeline.scheduler
#         self.unet = pipeline.unet
#         self.device = device
#         self.prompt_embeds = self._text_features()
#
#     def _text_features(self, prompt='an image of human face'):
#         tokens = self.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=77).to(
#             self.device)
#         with torch.no_grad():
#             text_embeddings = self.text_encoder(tokens.input_ids)[0]
#         return text_embeddings
#
#     def _inverse_schedule(self, sigma):
#         z = np.log(sigma ** 2 + 1)
#         beta_d = 19.9
#         beta_min = 0.1
#         t = (- beta_min + np.sqrt(beta_min ** 2 + 2 * beta_d * z)) / beta_d
#         return t
#
#     def _forward(self, z, t):
#         noise_pred = self.unet(z, t, encoder_hidden_states=self.prompt_embeds, return_dict=False)[0]
#         return noise_pred
#
#     def encode(self, x0, sample=True):
#         if sample:
#             return self.vae.encode(x0).latent_dist.sample()
#         else:
#             return self.vae.encode(x0).latent_dist.mean
#
#     def decode(self, z0):
#         return self.vae.decode(z0)['sample']
#
#     def score(self, z, sigma):
#         d = self.tweedie(z, sigma)
#         return (d - z) / sigma ** 2
#
#     def tweedie(self, z, sigma):
#         t = torch.tensor(int(self._inverse_schedule(sigma) * 999 + 1 / 2), device=self.device).repeat(z.shape[0])
#         model_output = self._forward(z / np.sqrt(sigma ** 2 + 1), t)
#         return z - sigma * model_output.detach()
#

@register_model(name='test_ldm_ffhq')
class TestLatentDiffusion(LatentDiffusionModel):
    def encode(self, x0):
        return torch.randn(x0.shape[0], 4, 64, 64).to(x0.device)

    def decode(self, z0):
        return torch.randn(z0.shape[0], 3, 256, 256).to(z0.device)

    def score(self, z, sigma):
        return torch.randn_like(z).to(z.device)

    def tweedie(self, z, sigma):
        return torch.randn_like(z).to(z.device)


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

@register_model(name='ldm_ffhq')
class LatentDiffusionFFHQ(LatentDiffusionModel):
    def __init__(self, ldm_config='config/model/_ldm_ffhq.yaml', diffusion_path='checkpoint/ldm_ffhq.pt', device='cuda'):
        super().__init__()
        config = OmegaConf.load(ldm_config)
        self.model = load_model_from_config(config, diffusion_path).to(device)
        self.device = device
        self.model.requires_grad_(False)

    def encode(self, x0):
        return self.model.encode_first_stage(x0)

    def decode(self, z0):
        # return self.model.decode_first_stage(z0)
        return self.model.differentiable_decode_first_stage(z0)

    def score(self, x, sigma=2e-3):
        d = self.tweedie(x, sigma)
        return (d - x) / sigma ** 2

    def inverse_schedule(self, sigma):
        z = np.log(sigma ** 2 + 1)
        beta_d = 19.9
        beta_min = 0.1
        t = (- beta_min + np.sqrt(beta_min ** 2 + 2 * beta_d * z)) / beta_d
        return t

    def tweedie(self, x, sigma=2e-3):
        t = torch.tensor(int(self.inverse_schedule(sigma) * 999 + 1 / 2), device=self.device).repeat(x.shape[0])
        # model_output = self.model(x / math.sqrt(sigma ** 2 + 1), t.cpu(), None)
        model_output = self.model.apply_model(x / math.sqrt(sigma ** 2 + 1), t, None)
        # model_output = torch.split(model_output, x.shape[1], dim=1)
        return x - sigma * model_output.detach()


@register_model(name='ldm_imagenet')
class LatentDiffusionImageNet(LatentDiffusionModel):
    def __init__(self, ldm_config='config/model/_ldm_imagenet.yaml', diffusion_path='checkpoint/ldm_imagenet.pt', device='cuda'):
        super().__init__()
        config = OmegaConf.load(ldm_config)
        self.model = load_model_from_config(config, diffusion_path).to(device)
        self.device = device
        self.model.requires_grad_(False)

    def encode(self, x0):
        return self.model.encode_first_stage(x0)

    def decode(self, z0):
        # return self.model.decode_first_stage(z0)
        return self.model.differentiable_decode_first_stage(z0)

    def score(self, x, sigma=2e-3):
        d = self.tweedie(x, sigma)
        return (d - x) / sigma ** 2

    def inverse_schedule(self, sigma):
        z = np.log(sigma ** 2 + 1)
        beta_d = 19.9
        beta_min = 0.1
        t = (- beta_min + np.sqrt(beta_min ** 2 + 2 * beta_d * z)) / beta_d
        return t

    def tweedie(self, x, sigma=2e-3):
        t = torch.tensor(int(self.inverse_schedule(sigma) * 999 + 1 / 2), device=self.device).repeat(x.shape[0])
        # model_output = self.model(x / math.sqrt(sigma ** 2 + 1), t.cpu(), None)
        model_output = self.model.apply_model(x / math.sqrt(sigma ** 2 + 1), t, None)
        # model_output = torch.split(model_output, x.shape[1], dim=1)
        return x - sigma * model_output.detach()


@register_model(name='ffhq64')
class FFHQ64Model(DiffusionModel):
    def __init__(self, device='cuda'):
        super().__init__()
        self.net = self.load_pretrained_model(
            'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-ve.pkl'
        ).to(device)
        self.device = device

    def load_pretrained_model(self, url, device='cuda'):
        with dnnlib.util.open_url(url) as f:
            net = pickle.load(f)['ema'].to(device)
        return net

    def score(self, x, sigma=2e-3):
        d = self.tweedie(x, sigma)
        return (d - x) / sigma ** 2

    def tweedie(self, x, sigma=2e-3):
        return self.net(x, torch.as_tensor(sigma).to(self.device)).clip(-1, 1)


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


@register_model(name='ffhq256ddpm')
class FFHQ256DDPM(DiffusionModel):
    def __init__(self, model_config, schedule='linear', device='cuda'):
        super().__init__()
        self.net = create_model(**model_config).to(device)
        self.net.eval()
        self.device = device
        # betas = np.array(betas, dtype=np.float64)
        self.betas = get_named_beta_schedule(schedule, 1000)
        alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.u = torch.tensor(np.sqrt((1 - self.alphas_cumprod) / self.alphas_cumprod), device=self.device)
        # print(self.u)

    def score(self, x, sigma=2e-3):
        d = self.tweedie(x, sigma)
        return (d - x) / sigma ** 2

    def inverse_schedule(self, sigma):
        z = np.log(sigma ** 2 + 1)
        beta_d = 19.9
        beta_min = 0.1
        t = (- beta_min + np.sqrt(beta_min ** 2 + 2 * beta_d * z)) / beta_d
        return t

    def tweedie(self, x, sigma=2e-3):
        t = torch.tensor(int(self.inverse_schedule(sigma) * 999 + 1 / 2), device=self.device).repeat(x.shape[0])
        # print(torch.argmin((self.u-sigma).abs()), t)
        model_output = self.net(x / math.sqrt(sigma ** 2 + 1), t)
        # model_mean, pred_xstart = self.mean_processor.get_mean_and_xstart(x, t, model_output)
        model_output, model_var_values = torch.split(model_output, x.shape[1], dim=1)
        # print(x.abs().max(), model_output.abs().max())
        return x - sigma * model_output.detach()


@register_model(name='imagenet256ddpm')
class ImageNet256DDPM(DiffusionModel):
    def __init__(self, model_config, schedule='linear', device='cuda'):
        super().__init__()
        self.net = create_model(**model_config).to(device)
        self.net.eval()
        self.device = device
        # betas = np.array(betas, dtype=np.float64)
        self.betas = get_named_beta_schedule(schedule, 1000)
        alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.u = torch.tensor(np.sqrt((1 - self.alphas_cumprod) / self.alphas_cumprod), device=self.device)
        # print(self.u)

    def score(self, x, sigma=2e-3):
        d = self.tweedie(x, sigma)
        return (d - x) / sigma ** 2

    def inverse_schedule(self, sigma):
        z = np.log(sigma ** 2 + 1)
        beta_d = 19.9
        beta_min = 0.1
        t = (- beta_min + np.sqrt(beta_min ** 2 + 2 * beta_d * z)) / beta_d
        return t

    def tweedie(self, x, sigma=2e-3):
        t = torch.tensor(int(self.inverse_schedule(sigma) * 999 + 1 / 2), device=self.device).repeat(x.shape[0])
        # print(torch.argmin((self.u-sigma).abs()), t)
        model_output = self.net(x / math.sqrt(sigma ** 2 + 1), t)
        # model_mean, pred_xstart = self.mean_processor.get_mean_and_xstart(x, t, model_output)
        model_output, model_var_values = torch.split(model_output, x.shape[1], dim=1)
        # print(x.abs().max(), model_output.abs().max())
        return x - sigma * model_output.detach()

@register_model(name='blackhole64ddpm')
class BlackHole64DDPM(DiffusionModel):
    def __init__(self, model_config, schedule='linear', device='cuda'):
        super().__init__()
        self.net = create_model(**model_config).to(device)
        self.net.eval()
        self.device = device
        # betas = np.array(betas, dtype=np.float64)
        self.betas = get_named_beta_schedule(schedule, 1000)
        alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.u = torch.tensor(np.sqrt((1 - self.alphas_cumprod) / self.alphas_cumprod), device=self.device)
        # print(self.u)

    def score(self, x, sigma=2e-3):
        d = self.tweedie(x, sigma)
        return (d - x) / sigma ** 2

    def inverse_schedule(self, sigma):
        z = np.log(sigma ** 2 + 1)
        beta_d = 19.9
        beta_min = 0.1
        t = (- beta_min + np.sqrt(beta_min ** 2 + 2 * beta_d * z)) / beta_d
        return t

    def tweedie(self, x, sigma=2e-3):
        t = torch.tensor(int(self.inverse_schedule(sigma) * 999 + 1 / 2), device=self.device).repeat(x.shape[0])
        # print(torch.argmin((self.u-sigma).abs()), t)
        model_output = self.net(x / math.sqrt(sigma ** 2 + 1), t)
        # model_mean, pred_xstart = self.mean_processor.get_mean_and_xstart(x, t, model_output)
        model_output, model_var_values = torch.split(model_output, x.shape[1], dim=1)
        # print(x.abs().max(), model_output.abs().max())
        out = x - sigma * model_output.detach()
        return out