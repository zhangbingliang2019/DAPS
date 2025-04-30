import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import warnings
import importlib
from abc import abstractmethod
from .ddpm.unet import create_model
from .precond import VPPrecond, LatentDMWrapper
from diffusers import StableDiffusionPipeline
from cores.scheduler import VPScheduler


__MODEL__ = {}


def register_model(name: str):
    def wrapper(cls):
        if name in __MODEL__ and __MODEL__[name] != cls:
            warnings.warn(f"Model '{name}' is already registered.", UserWarning)
        __MODEL__[name] = cls
        cls.name = name
        return cls
    return wrapper


def get_model(name: str, **kwargs):
    if name not in __MODEL__:
        raise NameError(f"Model '{name}' is not registered.")
    return __MODEL__[name](**kwargs)


class DiffusionModel(nn.Module):
    """
    Base Diffusion Model class.
    Requires overriding either 'score' or 'tweedie' method.
    """

    def __init__(self):
        super(DiffusionModel, self).__init__()
        if (self.score.__func__ is DiffusionModel.score and
            self.tweedie.__func__ is DiffusionModel.tweedie):
            raise NotImplementedError("Either 'score' or 'tweedie' method must be overridden.")

    def score(self, x, sigma):
        """
        Compute the score function \nabla_{x_t} log p(x_t; sigma_t).

        Args:
            x (Tensor): Noisy input tensor at time t, shape [B, *data_shape].
            sigma (float): Noise level at time t.
        """
        d = self.tweedie(x, sigma)
        return (d - x) / sigma ** 2

    def tweedie(self, x, sigma):
        """
        Compute the expected clean data given noisy data.

        Args:
            x (Tensor): Noisy input tensor at time t, shape [B, *data_shape].
            sigma (float): Noise level at time t.
        """
        return x + self.score(x, sigma) * sigma ** 2

    def get_in_shape(self):
        """Return the shape of the model's input data."""
        pass


class LatentDiffusionModel(nn.Module):
    """
    Base Latent Diffusion Model class.
    Requires overriding either 'score' or 'tweedie' method and 'encode', 'decode' methods.
    """

    def __init__(self):
        super(LatentDiffusionModel, self).__init__()
        if (self.score.__func__ is LatentDiffusionModel.score and
            self.tweedie.__func__ is LatentDiffusionModel.tweedie):
            raise NotImplementedError("Either 'score' or 'tweedie' method must be overridden.")

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
        return z + self.score(z, sigma) * sigma ** 2

    def get_in_shape(self):
        pass


@register_model(name='ddpm')
class DDPM(DiffusionModel):
    """
    DDPM (Diffusion Denoising Probabilistic Model).
    Attributes:
        model (VPPrecond): The neural network used for denoising.

    Methods:
        __init__(self, model_config, device='cuda'): Initializes the DDPM object.
        tweedie(self, x, sigma): Applies the DDPM model to denoise the input, using VP preconditioning from EDM.
    """

    def __init__(self, model_config, device='cuda', requires_grad=False):
        super().__init__()
        self.model = VPPrecond(model=create_model(**model_config), learn_sigma=model_config['learn_sigma'],
                               conditional=model_config['class_cond']).to(device)
        self.model.eval()
        self.model.requires_grad_(requires_grad)
        self.image_size = model_config['image_size']

    def tweedie(self, x, sigma):
        return self.model(x, torch.as_tensor(sigma).to(x.device))

    def get_in_shape(self):
        return (3, self.image_size, self.image_size)
        

@register_model(name='ldm')
class LDM(LatentDiffusionModel):
    """
    Latent Diffusion Model (LDM).

    Attributes:
        net (LatentDMWrapper):
            Wrapper around the latent diffusion model's internal network (encoder/decoder).
        model (VPPrecond):
            Diffusion model wrapped with VP preconditioning.
        is_conditional (bool):
            Indicates whether the model is conditional or unconditional.
    """

    def __init__(self, ldm_config, diffusion_path, device='cuda', requires_grad=False):
        super().__init__()
        self.net = LatentDMWrapper(load_model_from_config(ldm_config, diffusion_path)).to(device)
        self.is_conditional = not (ldm_config.model.params.cond_stage_config == '__is_unconditional__')
        label_dim = 1 if self.is_conditional else 0
        self.model = VPPrecond(label_dim=label_dim, model=self.net, conditional=self.is_conditional).to(device)
        self.image_size = ldm_config.model.params.image_size
        self.latent_channels = ldm_config.model.params.first_stage_config.params.embed_dim
        self.model.eval()
        self.model.requires_grad_(requires_grad)

    def encode(self, x0):
        return self.model.model.encode(x0)

    def decode(self, z0):
        return self.model.model.decode(z0)

    def tweedie(self, x, sigma):
        class_labels = None
        if self.is_conditional:
            # for ImageNet checkpoint
            class_labels = self.net.get_condition(torch.as_tensor([1000]).to(x.device))
        return self.model(x, torch.as_tensor(sigma).to(x.device), class_labels=class_labels)

    def get_in_shape(self):
        return (self.latent_channels, self.image_size, self.image_size)


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


@register_model(name='sdm')
class StableDiffusionModel(LatentDiffusionModel):
    """
    Stable Diffusion Model (SD) with fixed text prompts.

    Attributes:
        pipe (StableDiffusionPipeline): Hugging Face diffusion pipeline.
        vae (nn.Module): Variational Autoencoder for encoding/decoding images.
        unet (nn.Module): U-Net diffusion network for denoising.
        scheduler (VPScheduler): Scheduler for diffusion timesteps.
    """
    def __init__(self, model_id = "stabilityai/stable-diffusion-2-1", inner_resolution=768, target_resolution=256, guidance_scale=7.5, prompt='a natural looking human face', device='cuda', hf_home='checkpoints/.cache/huggingface'):
        super().__init__()
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        self.pipe = pipe.to(device)
        self.vae = self.pipe.vae
        self.guidance_scale = guidance_scale
        self.prompt = prompt
        self.device = device
        self.unet = self.pipe.unet
        self.latent_scale = self.pipe.vae.config.scaling_factor
        self.dtype = torch.float16
        self.resolution = inner_resolution
        self.target_resolution = target_resolution
        # scheduling
        scheduler = pipe.scheduler
        self.scheduler = VPScheduler(
            num_steps=scheduler.config.num_train_timesteps,
            beta_max=scheduler.config.beta_end * scheduler.config.num_train_timesteps,
            beta_min=scheduler.config.beta_start * scheduler.config.num_train_timesteps,
            epsilon=0,
            beta_type=scheduler.config.beta_schedule,
        )
        self.prediction_type = scheduler.config.prediction_type
        self.unet.requires_grad_(False)
    
    def get_noise_prediction(self, latent_model_input, model_output, sigma):
        alpha = (1 / (sigma ** 2 + 1))
        if self.prediction_type == 'epsilon':
            noise_pred = model_output
        elif self.prediction_type == 'v_prediction':
            noise_pred = alpha.sqrt() * model_output + (1 - alpha).sqrt() * latent_model_input
        else:
            raise NotImplementedError
        return noise_pred
    
    def encode(self, x0):
        source_dtype = x0.dtype
        x0 = x0.to(self.dtype)
        x0 = F.interpolate(x0, size=self.resolution, mode='bilinear')
        latents = (self.vae.encode(x0).latent_dist.sample()*self.latent_scale).to(source_dtype)
        return latents
    
    def decode(self, z0):
        source_dtype = z0.dtype
        z0 = z0.to(self.dtype)
        x0 = self.vae.decode(z0/self.latent_scale).sample.to(source_dtype)
        x0 = F.interpolate(x0, size=self.target_resolution, mode='bilinear')
        return x0
    
    def tweedie(self, z, sigma, c=None):
        if c is None:
            c = self.prompt
        # dtype: torch.float32
        source_dtype = z.dtype
        latent = z.to(self.dtype)

        # compute correct sigma
        sigma = sigma.to(self.dtype).view(-1, *([1] * len(latent.shape[1:])))

        # pre conditioning
        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = (self.scheduler.num_steps - 1) * self.scheduler.get_sigma_inv(sigma)

        # get tweedie
        # 1. encode prompt
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt=c, 
            device=z.device, 
            num_images_per_prompt=1, 
            do_classifier_free_guidance=True, 
        )
        prompt_embeds = torch.cat([negative_prompt_embeds] * z.shape[0]+ [prompt_embeds] * z.shape[0], dim=0)

        # 2. get unet output
        latent_model_input = torch.cat([latent] * 2) * c_in
        t_input = c_noise.flatten()
        model_output = self.unet(latent_model_input, t_input, encoder_hidden_states=prompt_embeds, return_dict=False)[0]
        noise_pred = self.get_noise_prediction(latent_model_input, model_output, sigma)

        # 3. classifier free guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)        
        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

        denoised = c_skip * z + c_out * noise_pred.to(source_dtype)
        return denoised

    def get_in_shape(self):
        num_channels_latents = self.pipe.unet.config.in_channels
        latents = self.pipe.prepare_latents(
            1, num_channels_latents, self.resolution, self.resolution, self.dtype, self.device, None, None
        )
        return latents.shape[1:]
    