from taming.models import vqgan
from ldm.models.diffusion.ddim import DDIMSampler
#@title loading utils
import torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid
from einops import rearrange, repeat
from utils import *
import PIL
import argparse

def load_model_from_config(config, ckpt, train=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    
    model.cuda()
    # model.train()
    if train:
      model.train()
    else:
      model.eval()
    return model


def get_model(model_type = None):
    if model_type is None:
      config = OmegaConf.load("configs/latent-diffusion/celebahq-ldm-vq-4.yaml")
      model = load_model_from_config(config, "models/ldm/celeba256/model.ckpt") 
    elif model_type == "celeb":
      config = OmegaConf.load("configs/latent-diffusion/celebahq-ldm-vq-4.yaml") 
      model = load_model_from_config(config, "models/ldm/celeba256/model.ckpt")
    else:
      model = None
    return model



def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.



device = torch.device("cuda")
model = get_model()
model.learning_rate = 5e-3
ddim_steps = 100
num_timesteps = 1000
shape=[1, 3, 64, 64]
z = torch.randn(shape, device=device)
z.requires_grad = True
init_image = load_img("/content/drive/MyDrive/stable-diffusion/ldct/test_recon.png").to(device)
init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))
opt = torch.optim.AdamW([z], lr = model.learning_rate)
loss = torch.nn.MSELoss()
angles = torch.tensor(np.linspace(0, np.pi, 25, endpoint=False))
sampler = DDIMSampler(model)
sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=0, verbose = False)
projection_orig = ct_parallel_project_2d_batch(init_image.permute(0,2,3,1), angles)



for i in range(2000):
  opt.zero_grad()
  decoded_z = sampler.ddecode(z, t_start = 100, temp = 0)
  decoded_img = model.differentiable_decode_first_stage(decoded_z)
  # decoded_img = model.differentiable_decode_first_stage(z)
  projection_recon = ct_parallel_project_2d_batch(decoded_img.permute(0,2,3,1), angles)
  output = loss(projection_orig, projection_recon)
  # output = loss(init_image, decoded_img)
  output.backward()
  opt.step()
  loss_val = output.detach().cpu().numpy()
  if i % 5 == 0:
    print(loss_val, "loss ", str(i), " iter")
  if loss_val < 6.5e-5:
    break

# sampler.make_schedule(ddim_num_steps=50, ddim_eta=0, verbose = False)


print("start encoding decoding")
# decoded_z = sampler.decode(z, cond = None, t_start = 499)
t = repeat(torch.tensor([250]), '1 -> b', b=1)
t = t.to(device).long()
# np.save("CSGM_nopnp_500iter.npy", model.decode_first_stage(z).detach().cpu().numpy())
np.save("CSGM_recon_2000iter_100_tstep_norandomness.npy", model.decode_first_stage(sampler.ddecode(z,cond=None,t_start=100, temp = 0)).detach().cpu().numpy())
# print(z)
# encoded = sampler.stochastic_encode(z, t)
# decoded = sampler.decode(encoded, None, 250)
# np.save("PnP_CSGM_recon500iter.npy", model.decode_first_stage(decoded).detach().cpu().numpy())

