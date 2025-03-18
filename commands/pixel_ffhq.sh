# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This file contains the commands to run the pixel diffusion experiment on FFHQ dataset with DAPS 1K.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ++++ Nonlinear Tasks ++++
# phase retrieval
python posterior_sample.py \
+data=test-ffhq \
+model=ffhq256ddpm \
+task=phase_retrieval \
+sampler=edm_daps \
task_group=pixel \
save_dir=results/pixel/ffhq \
num_runs=4 \
sampler.diffusion_scheduler_config.num_steps=5 \
sampler.annealing_scheduler_config.num_steps=200 \
batch_size=100 \
name=phase_retrieval \
gpu=0 &

# nonlinear deblur
python posterior_sample.py \
+data=test-ffhq \
+model=ffhq256ddpm \
+task=nonlinear_blur \
+sampler=edm_daps \
task_group=pixel \
save_dir=results/pixel/ffhq \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=5 \
sampler.annealing_scheduler_config.num_steps=200 \
batch_size=100 \
name=nonlinear_blur \
gpu=1 & 

# high dynamic range
python posterior_sample.py \
+data=test-ffhq \
+model=ffhq256ddpm \
+task=hdr \
+sampler=edm_daps \
task_group=pixel \
save_dir=results/pixel/ffhq \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=5 \
sampler.annealing_scheduler_config.num_steps=200 \
batch_size=100 \
name=hdr \
gpu=2 & 

# ++++ Linear Tasks ++++
# down sampling
python posterior_sample.py \
+data=test-ffhq \
+model=ffhq256ddpm \
+task=down_sampling \
+sampler=edm_daps \
task_group=pixel \
save_dir=results/pixel/ffhq \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=5 \
sampler.annealing_scheduler_config.num_steps=200 \
batch_size=100 \
name=down_sampling \
gpu=3 & 

# Gaussian blur
python posterior_sample.py \
+data=test-ffhq \
+model=ffhq256ddpm \
+task=gaussian_blur \
+sampler=edm_daps \
task_group=pixel \
save_dir=results/pixel/ffhq \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=5 \
sampler.annealing_scheduler_config.num_steps=200 \
batch_size=100 \
name=gaussian_blur \
gpu=4 & 

# motion blur
python posterior_sample.py \
+data=test-ffhq \
+model=ffhq256ddpm \
+task=motion_blur \
+sampler=edm_daps \
task_group=pixel \
save_dir=results/pixel/ffhq \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=5 \
sampler.annealing_scheduler_config.num_steps=200 \
batch_size=100 \
name=motion_blur \
gpu=5 & 

# box inpainting 
python posterior_sample.py \
+data=test-ffhq \
+model=ffhq256ddpm \
+task=inpainting \
+sampler=edm_daps \
task_group=pixel \
save_dir=results/pixel/ffhq \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=5 \
sampler.annealing_scheduler_config.num_steps=200 \
batch_size=100 \
name=inpainting \
gpu=6 & 

# random inpainting
python posterior_sample.py \
+data=test-ffhq \
+model=ffhq256ddpm \
+task=inpainting_rand \
+sampler=edm_daps \
task_group=pixel \
save_dir=results/pixel/ffhq \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=5 \
sampler.annealing_scheduler_config.num_steps=200 \
batch_size=100 \
name=inpainting_rand \
gpu=7 & 