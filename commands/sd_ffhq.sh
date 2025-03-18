# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This file contains the commands to run the stble diffusion (SD v1.5) experiment on FFHQ dataset with DAPS 100.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ++++ Nonlinear Tasks ++++
# phase retrieval
python posterior_sample.py \
+data=test-ffhq \
+model=stable-diffusion-v1.5 \
+task=phase_retrieval \
+sampler=sd_edm_daps \
task_group=sd \
save_dir=results/sd/ffhq \
num_runs=4 \
sampler.diffusion_scheduler_config.num_steps=2 \
sampler.annealing_scheduler_config.num_steps=50 \
batch_size=10 \
name=phase_retrieval \
gpu=0 &

# nonlinear deblur
python posterior_sample.py \
+data=test-ffhq \
+model=stable-diffusion-v1.5 \
+task=nonlinear_blur \
+sampler=sd_edm_daps \
task_group=sd \
save_dir=results/sd/ffhq \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=2 \
sampler.annealing_scheduler_config.num_steps=50 \
batch_size=10 \
name=nonlinear_blur \
gpu=1 & 

# high dynamic range
python posterior_sample.py \
+data=test-ffhq \
+model=stable-diffusion-v1.5 \
+task=hdr \
+sampler=sd_edm_daps \
task_group=sd \
save_dir=results/sd/ffhq \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=2 \
sampler.annealing_scheduler_config.num_steps=50 \
batch_size=10 \
name=hdr \
gpu=2 & 

# ++++ Linear Tasks ++++
# down sampling
python posterior_sample.py \
+data=test-ffhq \
+model=stable-diffusion-v1.5 \
+task=down_sampling \
+sampler=sd_edm_daps \
task_group=sd \
save_dir=results/sd/ffhq \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=2 \
sampler.annealing_scheduler_config.num_steps=50 \
batch_size=10 \
name=down_sampling \
gpu=3 & 

# Gaussian blur
python posterior_sample.py \
+data=test-ffhq \
+model=stable-diffusion-v1.5 \
+task=gaussian_blur \
+sampler=sd_edm_daps \
task_group=sd \
save_dir=results/sd/ffhq \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=2 \
sampler.annealing_scheduler_config.num_steps=50 \
batch_size=10 \
name=gaussian_blur \
gpu=4 & 

# motion blur
python posterior_sample.py \
+data=test-ffhq \
+model=stable-diffusion-v1.5 \
+task=motion_blur \
+sampler=sd_edm_daps \
task_group=sd \
save_dir=results/sd/ffhq \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=2 \
sampler.annealing_scheduler_config.num_steps=50 \
batch_size=10 \
name=motion_blur \
gpu=5 & 

# box inpainting 
python posterior_sample.py \
+data=test-ffhq \
+model=stable-diffusion-v1.5 \
+task=inpainting \
+sampler=sd_edm_daps \
task_group=sd \
save_dir=results/sd/ffhq \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=2 \
sampler.annealing_scheduler_config.num_steps=50 \
batch_size=10 \
name=inpainting \
gpu=6 & 

# random inpainting
python posterior_sample.py \
+data=test-ffhq \
+model=stable-diffusion-v1.5 \
+task=inpainting_rand \
+sampler=sd_edm_daps \
task_group=sd \
save_dir=results/sd/ffhq \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=2 \
sampler.annealing_scheduler_config.num_steps=50 \
batch_size=10 \
name=inpainting_rand \
gpu=7 & 