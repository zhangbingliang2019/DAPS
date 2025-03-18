# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This file contains the commands to run the LDM experiment on ImageNet dataset with DAPS 100.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ++++ Nonlinear Tasks ++++
# phase retrieval
python posterior_sample.py \
+data=test-imagenet \
+model=imagenet256ldm \
+task=phase_retrieval \
+sampler=latent_edm_daps \
task_group=ldm \
save_dir=results/ldm/imagenet \
num_runs=4 \
sampler.diffusion_scheduler_config.num_steps=2 \
sampler.annealing_scheduler_config.num_steps=50 \
batch_size=10 \
name=phase_retrieval \
gpu=0 &

# nonlinear deblur
python posterior_sample.py \
+data=test-imagenet \
+model=imagenet256ldm \
+task=nonlinear_blur \
+sampler=latent_edm_daps \
task_group=ldm \
save_dir=results/ldm/imagenet \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=2 \
sampler.annealing_scheduler_config.num_steps=50 \
batch_size=10 \
name=nonlinear_blur \
gpu=1 & 

# high dynamic range
python posterior_sample.py \
+data=test-imagenet \
+model=imagenet256ldm \
+task=hdr \
+sampler=latent_edm_daps \
task_group=ldm \
save_dir=results/ldm/imagenet \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=2 \
sampler.annealing_scheduler_config.num_steps=50 \
batch_size=10 \
name=hdr \
gpu=2 & 

# ++++ Linear Tasks ++++
# down sampling
python posterior_sample.py \
+data=test-imagenet \
+model=imagenet256ldm \
+task=down_sampling \
+sampler=latent_edm_daps \
task_group=ldm \
save_dir=results/ldm/imagenet \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=2 \
sampler.annealing_scheduler_config.num_steps=50 \
batch_size=10 \
name=down_sampling \
gpu=3 & 

# Gaussian blur
python posterior_sample.py \
+data=test-imagenet \
+model=imagenet256ldm \
+task=gaussian_blur \
+sampler=latent_edm_daps \
task_group=ldm \
save_dir=results/ldm/imagenet \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=2 \
sampler.annealing_scheduler_config.num_steps=50 \
batch_size=10 \
name=gaussian_blur \
gpu=4 & 

# motion blur
python posterior_sample.py \
+data=test-imagenet \
+model=imagenet256ldm \
+task=motion_blur \
+sampler=latent_edm_daps \
task_group=ldm \
save_dir=results/ldm/imagenet \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=2 \
sampler.annealing_scheduler_config.num_steps=50 \
batch_size=10 \
name=motion_blur \
gpu=5 & 

# box inpainting 
python posterior_sample.py \
+data=test-imagenet \
+model=imagenet256ldm \
+task=inpainting \
+sampler=latent_edm_daps \
task_group=ldm \
save_dir=results/ldm/imagenet \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=2 \
sampler.annealing_scheduler_config.num_steps=50 \
batch_size=10 \
name=inpainting \
gpu=6 & 

# random inpainting
python posterior_sample.py \
+data=test-imagenet \
+model=imagenet256ldm \
+task=inpainting_rand \
+sampler=latent_edm_daps \
task_group=ldm \
save_dir=results/ldm/imagenet \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=2 \
sampler.annealing_scheduler_config.num_steps=50 \
batch_size=10 \
name=inpainting_rand \
gpu=7 & 