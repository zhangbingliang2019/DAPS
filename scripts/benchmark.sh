# down_sampling_s42
nohup python posterior_sample.py +data=ffhq +model=ffhq256ddpm +sampler=edm_resample wandb=True data.start_id=980 data.end_id=1000 save_dir=./benchmark +project_name=ncs_benchmark +task=down_sampling num_runs=1 name=down_sampling_s42 +seed=42 gpu=1 &

# gaussian_blur_s42
nohup python posterior_sample.py +data=ffhq +model=ffhq256ddpm +sampler=edm_resample wandb=True data.start_id=980 data.end_id=1000 save_dir=./benchmark +project_name=ncs_benchmark +task=gaussian_blur num_runs=1 name=gaussian_blur_s42 +seed=42 gpu=2 &

# inpainting_s42
nohup python posterior_sample.py +data=ffhq +model=ffhq256ddpm +sampler=edm_resample wandb=True data.start_id=980 data.end_id=1000 save_dir=./benchmark +project_name=ncs_benchmark +task=inpainting num_runs=1 name=inpainting_s42 +seed=42 gpu=3 &

# motion_blur_s42
nohup python posterior_sample.py +data=ffhq +model=ffhq256ddpm +sampler=edm_resample wandb=True data.start_id=980 data.end_id=1000 save_dir=./benchmark +project_name=ncs_benchmark +task=motion_blur num_runs=1 name=motion_blur_s42 +seed=42 gpu=4 &

# phase_retrieval_s42
python posterior_sample.py +data=ffhq +model=ffhq256ddpm +sampler=edm_resample wandb=True data.start_id=980 data.end_id=1000 save_dir=./benchmark +project_name=ncs_benchmark +task=phase_retrieval num_runs=4 name=phase_retrieval_s42 +seed=42 gpu=5

