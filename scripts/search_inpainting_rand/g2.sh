# ode20_step300_tau0.005_alpha0.02
python posterior_sample.py +data=ffhq +model=ffhq256ddpm +sampler=edm_resample +task=inpainting_rand wandb=True data.start_id=980 data.end_id=1000 save_dir=./search_root/inpainting_rand +project_name=inpainting_rand_search num_runs=1 task.likelihood_estimator_config.step=300 task.likelihood_estimator_config.lr=5.000000000000001e-07 task.likelihood_estimator_config.ode_step=20 task.likelihood_estimator_config.tau=0.005 name=ode20_step300_tau0.005_alpha0.02 +seed=42 gpu=2

# ode10_step300_tau0.005_alpha0.02
python posterior_sample.py +data=ffhq +model=ffhq256ddpm +sampler=edm_resample +task=inpainting_rand wandb=True data.start_id=980 data.end_id=1000 save_dir=./search_root/inpainting_rand +project_name=inpainting_rand_search num_runs=1 task.likelihood_estimator_config.step=300 task.likelihood_estimator_config.lr=5.000000000000001e-07 task.likelihood_estimator_config.ode_step=10 task.likelihood_estimator_config.tau=0.005 name=ode10_step300_tau0.005_alpha0.02 +seed=42 gpu=2

# ode3_step300_tau0.005_alpha0.02
python posterior_sample.py +data=ffhq +model=ffhq256ddpm +sampler=edm_resample +task=inpainting_rand wandb=True data.start_id=980 data.end_id=1000 save_dir=./search_root/inpainting_rand +project_name=inpainting_rand_search num_runs=1 task.likelihood_estimator_config.step=300 task.likelihood_estimator_config.lr=5.000000000000001e-07 task.likelihood_estimator_config.ode_step=3 task.likelihood_estimator_config.tau=0.005 name=ode3_step300_tau0.005_alpha0.02 +seed=42 gpu=2

# ode1_step300_tau0.005_alpha0.02
python posterior_sample.py +data=ffhq +model=ffhq256ddpm +sampler=edm_resample +task=inpainting_rand wandb=True data.start_id=980 data.end_id=1000 save_dir=./search_root/inpainting_rand +project_name=inpainting_rand_search num_runs=1 task.likelihood_estimator_config.step=300 task.likelihood_estimator_config.lr=5.000000000000001e-07 task.likelihood_estimator_config.ode_step=1 task.likelihood_estimator_config.tau=0.005 name=ode1_step300_tau0.005_alpha0.02 +seed=42 gpu=2

# ode20_step100_tau0.005_alpha0.02
python posterior_sample.py +data=ffhq +model=ffhq256ddpm +sampler=edm_resample +task=inpainting_rand wandb=True data.start_id=980 data.end_id=1000 save_dir=./search_root/inpainting_rand +project_name=inpainting_rand_search num_runs=1 task.likelihood_estimator_config.step=100 task.likelihood_estimator_config.lr=5.000000000000001e-07 task.likelihood_estimator_config.ode_step=20 task.likelihood_estimator_config.tau=0.005 name=ode20_step100_tau0.005_alpha0.02 +seed=42 gpu=2

# ode10_step100_tau0.005_alpha0.02
python posterior_sample.py +data=ffhq +model=ffhq256ddpm +sampler=edm_resample +task=inpainting_rand wandb=True data.start_id=980 data.end_id=1000 save_dir=./search_root/inpainting_rand +project_name=inpainting_rand_search num_runs=1 task.likelihood_estimator_config.step=100 task.likelihood_estimator_config.lr=5.000000000000001e-07 task.likelihood_estimator_config.ode_step=10 task.likelihood_estimator_config.tau=0.005 name=ode10_step100_tau0.005_alpha0.02 +seed=42 gpu=2

# ode3_step100_tau0.005_alpha0.02
python posterior_sample.py +data=ffhq +model=ffhq256ddpm +sampler=edm_resample +task=inpainting_rand wandb=True data.start_id=980 data.end_id=1000 save_dir=./search_root/inpainting_rand +project_name=inpainting_rand_search num_runs=1 task.likelihood_estimator_config.step=100 task.likelihood_estimator_config.lr=5.000000000000001e-07 task.likelihood_estimator_config.ode_step=3 task.likelihood_estimator_config.tau=0.005 name=ode3_step100_tau0.005_alpha0.02 +seed=42 gpu=2

# ode1_step100_tau0.005_alpha0.02
python posterior_sample.py +data=ffhq +model=ffhq256ddpm +sampler=edm_resample +task=inpainting_rand wandb=True data.start_id=980 data.end_id=1000 save_dir=./search_root/inpainting_rand +project_name=inpainting_rand_search num_runs=1 task.likelihood_estimator_config.step=100 task.likelihood_estimator_config.lr=5.000000000000001e-07 task.likelihood_estimator_config.ode_step=1 task.likelihood_estimator_config.tau=0.005 name=ode1_step100_tau0.005_alpha0.02 +seed=42 gpu=2

# ode20_step10_tau0.005_alpha0.02
python posterior_sample.py +data=ffhq +model=ffhq256ddpm +sampler=edm_resample +task=inpainting_rand wandb=True data.start_id=980 data.end_id=1000 save_dir=./search_root/inpainting_rand +project_name=inpainting_rand_search num_runs=1 task.likelihood_estimator_config.step=10 task.likelihood_estimator_config.lr=5.000000000000001e-07 task.likelihood_estimator_config.ode_step=20 task.likelihood_estimator_config.tau=0.005 name=ode20_step10_tau0.005_alpha0.02 +seed=42 gpu=2

# ode10_step10_tau0.005_alpha0.02
python posterior_sample.py +data=ffhq +model=ffhq256ddpm +sampler=edm_resample +task=inpainting_rand wandb=True data.start_id=980 data.end_id=1000 save_dir=./search_root/inpainting_rand +project_name=inpainting_rand_search num_runs=1 task.likelihood_estimator_config.step=10 task.likelihood_estimator_config.lr=5.000000000000001e-07 task.likelihood_estimator_config.ode_step=10 task.likelihood_estimator_config.tau=0.005 name=ode10_step10_tau0.005_alpha0.02 +seed=42 gpu=2

# ode3_step10_tau0.005_alpha0.02
python posterior_sample.py +data=ffhq +model=ffhq256ddpm +sampler=edm_resample +task=inpainting_rand wandb=True data.start_id=980 data.end_id=1000 save_dir=./search_root/inpainting_rand +project_name=inpainting_rand_search num_runs=1 task.likelihood_estimator_config.step=10 task.likelihood_estimator_config.lr=5.000000000000001e-07 task.likelihood_estimator_config.ode_step=3 task.likelihood_estimator_config.tau=0.005 name=ode3_step10_tau0.005_alpha0.02 +seed=42 gpu=2

# ode1_step10_tau0.005_alpha0.02
python posterior_sample.py +data=ffhq +model=ffhq256ddpm +sampler=edm_resample +task=inpainting_rand wandb=True data.start_id=980 data.end_id=1000 save_dir=./search_root/inpainting_rand +project_name=inpainting_rand_search num_runs=1 task.likelihood_estimator_config.step=10 task.likelihood_estimator_config.lr=5.000000000000001e-07 task.likelihood_estimator_config.ode_step=1 task.likelihood_estimator_config.tau=0.005 name=ode1_step10_tau0.005_alpha0.02 +seed=42 gpu=2

