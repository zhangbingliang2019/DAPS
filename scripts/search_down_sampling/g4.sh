# ode20_step300_tau0.003_alpha0.02
python posterior_sample.py +data=ffhq +model=ffhq256ddpm +sampler=edm_resample +task=down_sampling wandb=True data.start_id=980 data.end_id=1000 save_dir=./search_root/down_sampling +project_name=down_sampling_search num_runs=1 task.likelihood_estimator_config.step=300 task.likelihood_estimator_config.lr=1.8e-07 task.likelihood_estimator_config.ode_step=20 task.likelihood_estimator_config.tau=0.003 name=ode20_step300_tau0.003_alpha0.02 +seed=42 gpu=4

# ode10_step300_tau0.003_alpha0.02
python posterior_sample.py +data=ffhq +model=ffhq256ddpm +sampler=edm_resample +task=down_sampling wandb=True data.start_id=980 data.end_id=1000 save_dir=./search_root/down_sampling +project_name=down_sampling_search num_runs=1 task.likelihood_estimator_config.step=300 task.likelihood_estimator_config.lr=1.8e-07 task.likelihood_estimator_config.ode_step=10 task.likelihood_estimator_config.tau=0.003 name=ode10_step300_tau0.003_alpha0.02 +seed=42 gpu=4

# ode3_step300_tau0.003_alpha0.02
python posterior_sample.py +data=ffhq +model=ffhq256ddpm +sampler=edm_resample +task=down_sampling wandb=True data.start_id=980 data.end_id=1000 save_dir=./search_root/down_sampling +project_name=down_sampling_search num_runs=1 task.likelihood_estimator_config.step=300 task.likelihood_estimator_config.lr=1.8e-07 task.likelihood_estimator_config.ode_step=3 task.likelihood_estimator_config.tau=0.003 name=ode3_step300_tau0.003_alpha0.02 +seed=42 gpu=4

# ode1_step300_tau0.003_alpha0.02
python posterior_sample.py +data=ffhq +model=ffhq256ddpm +sampler=edm_resample +task=down_sampling wandb=True data.start_id=980 data.end_id=1000 save_dir=./search_root/down_sampling +project_name=down_sampling_search num_runs=1 task.likelihood_estimator_config.step=300 task.likelihood_estimator_config.lr=1.8e-07 task.likelihood_estimator_config.ode_step=1 task.likelihood_estimator_config.tau=0.003 name=ode1_step300_tau0.003_alpha0.02 +seed=42 gpu=4

# ode20_step100_tau0.003_alpha0.02
python posterior_sample.py +data=ffhq +model=ffhq256ddpm +sampler=edm_resample +task=down_sampling wandb=True data.start_id=980 data.end_id=1000 save_dir=./search_root/down_sampling +project_name=down_sampling_search num_runs=1 task.likelihood_estimator_config.step=100 task.likelihood_estimator_config.lr=1.8e-07 task.likelihood_estimator_config.ode_step=20 task.likelihood_estimator_config.tau=0.003 name=ode20_step100_tau0.003_alpha0.02 +seed=42 gpu=4

# ode10_step100_tau0.003_alpha0.02
python posterior_sample.py +data=ffhq +model=ffhq256ddpm +sampler=edm_resample +task=down_sampling wandb=True data.start_id=980 data.end_id=1000 save_dir=./search_root/down_sampling +project_name=down_sampling_search num_runs=1 task.likelihood_estimator_config.step=100 task.likelihood_estimator_config.lr=1.8e-07 task.likelihood_estimator_config.ode_step=10 task.likelihood_estimator_config.tau=0.003 name=ode10_step100_tau0.003_alpha0.02 +seed=42 gpu=4

# ode3_step100_tau0.003_alpha0.02
python posterior_sample.py +data=ffhq +model=ffhq256ddpm +sampler=edm_resample +task=down_sampling wandb=True data.start_id=980 data.end_id=1000 save_dir=./search_root/down_sampling +project_name=down_sampling_search num_runs=1 task.likelihood_estimator_config.step=100 task.likelihood_estimator_config.lr=1.8e-07 task.likelihood_estimator_config.ode_step=3 task.likelihood_estimator_config.tau=0.003 name=ode3_step100_tau0.003_alpha0.02 +seed=42 gpu=4

# ode1_step100_tau0.003_alpha0.02
python posterior_sample.py +data=ffhq +model=ffhq256ddpm +sampler=edm_resample +task=down_sampling wandb=True data.start_id=980 data.end_id=1000 save_dir=./search_root/down_sampling +project_name=down_sampling_search num_runs=1 task.likelihood_estimator_config.step=100 task.likelihood_estimator_config.lr=1.8e-07 task.likelihood_estimator_config.ode_step=1 task.likelihood_estimator_config.tau=0.003 name=ode1_step100_tau0.003_alpha0.02 +seed=42 gpu=4

# ode20_step10_tau0.003_alpha0.02
python posterior_sample.py +data=ffhq +model=ffhq256ddpm +sampler=edm_resample +task=down_sampling wandb=True data.start_id=980 data.end_id=1000 save_dir=./search_root/down_sampling +project_name=down_sampling_search num_runs=1 task.likelihood_estimator_config.step=10 task.likelihood_estimator_config.lr=1.8e-07 task.likelihood_estimator_config.ode_step=20 task.likelihood_estimator_config.tau=0.003 name=ode20_step10_tau0.003_alpha0.02 +seed=42 gpu=4

# ode10_step10_tau0.003_alpha0.02
python posterior_sample.py +data=ffhq +model=ffhq256ddpm +sampler=edm_resample +task=down_sampling wandb=True data.start_id=980 data.end_id=1000 save_dir=./search_root/down_sampling +project_name=down_sampling_search num_runs=1 task.likelihood_estimator_config.step=10 task.likelihood_estimator_config.lr=1.8e-07 task.likelihood_estimator_config.ode_step=10 task.likelihood_estimator_config.tau=0.003 name=ode10_step10_tau0.003_alpha0.02 +seed=42 gpu=4

# ode3_step10_tau0.003_alpha0.02
python posterior_sample.py +data=ffhq +model=ffhq256ddpm +sampler=edm_resample +task=down_sampling wandb=True data.start_id=980 data.end_id=1000 save_dir=./search_root/down_sampling +project_name=down_sampling_search num_runs=1 task.likelihood_estimator_config.step=10 task.likelihood_estimator_config.lr=1.8e-07 task.likelihood_estimator_config.ode_step=3 task.likelihood_estimator_config.tau=0.003 name=ode3_step10_tau0.003_alpha0.02 +seed=42 gpu=4

# ode1_step10_tau0.003_alpha0.02
python posterior_sample.py +data=ffhq +model=ffhq256ddpm +sampler=edm_resample +task=down_sampling wandb=True data.start_id=980 data.end_id=1000 save_dir=./search_root/down_sampling +project_name=down_sampling_search num_runs=1 task.likelihood_estimator_config.step=10 task.likelihood_estimator_config.lr=1.8e-07 task.likelihood_estimator_config.ode_step=1 task.likelihood_estimator_config.tau=0.003 name=ode1_step10_tau0.003_alpha0.02 +seed=42 gpu=4

