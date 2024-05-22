from pathlib import Path

CONFIGURATION = {
    'daps50': [2, 25],
    'daps100': [2, 50],
    'daps200': [4, 50],
    'daps400': [4, 100],
    'daps1k': [5, 200],
    # 'daps2k': [8, 250],
    # 'daps4k': [10, 400],
    # 'daps8k': [16, 500]
}

LATENT_CONFIGURATION = {
    'latent_daps_tiny': [2, 25],
    'latent_daps_small': [2, 50],
    'latent_daps_normal': [5, 100],
    'latent_daps_large': [5, 200]
}




LatentTASK_LIST = [
    ('down_sampling', 1, 5, 80, 20, 0.5, 0.5),
    ('inpainting', 1, 5, 80, 20, 0.5, 0.1),
    ('inpainting_rand', 1, 5, 80, 20, 0.5, 0.5),
    ('gaussian_blur', 1, 5, 50, 50, 0.5, 1),
    ('motion_blur', 1, 5, 50, 50, 0.5, 1),
    ('hdr', 1, 10, 180, 20, 0.5, 3),
    ('nonlinear_blur', 1, 5, 50, 50, 0.5, 1),
    ('phase_retrieval', 4)
]


DATA_MODEL_PAIR = [
    # ('ffhq', 'ffhq256ddpm'),
    ('imagenet', 'imagenet256ddpm'),
    # ('lsun_bedroom', 'lsun_bedroom256ddpm')
]

DATA_LatentMODEL_PAIR = [
    ('ffhq', 'ldm_ffhq'),
    # ('imagenet', 'imagenet256ddpm'),
    # ('lsun_bedroom', 'lsun_bedroom256ddpm')
]


def safe_dir(dir):
    if not dir.exists():
        dir.mkdir()
    return dir


def write_command(name, command, nohup=False, sh_file='scripts/tast.sh'):
    if nohup:
        command = 'nohup ' + command + ' &'
    with open(sh_file, 'a') as f:
        f.write(f'# {name}')
        f.write('\n')
        f.write(command)
        f.write('\n')
        f.write('\n')


def generate_final_pre(seed=42, gpu_list=[0], scripts_dir='final'):
    safe_dir(Path('scripts') / scripts_dir)
    sh_file = 'scripts/{}/g{}.sh'
    for gpu in gpu_list:
        open(sh_file.format(scripts_dir, gpu), 'w')
    idx = 0
    root = Path('final')
    template = 'python posterior_sample.py +data={} +model={} +task={} ' + \
               'save_dir={} +project_name=final_imagenet num_runs={} ' + \
               'task.likelihood_estimator_config.ode_step={} sampler.num_steps={} ' + \
               'batch_size=100 +sampler=edm_resample wandb=True data.start_id=49000 data.end_id=49100 ' + \
               'name={} +seed={} gpu={}'
    for (dataset_name, model_name) in DATA_MODEL_PAIR:
        dataset_dir = safe_dir(root / dataset_name)
        for (task, num_run) in TASK_LIST:
            task_dir = safe_dir(dataset_dir / task)
            for config_name, (ode_step, step) in CONFIGURATION.items():
                save_dir = safe_dir(task_dir / config_name)
                name = dataset_name + '_' + task + '_' + config_name
                command = template.format(
                    dataset_name, model_name, task, str(save_dir), num_run,
                    ode_step, step, name, seed, gpu_list[idx])
                write_command(name, command, False, sh_file.format(scripts_dir, gpu_list[idx]))
                idx = (idx + 1) % (len(gpu_list))

    open('scripts/final.sh', 'w')
    for gpu in gpu_list:
        command = 'sh {}'.format(sh_file.format(scripts_dir, gpu))
        write_command('gpu{}'.format(gpu), command, True, 'scripts/final.sh')


def generate_final(seed=42, gpu_list=[0], scripts_dir='final_benchmark_ffhq_4k'):
    safe_dir(Path('scripts') / scripts_dir)
    sh_file = 'scripts/{}/g{}.sh'
    for gpu in gpu_list:
        open(sh_file.format(scripts_dir, gpu), 'w')
    idx = 0
    root = Path('final_benchmark')
    template = 'python posterior_sample.py +data={} +model={} +task={} ' + \
               'save_dir={} +project_name=final_benchmark num_runs={} ' + \
               'task.likelihood_estimator_config.ode_step={} sampler.num_steps={} ' + \
               'batch_size=100 +sampler=edm_resample wandb=True data.start_id=49000 data.end_id=49100 ' + \
               'name={} +seed={} gpu={}'
    for (dataset_name, model_name) in DATA_MODEL_PAIR:
        dataset_dir = safe_dir(root / dataset_name)
        for (task, num_run) in TASK_LIST:
            task_dir = safe_dir(dataset_dir / task)
            for config_name, (ode_step, step) in CONFIGURATION.items():
                save_dir = safe_dir(task_dir / config_name)
                name = dataset_name + '_' + task + '_' + config_name
                command = template.format(
                    dataset_name, model_name, task, str(save_dir), num_run,
                    ode_step, step, name, seed, gpu_list[idx])
                write_command(name, command, False, sh_file.format(scripts_dir, gpu_list[idx]))
                idx = (idx + 1) % (len(gpu_list))

    open('scripts/final.sh', 'w')
    for gpu in gpu_list:
        command = 'sh {}'.format(sh_file.format(scripts_dir, gpu))
        write_command('gpu{}'.format(gpu), command, True, 'scripts/final.sh')


def generate_final_latent_pre(seed=42, gpu_list=[0], scripts_dir='final_benchmark_latent_pre'):
    safe_dir(Path('scripts') / scripts_dir)
    sh_file = 'scripts/{}/g{}.sh'
    for gpu in gpu_list:
        open(sh_file.format(scripts_dir, gpu), 'w')
    idx = 0
    root = Path('final_benchmark')
    template = 'python posterior_sample.py +data={} +model={} +task={} ' + \
               'save_dir={} +project_name=final_benchmark num_runs={} ' + \
               'task.likelihood_estimator_config.config1.ode_step={} task.likelihood_estimator_config.config2.ode_step={} sampler.num_steps={} ' + \
               'batch_size=20 +sampler=latent_edm_resample wandb=True data.start_id=49000 data.end_id=49100 ' + \
               'name={} +seed={} gpu={}'
    for (dataset_name, model_name) in DATA_LatentMODEL_PAIR:
        dataset_dir = safe_dir(root / dataset_name)
        for (task, num_run) in TASK_LIST:
            task_dir = safe_dir(dataset_dir / task)
            for config_name, (ode_step, step) in LATENT_CONFIGURATION.items():
                save_dir = safe_dir(task_dir / config_name)
                name = dataset_name + '_' + task + '_' + config_name
                command = template.format(
                    dataset_name, model_name, task + '_latent', str(save_dir), num_run,
                    ode_step, ode_step, 2*step, name, seed, gpu_list[idx])
                write_command(name, command, False, sh_file.format(scripts_dir, gpu_list[idx]))
                idx = (idx + 1) % (len(gpu_list))

    open('scripts/final.sh', 'w')
    for gpu in gpu_list:
        command = 'sh {}'.format(sh_file.format(scripts_dir, gpu))
        write_command('gpu{}'.format(gpu), command, True, 'scripts/final.sh')


def generate_final_latent(seed=42, gpu_list=[0], scripts_dir='final_benchmark_ffhq_latent_1k'):
    safe_dir(Path('scripts') / scripts_dir)
    sh_file = 'scripts/{}/g{}.sh'
    for gpu in gpu_list:
        open(sh_file.format(scripts_dir, gpu), 'w')
    idx = 0
    root = Path('final_benchmark')
    template = 'python posterior_sample.py +data={} +model={} +task={} ' + \
               'save_dir={} +project_name=final_benchmark num_runs={} ' + \
               'task.likelihood_estimator_config.config1.ode_step={} task.likelihood_estimator_config.config2.ode_step={} sampler.num_steps={} ' + \
               'batch_size=100 +sampler=latent_edm_resample wandb=True data.start_id=49000 data.end_id=49100 ' + \
               'name={} +seed={} gpu={}'
    for (dataset_name, model_name) in DATA_LatentMODEL_PAIR:
        dataset_dir = safe_dir(root / dataset_name)
        for (task, num_run) in TASK_LIST:
            task_dir = safe_dir(dataset_dir / task)
            for config_name, (ode_step, step) in LATENT_CONFIGURATION.items():
                save_dir = safe_dir(task_dir / config_name)
                name = dataset_name + '_' + task + '_' + config_name
                command = template.format(
                    dataset_name, model_name, task + '_latent', str(save_dir), num_run,
                    ode_step, ode_step, 2*step, name, seed, gpu_list[idx])
                write_command(name, command, False, sh_file.format(scripts_dir, gpu_list[idx]))
                idx = (idx + 1) % (len(gpu_list))

    open('scripts/final.sh', 'w')
    for gpu in gpu_list:
        command = 'sh {}'.format(sh_file.format(scripts_dir, gpu))
        write_command('gpu{}'.format(gpu), command, True, 'scripts/final.sh')


generate_final_latent_pre(gpu_list=[1, 4, 5, 7])
