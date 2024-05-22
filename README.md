# Decoupled Annealing Posterior Sampling

## Getting started

#### 1. Prepare the Conda environment

```
# in daps folder

conda create -n daps python=3.8
conda activate daps

pip install -r requirements.txt
pip install torch torchvision torchaudio

git clone https://github.com/VinAIResearch/blur-kernel-space-exploring bkse
git clone https://github.com/LeviBorodenko/motionblur motionblur

mv bkse forward_operator/
mv motionblur forward_operator/
```



#### 2. Prepare the pretrainedcheckpoint

Download the public available FFHQ checkpoint [here](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh).

```
# in daps folder

mkdir checkpoint
mv {DOWNLOAD_DIR}/ffqh_10m.pt checkpoint/ffhq256.pt
```



#### 3. Prepare the dataset (or use provided examples) (optional)

You can add any FFHQ256 images you like to `dataset/demo` folder



## Inference

Make a folder to save results:

```
mkdir results
```

Now you are ready for run. For **phase retrieval** with DAPS-1k in 4 runs for $10$ demo images in `dataset/demo`:

```
python posterior_sample.py +data=demo +model=ffhq256ddpm +task=phase_retrieval save_dir=results num_runs=4 task.likelihood_estimator_config.ode_step=5 sampler.num_steps=200 batch_size=10 +sampler=edm_daps data.start_id=0 data.end_id=10 name=phase_retrieval_demo +seed=42 gpu=0
```

It taks about $8$ minutes ($2$ for each run) on a single NVIDIA A100-SXM4-80GB GPU. The results are saved at foloder `\results/phase_retrieval_demo`. You might find figure like below in `grid_results.png`:

![image-20240522081725814](README.assets/demo.png)

For other task, 

```
python posterior_sample.py +data=demo +model=ffhq256ddpm +task={TASK_NAME} save_dir=results num_runs=1 task.likelihood_estimator_config.ode_step=5 sampler.num_steps=200 batch_size=10 +sampler=edm_daps data.start_id=0 data.end_id=10 name={TASK_NAME}_demo +seed=42 gpu=0
```

replace the {TASK_NAME} by one of following:

```
down_sampling, inpainting, inpainting_rand, gaussian_blur, motion_blur, nonlinear_blur, hdr, phase_retrieval
```







