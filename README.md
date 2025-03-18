# DAPS: Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing (CVPR 2025)

#### [website](https://daps-inverse-problem.github.io/) |  [paper](https://arxiv.org/abs/2407.01521)

![img](README.assets/teaser.png)



## Abstract

We propose a new method called Decoupled Annealing Posterior Sampling (DAPS) that relies on a novel noise annealing process to solve posterior sampling with diffusion prior. Specifically, we decouple consecutive steps in a diffusion sampling trajectory, allowing them to vary considerably from one another while ensuring their time-marginals anneal to the true posterior as we reduce noise levels. 

![img](README.assets/method.png)

This approach enables the exploration of a larger solution space, improving the success rate for accurate reconstructions. We demonstrate that DAPS significantly improves sample quality and stability across multiple image restoration tasks, particularly in complicated nonlinear inverse problems.



## News:

* **2025-03:** update code structure and usability to support different MCMC algorithms and various types of diffusion models. The previous code structure has been moved to the `legacy` branch. Major updates are summarized as below:

  1. update diffusion schulers in `cores/scheduler.py`.

  2. update MCMC sampler to support different algorithms and approximations in `cores/mcmc.py`.

  3. enhance LatentDAPS with $\texttt{HMC}$ which sustantially improve the performance.

     

## 🕹️ Try and Play with DAPS on Colab!

| Link                                                         | Description                                                |
| ------------------------------------------------------------ | ---------------------------------------------------------- |
| [<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>](https://colab.research.google.com/github/zhangbingliang2019/DAPS/blob/main/scripts/DAPS_Demo.ipynb) | Try DAPS on demo datasets with different diffusion models. |
| [<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>](https://colab.research.google.com/github/zhangbingliang2019/DAPS/blob/main/scripts/Customization.ipynb) | Customizing DAPS for New Inverse Problems                  |



## 💻 Getting start locally

### 1. Prepare the Environment

- python 3.8
- PyTorch 2.3
- CUDA 12.1

Lower version of PyTorch with proper CUDA should work but not be fully tested.

```
# in DAPS folder

conda create -n DAPS python=3.8
conda activate DAPS

pip install -r requirements.txt

# (optional) install PyTorch with proper CUDA
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

We use [bkse](https://github.com/VinAIResearch/blur-kernel-space-exploring) for nonlinear blurring and [motionblur](https://github.com/LeviBorodenko/motionblur) for motion blur. **No further action required then**.



### 2. Prepare the pretrained checkpoint

#### 2.1 pixel diffusion model

Download the public available FFHQ and ImageNet checkpoint (ffhq_10m.pt, imagenet256.pt) [here](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh).

```
# in DAPS folder

mkdir checkpoints
mv {DOWNLOAD_DIR}/ffqh_10m.pt checkpoints/ffhq256.pt
mv {DOWNLOAD_DIR}/imagenet256.pt.pt checkpoints/imagenet256.pt
```

#### 2.2  latent diffusion model (LDM)

Download the public available LDM checkpoint for FFHQ and ImageNet with following commands:

```
# in DAPS folder

wget https://ommer-lab.com/files/latent-diffusion/ffhq.zip -P ./checkpoints
unzip checkpoints/ffhq.zip -d ./checkpoints
mv checkpoints/model.ckpt checkpoints/ldm_ffhq256.pt
rm checkpoints/ffhq.zip

wget https://ommer-lab.com/files/latent-diffusion/nitro/cin/model.ckpt -P ./checkpoints/
mv checkpoints/model.ckpt checkpoints/ldm_imagenet256.pt
```

#### 2.3 stable diffusion

Checkpoints will be automatically downloaded.

**(Optional)** For nonlinear deblur task, we need the pretrained model from [bkse](https://github.com/VinAIResearch/blur-kernel-space-exploring) at [here](https://drive.google.com/file/d/1vRoDpIsrTRYZKsOMPNbPcMtFDpCT6Foy/view?usp=drive_link):

```
# in DAPS folder

mv {DOWNLOAD_DIR}/GOPRO_wVAE.pth forward_operator/bkse/experiments/pretrained
```



### 3.  (Optional) Prepare the test dataset

You can download the selected test dataset used [here](https://drive.google.com/drive/folders/1RHNif32W0hvB4M75ppG1ypTChy-W3q3Z?usp=sharing), unzip and move to `dataset` folder. Otherwise, you can test on our provided 10 demo images at `dataset\demo-ffhq` and `dataset\demo-imagenet`.



### 4. Posterior sampling with DAPS

Now you are ready for run. For **phase retrieval** with `DAPS-1k` and `ffhq256ddpm` model in 4 runs for 10 demo FFHQ images in `dataset/demo-ffhq`:

```
python posterior_sample.py \
+data=demo-ffhq \
+model=ffhq256ddpm \
+task=phase_retrieval \
+sampler=edm_daps \
task_group=pixel \
save_dir=results \
num_runs=4 \
sampler.diffusion_scheduler_config.num_steps=5 \
sampler.annealing_scheduler_config.num_steps=200 \
batch_size=10 \
data.start_id=0 data.end_id=10 \
name=phase_retrieval_demo \
gpu=0
```

It takes about 8 minutes (2 for each run) and 6G GPU memory on a single NVIDIA A100-SXM4-80GB GPU. The results are saved at foloder `\results`.



#### Full commands on test dataset

Full comands used to reproduce the results in paper are provided in `commands` folder:

* pixel space diffusion: `commands/pixel.sh`
* latent diffusion (LDM): `commands/ldm.sh`
* stable diffusion: `commands/sd.sh`



#### Supported diffusion models

| Model                                                        | Dataset  | Model Config Name     | Sampler         | Task Group |
| ------------------------------------------------------------ | -------- | --------------------- | --------------- | ---------- |
| [ffhq-256](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh) | FFHQ     | ffhq256ddpm           | edm_daps        | pixel      |
| [imagenet-256](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh) | ImageNet | imagenet256ddpm       | edm_daps        | pixel      |
| [ldm-ffhq-256](https://github.com/CompVis/latent-diffusion?tab=readme-ov-file#unconditional-models) | FFHQ     | ffhq256ldm            | latent_edm_daps | ldm        |
| [ldm-imagenet-256](https://github.com/CompVis/latent-diffusion?tab=readme-ov-file#class-conditional-imagenet) | ImageNet | imagenet256ldm        | latent_edm_daps | ldm        |
| [sd-v1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) | Any      | stable-diffusion-v1.5 | sd_edm_daps     | sd         |
| [sd-v2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1) | Any      | stable-diffusion-v2.1 | sd_edm_daps     | sd         |



#### Command template

```
python posterior_sample.py \
+data={DATASET_CONFIG_NAME} \
+model={MODEL_CONFIG_NAME} \
+task={TASK_CONFIG_NAME} \
+sampler={SAMPLER_CONFIG_NAME} \
task_group={pixel, ldm, sd} # choose the used task parameters group \
save_dir=results \
num_runs={NUMBER_OF_RUNS} \
sampler.diffusion_scheduler_config.num_steps={DIFFUSION_ODE_STEPS} \
sampler.annealing_scheduler_config.num_steps={ANNEALING_STEPS} \
batch_size=100 \
name={SUB_FOLDER_NAME} \
gpu=0
```

Currently supported tasks are:

* `phase_retrieval`: phase retrival of oversample ratio of 2.0

* `down_sampling`: super resolution ($\times$4)

* `inpainting`:  128x128 box inpainting

* `inpainting_rand`: 70% random inpainting 

* `gaussian_blur`: gaussian deblur of kernel size 61 and intensity 3

* `motion_blur`: gaussian deblur of kernel size 61 and intensity 0.5

* `nonlinear_blur`: nonlinear deblur of default setting in bkse repo

* `hdr`: high dynamic range reconstruction of factor 2 



## Citation

If you find our work interesting, please consider citing

```
@misc{zhang2024improvingdiffusioninverseproblem,
      title={Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing}, 
      author={Bingliang Zhang and Wenda Chu and Julius Berner and Chenlin Meng and Anima Anandkumar and Yang Song},
      year={2024},
      eprint={2407.01521},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.01521}, 
}
```
