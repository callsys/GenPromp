<div align=center>
  
# Generative Prompt Model for Weakly Supervised Object Localization [ICCV'23]
</div>

<div align=center>
  
  <p >GenPromp is a Weakly Supervised Object Localization (WSOL) method based on generative models.</p>
</div>

<div align=center>
  
[![arXiv preprint](http://img.shields.io/badge/arXiv-2307.09756-b31b1b)](https://arxiv.org/abs/2307.09756)
</div>

<div align=center>
  
<img src="assets/intro.png" width="69%">
</div>

## 1. Introduction

Weakly supervised object localization (WSOL) remains challenging when learning object localization models from image category labels. Conventional methods that discriminatively train activation models ignore representative yet less discriminative object parts. In this study, we propose a generative prompt model (GenPromp), defining the first generative pipeline to localize less discriminative object parts by formulating WSOL as a conditional image denoising procedure. During training, GenPromp converts image category labels to learnable prompt embeddings which are fed to a generative model to conditionally recover the input image with noise and learn representative embeddings. During inference, GenPromp combines the representative embeddings with discriminative embeddings (queried from an off-the-shelf vision-language model) for both representative and discriminative capacity. The combined embeddings are finally used to generate multi-scale high-quality attention maps, which facilitate localizing full object extent. Experiments on CUB-200-2011 and ILSVRC show that GenPromp respectively outperforms the best discriminative models, setting a solid baseline for WSOL with the generative model.


## 2. Results

<div align=center>
  
<img src="assets/results.png" width="99%">
</div>

## 3. Get Start

### 3.1 Installation

To setup the environment of GenPromp, we use `conda` to manage our dependencies. Our developers use `CUDA 11.3` to do experiments. Run the following commands to install GenPromp:
 ```
conda create -n gpm python=3.8 -y && conda activate gpm
pip install --upgrade pip
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install --upgrade diffusers[torch]==0.13.1
pip install transformers==4.29.2 accelerate==0.19.0
pip install matplotlib opencv-python OmegaConf tqdm
 ```

### 3.2 Dataset & Files Preparation
To train GenPromp with pre-training weights and infer GenPromp with the given weights, download the files in the table and arrange the files according to the file tree below.


  | Dataset & Files                        | Download                                                               | Usage                                                                 |
  | -------------------------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------------------- |
  | data/ImageNet_ILSVRC2012 (146GB)            | [Official Link](http://image-net.org/)                        | Benchmark dataset                                                     |
  | data/CUB_200_2011 (1.2GB)                   | [Official Link](http://www.vision.caltech.edu/datasets/cub_200_2011/)      | Benchmark dataset                                        |
  | ckpts/pretrains (5.2GB)                | [Official Link](), [Google Drive](), [Baidu Drive]()          | Stable Diffusion pretrain weights                                     |
  | ckpts/classifications (1.2GB)          | [Google Drive](), [Baidu Drive]()                                      | Classfication results on benchmark datasets                           |
  | ckpts/imagenet750 (3.3.GB)             | [Google Drive](), [Baidu Drive]()                                      | Weights that achieves 75.0% GT-Known Loc on ImageNet                  |
  | ckpts/cub980 (832KB)                   | [Google Drive](), [Baidu Drive]()                                      | Weights that achieves 98.0% GT-Known Loc on CUB                       |

```text
    |--GenPromp/
      |--data/
        |--ImageNet_ILSVRC2012/
           |--ILSVRC2012_list/
           |--train/
           |--val/
        |--CUB_200_2011
           |--attributes/
           |--images/
           ...
      |--ckpts/
        |--pretrains/
          |--stable-diffusion-v1-4/
        |--classifications/
          |--cub_efficientnetb7.json
          |--imagenet_efficientnet-b7_3rdparty_8xb32-aa-advprop_in1k.json
        |--ckpt_imagenet/
          |--tokens/
          |--unet/
        |--ckpt_cub/
          |--tokens/
      |--configs/
      |--datasets
      |--models
      |--main.py
```


### 3.3 Training

Here is a training example of GenPromp on ImageNet.
```
accelerate config
accelerate launch python main.py --function train_token --config configs/imagenet.yml --opt "{'train':{'save_path':'ckpts/imagenet/'}}"
accelerate launch python main.py --function train_unet --config configs/imagenet_stage2.yml --opt "{'train':{'load_token_path':'ckpts/imagenet/tokens/','save_path':'ckpts/imagenet/'}}"
```
`accelerate` is used for multi-GPU training. In the first training stage, the weights of concept tokens of the representative embeddings are learned and saved to `ckpts/imagenet/`. In the second training stage, the weights of the learned concept tokens are loaded from `ckpts/imagenet/tokens`, then the weights of the UNet are finetuned and saved to `ckpts/imagenet/`. Other configurations can be seen in the config files (i.e. `configs/imagenet.yml` and `configs/imagenet_stage2.yml`) and can be modified by `--opt` with a parameter dict.

Here is a training example of GenPromp on CUB_200_2011.
```
accelerate config
accelerate launch python main.py --function train_token --config configs/cub.yml --opt "{'train':{'save_path':'ckpts/cub/'}}"
accelerate launch python main.py --function train_unet --config configs/cub_stage2.yml --opt "{'train':{'load_token_path':'ckpts/cub/tokens/','save_path':'ckpts/cub/'}}"
```

### 3.4 Inference

```
python main.py --function test --config configs/imagenet_stage2.yml --opt "{'test':{'load_token_path':'ckpts/imagenet750/tokens/','load_unet_path':'ckpts/imagenet750/unet/','save_log_path':'ckpts/imagnet750/log.txt'}}"
```

```
python main.py --function test --config configs/cub.yml --opt "{'test':{'load_token_path':'ckpts/cub980/tokens/','save_log_path':'ckpts/cub980/log.txt'}}"
```

### 3.5 Extra Options

## 4. License

- The repository is released under the MIT license.

## 5. Contacts
If you have any question about our work or this repository, please don't hesitate to contact us by emails.
- [zhaoyuzhong20@mails.ucas.ac.cn](zhaoyuzhong20@mails.ucas.ac.cn)
- [wanfang@ucas.ac.cn](wanfang@ucas.ac.cn)

You can also open an issue under this project.

## 6. Acknowledgment

- Part of the code is borrowed from [TS-CAM](https://github.com/vasgaowei/TS-CAM), [diffusers](https://github.com/huggingface/diffusers), and [prompt-to-prompt](https://github.com/google/prompt-to-prompt/), we sincerely thank them for their contributions to the community.


## 7. Citation

```text
@article{zhao2023generative,
  title={Generative Prompt Model for Weakly Supervised Object Localization},
  author={Zhao, Yuzhong and Ye, Qixiang and Wu, Weijia and Shen, Chunhua and Wan, Fang},
  journal={arXiv preprint arXiv:2307.09756},
  year={2023}
}
```
