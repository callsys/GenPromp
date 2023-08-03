<div align=center>

# Generative Prompt Model for Weakly Supervised Object Localization [ICCV'23]

</div>

<div align=center>
  <p >GenPromp is a Weakly Supervised Object Localization (WSOL) method, which significantly outperforms its discriminative counterparts on commonly used benchmarks, setting a solid baseline for WSOL with generative models.</p>
</div>

<div align=center>

[![arXiv preprint](http://img.shields.io/badge/arXiv-2307.09756-b31b1b)](https://arxiv.org/abs/2307.09756)

</div>

## 1. Introduction

## 2. Results

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

### 3.2 Dataset Preparation

### 3.3 Training

### 3.4 Inference

### 3.5 Extra Options

## 4. License

- The repository is released under the MIT license.

## 5. Acknowledgment

- Part of the code is borrowed from [TS-CAM](https://github.com/vasgaowei/TS-CAM), [diffusers](https://github.com/huggingface/diffusers), and [prompt-to-prompt](https://github.com/google/prompt-to-prompt/), we sincerely thank them for their contributions to the community.

## 6. Citation

```text
@article{zhao2023generative,
  title={Generative Prompt Model for Weakly Supervised Object Localization},
  author={Zhao, Yuzhong and Ye, Qixiang and Wu, Weijia and Shen, Chunhua and Wan, Fang},
  journal={arXiv preprint arXiv:2307.09756},
  year={2023}
}
```
