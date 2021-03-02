# Neural Architecture Search on ImageNet in Four GPU Hours: A Theoretically Inspired Perspective [[PDF](https://arxiv.org/pdf/2102.11535.pdf)]

[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/VITA-Group/TENAS.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/VITA-Group/TENAS/context:python)
[![MIT licensed](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE.md)

Wuyang Chen, Xinyu Gong, Zhangyang Wang

In ICLR 2021.

## Overview

<!--
<p align="center">
  <img src="images/????.gif" alt="ntk_regions" width="300"/></br>
  <span align="center">????</span>
</p>
-->

We present TE-NAS, the first published training-free neural architecture search method with extremely fast search speed (no gradient descent at all!) and high-quality performance.

Highlights:
* **Trainig-free and label-free NAS**: we achieved extreme fast neural architecture search without a single gradient descent.
* **Bridging the theory-application gap**: We identified two training-free indicators to rank the quality of deep networks: the condition number of their NTKs, and the number of linear regions in their input space.
* **SOTA**: TE-NAS achieved extremely fast search speed (one 1080Ti, 20 minutes on NAS-Bench-201 space / four hours on DARTS space on ImageNet) and maintains competitive accuracy.

<!--
<p align="center">
<img src="images/????.png" alt="201" width="550"/></br>
</p>
<p align="center">
<img src="images/????.png" alt="darts_cifar10" width="550"/></br>
</p>
<p align="center">
<img src="images/????.png" alt="darts_imagenet" width="550"/></br>
</p>
-->

<!--
## Methods

<p align="center">
<img src="images/????.png" alt="algorithm" width="800"/></br>
</p>
-->

## Prerequisites
- Ubuntu 16.04
- Python 3.6.9
- CUDA 10.1 (lower versions may work but were not tested)
- NVIDIA GPU + CuDNN v7.3

This repository has been tested on GTX 1080Ti. Configurations may need to be changed on different platforms.

## Installation
* Clone this repo:
```bash
git clone https://github.com/chenwydj/TENAS.git
cd TENAS
```
* Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
### 0. Prepare the dataset
* Please follow the guideline [here](https://github.com/D-X-Y/AutoDL-Projects#requirements-and-preparation) to prepare the CIFAR-10/100 and ImageNet dataset, and also the NAS-Bench-201 database.
* **Remember to properly set the `TORCH_HOME` and `data_paths` in the `prune_launch.py`.**

### 1. Search
#### [NAS-Bench-201 Space](https://openreview.net/forum?id=HJxyZkBKDr)
```python
python prune_launch.py --space nas-bench-201 --dataset cifar10 --gpu 0
python prune_launch.py --space nas-bench-201 --dataset cifar100 --gpu 0
python prune_launch.py --space nas-bench-201 --dataset ImageNet16-120 --gpu 0
```

#### [DARTS Space](https://openreview.net/forum?id=S1eYHoC5FX) ([NASNET](https://openaccess.thecvf.com/content_cvpr_2018/html/Zoph_Learning_Transferable_Architectures_CVPR_2018_paper.html))
```python
python prune_launch.py --space darts --dataset cifar10 --gpu 0
python prune_launch.py --space darts --dataset imagenet-1k --gpu 0
```

### 2. Evaluation
* For architectures searched on `nas-bench-201`, the accuracies are immediately available at the end of search (from the console output).
* For architectures searched on `darts`, please use [DARTS_evaluation](https://github.com/chenwydj/DARTS_evaluation) for training the searched architecture from scratch and evaluation.


## Citation
```
@inproceedings{chen2020tenas,
  title={Neural Architecture Search on ImageNet in Four GPU Hours: A Theoretically Inspired Perspective},
  author={Chen, Wuyang and Gong, Xinyu and Wang, Zhangyang},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```

## Acknowledgement
* Code base from [NAS-Bench-201](https://github.com/D-X-Y/AutoDL-Projects/blob/master/docs/NAS-Bench-201.md).
