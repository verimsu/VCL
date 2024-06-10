# VCL: Variational Self-supervised Contrastive Learning
## ImageNet ResNet50

Reference implementations for Variational Contrastive Learning (VCL) methods on ImageNet with ResNet50 backbones.

**100 epoch results by Variational Contrastive Learning algorithms**

| Model            | Batch Size | Epochs | Linear Top1 | Linear Top5 | Finetune Top1 | Finetune Top5 | kNN Top1 | kNN Top5 | Checkpoint                                                                 |
|------------------|----------|------------|--------|--------------|--------------|---------------|---------------|----------|----------------------------------------------------------------------------|
| VCL               | 256        | 100    | 61.19     | 83.55        | 70.41             | 90.10             | 40.67     | 70.11        | [link](https://huggingface.co/ogrenenmakine/vcl/resolve/main/vcl_e100.ckpt) |
| VCL               | 256        | 600    | -            | -            | -             | -             | -        | -        | -                                                                          |

**100 epoch results by competitor algorithms**

| Model            | Batch Size | Epochs | Linear Top1 | Linear Top5 | Finetune Top1 | Finetune Top5 | kNN Top1 | kNN Top5 | Checkpoint                                                                 |
|------------------|------------|--------|--------------|--------------|---------------|---------------|----------|----------|----------------------------------------------------------------------------|
| BarlowTwins       | 256        | 100    | 62.9         | 84.3         | 72.6          | 90.9          | 45.6     | 73.9     | [link](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_barlowtwins_2023-08-18_00-11-03/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt) |
| BYOL              | 256        | 100    | 62.5         | 85.0         | 74.5          | 92.0          | 46.0     | 74.8     | [link](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_byol_2024-02-14_16-10-09/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt) |
| MoCoV2            | 256        | 100    | 61.5         | 84.1         | 74.3          | 91.9          | 41.8     | 72.2     | [link](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_mocov2_2024-02-18_10-29-14/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt) |
| SimCLR*           | 256        | 100    | 63.2         | 85.2         | 73.9          | 91.9          | 44.8     | 73.9     | [link](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_simclr_2023-06-22_09-11-13/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt) |
| SimCLR* + DCL     | 256        | 100    | 65.1         | 86.2         | 73.5          | 91.7          | 49.6     | 77.5     | [link](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_dcl_2023-07-04_16-51-40/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt) |
| SimCLR* + DCLW    | 256        | 100    | 64.5         | 86.0         | 73.2          | 91.5          | 48.5     | 76.8     | [link](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_dclw_2023-07-07_14-57-13/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt) |
| SwAV              | 256        | 100    | 67.2         | 88.1         | 75.4          | 92.7          | 49.5     | 78.6     | [link](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_swav_2023-05-25_08-29-14/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt) |
| TiCo              | 256        | 100    | 49.7         | 74.4         | 72.7          | 90.9          | 26.6     | 53.6     | [link](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_tico_2024-01-07_18-40-57/pretrain/version_0/checkpoints/epoch%3D99-step%3D250200.ckpt) |
| VICReg            | 256        | 100    | 63.0         | 85.4         | 73.7          | 91.9          | 46.3     | 75.2     | [link](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_vicreg_2023-09-11_10-53-08/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt) |
## Installation

Follow these steps to set up your environment and install the necessary packages.

#### Step 1: Create a Conda Environment

First, create a new Conda environment with Python 3.8 (or another compatible version).

```bash
conda create --name lightly-env python=3.8
conda activate lightly-env
```

#### Step 2: Install PyTorch with GPU Support
Install PyTorch with the appropriate CUDA version. You can find the correct command for your system on the PyTorch website.
For example, to install PyTorch with CUDA 11.3, use:

```bash
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
```

#### Step 3: Install Lightly
Install the Lightly library using pip.

```bash
pip install lightly
```

#### Step 4: Verify Installation
Ensure that PyTorch can access the GPU. You can do this by running a simple script in Python.

```bash
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Run Benchmark

To run the benchmark first download the ImageNet ILSVRC2012 split from here: https://www.image-net.org/challenges/LSVRC/2012/.


Then start the benchmark with:
```
python main.py --epochs 100 --train-dir /datasets/imagenet/train --val-dir /datasets/imagenet/val --num-workers 12 --devices 2 --batch-size-per-device 128 --skip-finetune-eval
```

Or with SLURM, create the following script (`run_imagenet.sh`):
```
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:2            # Must match --devices argument
#SBATCH --ntasks-per-node=2     # Must match --devices argument
#SBATCH --cpus-per-task=16      # Must be >= --num-workers argument
#SBATCH --mem=0

eval "$(conda shell.bash hook)"

conda activate lightly-env
srun python main.py --epochs 100 --train-dir /datasets/imagenet/train --val-dir /datasets/imagenet/val --num-workers 12 --devices 2 --batch-size-per-device 128
conda deactivate
```

And run it with sbatch: `sbatch run_imagenet.sh`.
