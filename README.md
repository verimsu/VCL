# VCL: Variational Self-supervised Contrastive Learning
## ImageNet ResNet50

Reference implementations for variational contrastive learning (VCL) methods on ImageNet with ResNet50 backbones.

**100 epoch results by competitor algorithms**

.. csv-table:: Imagenet benchmark results.
  :header: "Model", "Backbone", "Batch Size", "Epochs", "Linear Top1", "Linear Top5", "Finetune Top1", "Finetune Top5", "kNN Top1", "kNN Top5", "Checkpoint"
  :widths: 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20

  "BarlowTwins", "Res50", "256", "100", "62.9", "84.3", "72.6", "90.9", "45.6", "73.9", "[link](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_barlowtwins_2023-08-18_00-11-03/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt)"
  "BYOL", "Res50", "256", "100", "62.5", "85.0", "74.5", "92.0", "46.0", "74.8", "[link](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_byol_2024-02-14_16-10-09/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt)"
  "DINO", "Res50", "128", "100", "68.2", "87.9", "72.5", "90.8", "49.9", "78.7", "[link](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_dino_2023-06-06_13-59-48/pretrain/version_0/checkpoints/epoch%3D99-step%3D1000900.ckpt)"
  "MAE", "ViT-B/16", "256", "100", "46.0", "70.2", "81.3", "95.5", "11.2", "24.5", "[link](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_vitb16_mae_2024-02-25_19-57-30/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt)"
  "MoCoV2", "Res50", "256", "100", "61.5", "84.1", "74.3", "91.9", "41.8", "72.2", "[link](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_mocov2_2024-02-18_10-29-14/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt)"
  "SimCLR*", "Res50", "256", "100", "63.2", "85.2", "73.9", "91.9", "44.8", "73.9", "[link](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_simclr_2023-06-22_09-11-13/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt)"
  "SimCLR* + DCL", "Res50", "256", "100", "65.1", "86.2", "73.5", "91.7", "49.6", "77.5", "[link](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_dcl_2023-07-04_16-51-40/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt)"
  "SimCLR* + DCLW", "Res50", "256", "100", "64.5", "86.0", "73.2", "91.5", "48.5", "76.8", "[link](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_dclw_2023-07-07_14-57-13/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt)"
  "SwAV", "Res50", "256", "100", "67.2", "88.1", "75.4", "92.7", "49.5", "78.6", "[link](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_swav_2023-05-25_08-29-14/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt)"
  "TiCo", "Res50", "256", "100", "49.7", "74.4", "72.7", "90.9", "26.6", "53.6", "[link](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_tico_2024-01-07_18-40-57/pretrain/version_0/checkpoints/epoch%3D99-step%3D250200.ckpt)"
  "VICReg", "Res50", "256", "100", "63.0", "85.4", "73.7", "91.9", "46.3", "75.2", "[link](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_vicreg_2023-09-11_10-53-08/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt)"
  
.... csv-table:: Imagenet benchmark results.
  :header: "Model", "Backbone", "Batch Size", "Epochs", "Linear Top1", "Linear Top5", "Finetune Top1", "Finetune Top5", "kNN Top1", "kNN Top5", "Checkpoint"
  :widths: 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20

  "VCL", "Res50", "256", "100", "-", "-", "-", "-", "-", "", "-"

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
