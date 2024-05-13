# VCL: Variational Self-supervised Contrastive Learning
## ImageNet ResNet50

Reference implementations for self-supervised learning (SSL) methods on ImageNet with
ResNet50 backbones.

**Note**
> The benchmarks are still in beta phase and there will be breaking changes and
frequent updates. PRs for new methods are highly welcome!

**Goals**
* Provide easy to use/adapt reference implementations of SSL methods.
* Implemented methods should be self-contained and use the Lightly building blocks.
See [simclr.py](simclr.py).
* Remain as framework agnostic as possible. The benchmarks currently only rely on PyTorch and PyTorch Lightning.


**Non-Goals**
* Lightly doesn't strive to be an end-to-end SSL framework with vast configuration options.
Instead, we try to provide building blocks and examples to make it as easy as possible to
build on top of existing SSL methods.

You can find benchmark resuls in our [docs](https://docs.lightly.ai/self-supervised-learning/getting_started/benchmarks.html).

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
