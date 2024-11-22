# Variational Contrastive Learning (VCL) for Face Recognition
A robust self-supervised learning method for handling noisy and unlabeled data in multi-label settings, with applications to face understanding.

## Overview

VCL combines variational contrastive learning with beta-divergence to learn effectively from unlabeled and noisy datasets. The method is particularly effective for face attribute recognition and verification tasks[1].

## Key Features

- Self-supervised learning approach robust to data noise
- Variational contrastive framework with beta-divergence
- Effective for multi-label classification problems
- Superior performance on face understanding tasks

## Architecture

The model consists of three main components:
- Feature extraction backbone (ResNet10t or VGG11bn)
- Gaussian sampling head for distribution learning
- Contrastive learning framework with augmentations[1]

# F1-scores on the CelebA test set

This table shows F1-scores on the CelebA test set when YFCC-CelebA dataset is used for pretraining. The models are then fine-tuned using 1% and 10% of the labeled CelebA dataset. Bold indicates the best results; underline indicates the second best.

| Method | Supervised Pretraining w. Imagenet | Self-supervised Training w. YFCC-CelebA | CelebA 1% Resnet10t | CelebA 1% VGG11bn | CelebA 10% Resnet10t | CelebA 10% VGG11bn |
|--------|-----------------------------------|----------------------------------------|---------------------|-------------------|----------------------|---------------------|
| **Transfer Learning** | yes | no | 0.5784 | 0.5673 | 0.6517 | 0.6654 |
| BarlowTwins | no | yes | <u>0.5894</u> | 0.5712 | 0.6647 | 0.6665 |
| BYOL | no | yes | 0.5726 | 0.5683 | 0.6747 | 0.6725 |
| MoCo | no | yes | 0.5467 | 0.5535 | <u>0.6987</u> | <u>0.6896</u> |
| NNCLR | no | yes | 0.5337 | 0.5474 | 0.6487 | 0.6359 |
| SimCLR | no | yes | 0.5564 | 0.5652 | 0.6748 | 0.6693 |
| SimSiam | no | yes | 0.5484 | 0.5565 | 0.6684 | 0.6641 |
| Tico | no | yes | 0.5637 | <u>0.5790</u> | 0.6738 | 0.6683 |
| SwaV | no | yes | 0.5638 | 0.5741 | 0.6646 | 0.6597 |
| **Proposed** |
| VCL | no | yes | 0.5836 | 0.5719 | 0.6848 | 0.6796 |
| VCL (beta) | no | yes | **0.5998** | **0.5958** | **0.7098** | **0.6998** |

## Installation

```bash
git clone https://github.com/username/VCL
cd VCL
pip install -r requirements.txt
```

## Usage

```bash
python train_beta.py
```

## Training Parameters

- Optimizer: AdamW
- Learning rate: 1e-3
- Weight decay: 0.01
- Batch size: 128
- Temperature: 0.07
- Beta: 0.005[1]

## Citation

```bibtex
@INPROCEEDINGS{10582001,
  author={Yavuz, Mehmet Can and Yanikoglu, Berrin},
  booktitle={2024 IEEE 18th International Conference on Automatic Face and Gesture Recognition (FG)}, 
  title={Self-Supervised Variational Contrastive Learning with Applications to Face Understanding}, 
  year={2024},
  volume={},
  number={},
  pages={1-9},
  keywords={Face recognition;Semantics;Noise;Contrastive learning;Gesture recognition;Noise measurement},
  doi={10.1109/FG59268.2024.10582001}}

```

## License

MIT License
