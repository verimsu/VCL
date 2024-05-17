import math
from typing import List, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Identity
from torchvision.models import resnet50

from lightly.loss.ntx_ent_loss import NTXentLoss
from lightly.models.utils import get_weight_decay_parameters
from lightly.transforms import SimCLRTransform
from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.utils.lars import LARS
from lightly.utils.scheduler import CosineWarmupScheduler

import torch.nn as nn
import torch.nn.functional as F

import torch.distributions as tdist

class vloss(nn.Module):
    def __init__(self):
        super(vloss,self).__init__()

    def l1(self, x, y):
        return x - y

    def l2(self, x, y):
        return (x - y).square()

    def jsd_gaussian(self, p_mean, q_mean, logVar_):
        p_std = (logVar_[:,0,:] / 2).exp()
        q_std = (logVar_[:,1,:] / 2).exp()

        # Create Gaussian distributions using mean and std
        p_dist = tdist.Normal(p_mean, p_std)
        q_dist = tdist.Normal(q_mean, q_std)

        # Calculate M as the average of the two distributions
        m_dist = tdist.Normal((p_mean + q_mean) / 2, (p_std + q_std) / 2)

        # Calculate KL divergence between each distribution and M
        kl_pm = tdist.kl_divergence(p_dist, m_dist)
        kl_qm = tdist.kl_divergence(q_dist, m_dist)

        # Calculate JSD as the average of the KL divergences
        jsd = (kl_pm + kl_qm) / 2

        return jsd
    
    def forward(self, mu, logVar):
        mp = mu[:,0,:]
        mq = mu[:,1,:]
        mm = 0.5 * (mp + mq)

        jsd = self.jsd_gaussian(mp, mq, logVar).mean()
        norm_ = - 0.5 * (1 + logVar - mu.pow(2) - logVar.exp()).mean()
        return jsd + 0.002 * norm_

class Projector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Projector, self).__init__()
        self.interpreter = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True))
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_var = nn.Linear(hidden_dim, output_dim)

    def reparameterize(self, mu, logVar):
        std = logVar.mul(0.5).exp_()
        eps = torch.randn_like(std)
        return eps * std + mu      
        
    def forward(self, x):
        x = self.interpreter(x)
        mu = self.fc_mu(x)
        logVar = self.fc_var(x)
        rp = self.reparameterize(mu, logVar)
        return rp, mu, logVar

class VCL(LightningModule):
    def __init__(self, batch_size_per_device: int, num_classes: int, num_gpus: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.batch_size_per_device = batch_size_per_device
        self.num_gpus = num_gpus

        resnet = resnet50()
        resnet.fc = Identity()  # Ignore classification head
        self.backbone = resnet
        
        feature_dim = 2048
        self.projection_head = Projector(feature_dim, feature_dim, 128)
        
        self.criterion = NTXentLoss(temperature=0.07, gather_distributed=True)
        self.vloss = vloss()

        self.online_classifier = OnlineLinearClassifier(num_classes=num_classes)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z, mu, logVar = self.projection_head(x)
        return x, z, mu, logVar

    def training_step(
        self, batch: Tuple[List[Tensor], Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        (x0, x1), targets = batch[0], batch[1]
        feature1, z0, mu0, logVar0 = self.forward(x0)
        feature2, z1, mu1, logVar1 = self.forward(x1)
        features = torch.cat([feature1, feature2])

        mu = torch.cat((mu0.unsqueeze(1), mu1.unsqueeze(1)),1)
        logVar = torch.cat((logVar0.unsqueeze(1), logVar1.unsqueeze(1)),1)

        loss = self.criterion(z0, z1) + self.vloss(mu, logVar)        

        self.log(
            "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=len(targets)
        )

        cls_loss, cls_log = self.online_classifier.training_step(
            (features.detach(), targets.repeat(len(batch[0]))), batch_idx
        )
        self.log_dict(cls_log, sync_dist=True, batch_size=len(targets))
        return loss + cls_loss

    def validation_step(
        self, batch: Tuple[Tensor, Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        images, targets = batch[0], batch[1]
        features, _, _, _ = self.forward(images)        
        cls_loss, cls_log = self.online_classifier.validation_step(
            (features.detach(), targets), batch_idx
        )
        self.log_dict(cls_log, prog_bar=True, sync_dist=True, batch_size=len(targets))
        return cls_loss

    def configure_optimizers(self):
        # Don't use weight decay for batch norm, bias parameters, and classification
        # head to improve performance.
        params, params_no_weight_decay = get_weight_decay_parameters(
            [self.backbone, self.projection_head]
        )
        optimizer = LARS(
            [
                {"name": "simclr", "params": params},
                {
                    "name": "simclr_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                },
                {
                    "name": "online_classifier",
                    "params": self.online_classifier.parameters(),
                    "weight_decay": 0.0,
                },
            ],
            # Square root learning rate scaling improves performance for small
            # batch sizes (<=2048) and few training epochs (<=200). Alternatively,
            # linear scaling can be used for larger batches and longer training:
            # lr=0.3 * self.batch_size_per_device * 2 / 256,
            # See Appendix B.1. in the SimCLR paper https://arxiv.org/abs/2002.05709
            lr=0.075 * math.sqrt(self.batch_size_per_device * 2),
            momentum=0.9,
            # Note: Paper uses weight decay of 1e-6 but reference code 1e-4. See:
            # https://github.com/google-research/simclr/blob/2fc637bdd6a723130db91b377ac15151e01e4fc2/README.md?plain=1#L103
            weight_decay=1e-6,
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=int(
                    self.trainer.estimated_stepping_batches
                    / self.trainer.max_epochs
                    * 10
                ),
                max_epochs=int(self.trainer.estimated_stepping_batches),
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]


transform = SimCLRTransform()
