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

from lightly.utils import dist
import torch.distributed as torch_dist
import torch.distributions as tdist

torch.set_float32_matmul_precision('high')

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
        std = torch.exp(logVar / 2)
        _dist = tdist.Normal(mu, std)
        return _dist.rsample()
        
    def forward(self, x):
        x = self.interpreter(x)
        mu = self.fc_mu(x)
        logVar = self.fc_var(x)
        rp = self.reparameterize(mu, logVar)
        return rp, mu, logVar

class VCL(LightningModule):
    def __init__(self, batch_size_per_device: int, num_classes: int, num_gpus: int, temperature: float) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.batch_size_per_device = batch_size_per_device
        self.num_gpus = num_gpus

        resnet = resnet50()
        resnet.fc = Identity()  # Ignore classification head
        self.backbone = resnet
        
        self.temperature = temperature
        feature_dim = 2048
        self.projection_head = Projector(feature_dim, feature_dim, 128)
        
        self.criterion = NTXentLoss(temperature=temperature, gather_distributed=True)
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        self.online_classifier = OnlineLinearClassifier(num_classes=num_classes)

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten(1)

    def compute_objective(self, mu, logVar):
        # Gather tensors on rank 0 device
        gathered_mu = dist.gather(mu)
        gathered_logVar = dist.gather(logVar)
        gathered_mu = torch.cat(gathered_mu, 0)
        gathered_logVar = torch.cat(gathered_logVar, 0)

        # Compute the objective using gathered tensors
        gathered_p_mean, gathered_p_logVar, gathered_q_mean, gathered_q_logVar = gathered_mu[:,0,:], gathered_logVar[:,0,:], gathered_mu[:,1,:], gathered_logVar[:,1,:]
        gathered_p_std, gathered_q_std = torch.exp(gathered_p_logVar / 2.0), torch.exp(gathered_q_logVar / 2.0)
        
        objective = self.variational_loss(gathered_p_mean, gathered_p_std, gathered_q_mean, gathered_q_std).mean(1)
        _norm_p = - 0.5 * (1 + gathered_p_logVar - gathered_p_mean.pow(2) - gathered_p_logVar.exp()).mean(1) 
        _norm_q = - 0.5 * (1 + gathered_q_logVar - gathered_q_mean.pow(2) - gathered_q_logVar.exp()).mean(1)
        pairwise_objective = ((1 - torch.eye(gathered_p_mean.size(0)).to(_norm_q.device)) * self.pairwise_variational_loss(gathered_p_mean, gathered_p_std, gathered_q_mean, gathered_q_std)).sum(1) / (gathered_p_mean.size(0) - 1)

        return objective + _norm_p + _norm_q - pairwise_objective

    def variational_loss(self, p_mean, p_std, q_mean, q_std):
        # Create Gaussian distributions using mean and std
        p_dist = tdist.Normal(p_mean, p_std)
        q_dist = tdist.Normal(q_mean, q_std)

        # Calculate M as the average of the two distributions
        m_mean = (p_mean + q_mean) / 2
        m_std = (p_std + q_std) / 2
        m_dist = tdist.Normal(m_mean, m_std)

        # Calculate KL divergence between each distribution and M
        kl_pm = tdist.kl_divergence(p_dist, m_dist)
        kl_qm = tdist.kl_divergence(q_dist, m_dist)

        # Calculate JSD as the average of the KL divergences
        jsd = (kl_pm + kl_qm) / 2
        return jsd
    
    def pairwise_variational_loss(self, p_mean, p_std, q_mean, q_std):
        # Create Gaussian distributions using mean and std
        p_dist = tdist.Normal(p_mean, p_std)
        q_dist = tdist.Normal(q_mean, q_std)

        # Calculate M as the average of the two distributions
        m_mean = (p_mean + q_mean) / 2
        m_std = (p_std + q_std) / 2
        m_dist = tdist.Normal(m_mean, m_std)

        # Calculate KL divergence between each distribution and M
        kl_pm = tdist.kl_divergence(p_dist, m_dist)
        kl_qm = tdist.kl_divergence(q_dist, m_dist)

        # Calculate JSD as the average of the KL divergences
        jsd = (kl_pm.unsqueeze(1) + kl_qm.unsqueeze(0)) / 2
        return jsd.mean(-1)

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
        
        loss = self.criterion(z0, z1) + self.compute_objective(mu, logVar).mean()

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
        optimizer = torch.optim.AdamW(
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
            lr = 1e-3 * self.num_gpus * self.batch_size_per_device / 128,
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
