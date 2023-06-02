# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import pytorch_lightning as pl
import torch
import torchvision
from torch import nn

from lightly.data import LightlyDataset
from lightly.data.multi_view_collate import MultiViewCollate
from lightly.loss import NTXentLoss
from beta_ntx_ent_loss import vloss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms.simclr_transform import SimCLRTransform

class vloss(nn.Module):
    def __init__(self):
        super(vloss,self).__init__()

    def l1(self, x, y):
        return x - y

    def l2(self, x, y):
        return (x - y).square()

    def forward(self, mu, logVar):
        lVp = logVar[:,0,:]
        lVq = logVar[:,1,:]
        mp = mu[:,0,:]
        mq = mu[:,1,:]
        sm = 0.5 * (lVp.exp() + lVq.exp())
        lSm = sm.log()/2
        mm = 0.5 * (mp + mq)
        eq_ = (- 0.5 * (self.l1(lVp/2,lSm) - self.l1(lVq/2,lSm)) + 0.25 * ((self.l2(mp,mm) + self.l2(mq,mm)) / sm)).mean()
        norm_ = - 0.5 * (1 + logVar - mu.pow(2) - logVar.exp()).mean()
        return eq_ + norm_

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
    
class SimCLR(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.vgg16_bn()
        del self.backbone.classifier[6]
        self.backbone.avgpool = torch.nn.Identity()
        self.backbone.classifier[0] = torch.nn.Linear(in_features=8192, out_features=4096, bias=True)    
        self.projection_head = Projector(4096, 2048, 128)
        self.criterion = NTXentLoss(temperature=0.07)
        self.vloss = vloss()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z, mu, logVar = self.projection_head(x)
        return z, mu, logVar

    def training_step(self, batch, batch_index):
        (x0, x1), _, _ = batch
        z0, mu0, logVar0 = self.forward(x0)
        z1, mu1, logVar1 = self.forward(x1)

        mu = torch.cat((mu0.unsqueeze(1), mu1.unsqueeze(1)),1)
        logVar = torch.cat((logVar0.unsqueeze(1), logVar1.unsqueeze(1)),1)

        loss = self.criterion(z0, z1) + self.vloss(mu, logVar)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 500)
        return [optim], [scheduler] 


model = SimCLR()

transform = SimCLRTransform(input_size=128)
dataset = LightlyDataset("/truba_scratch/meyavuz/YFCC392K", transform=transform)

collate_fn = MultiViewCollate()

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=128,
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

import os
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

sub_dir = 'VCL'
logs_root_dir = os.path.join(os.getcwd(), 'benchmark_logs')
logger = TensorBoardLogger(
        save_dir=os.path.join(logs_root_dir, sub_dir),
        name='',
        sub_dir=sub_dir,
    )
checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir, 'checkpoints')
    )

trainer = pl.Trainer(max_epochs=500, devices=1, accelerator="gpu", logger=logger, callbacks=[checkpoint_callback], enable_progress_bar = True)
trainer.fit(model=model, train_dataloaders=dataloader)
