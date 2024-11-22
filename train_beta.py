import pytorch_lightning as pl
import torch
from torch import nn
import timm
from lightly.data import LightlyDataset
from lightly.data.multi_view_collate import MultiViewCollate
from beta_ntx_ent_loss import betaNTXentLoss, vloss
from lightly.transforms.simclr_transform import SimCLRTransform

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
        self.backbone = timm.create_model("vgg11_bn", num_classes=0)
        self.projection_head = Projector(self.backbone.num_features, self.backbone.num_features, 128)
        self.criterion = betaNTXentLoss(beta=0.005, temperature=0.07)
        self.vloss = vloss()

    def forward(self, x):
        x = self.backbone(x)
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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 600)
        return [optim], [scheduler]

model = SimCLR()

transform = SimCLRTransform()
dataset = LightlyDataset("## PATH TO YFCC DATASET ##", transform=transform)

collate_fn = MultiViewCollate()

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

import os
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

sub_dir = 'beta'
logs_root_dir = os.path.join(os.getcwd(), 'benchmark_logs')
logger = TensorBoardLogger(
        save_dir=os.path.join(logs_root_dir, sub_dir),
        name='',
        sub_dir=sub_dir,
    )
checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir, 'checkpoints')
    )

trainer = pl.Trainer(max_epochs=600, devices=1, accelerator="gpu", logger=logger, callbacks=[checkpoint_callback], enable_progress_bar = True)
trainer.fit(model=model, train_dataloaders=dataloader)
