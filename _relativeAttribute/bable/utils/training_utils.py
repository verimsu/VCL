import torch


def get_optimizer_and_lr_schedule(model):

    optimizer = torch.optim.AdamW(
            model.parameters()
        )        
    lr_scheduler = None
        # torch.optim.lr_scheduler.LambdaLR(optimizer, )
        # torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, )
        # torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, )
        # torch.optim.lr_scheduler.CyclicLR(optimizer, )
        # torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, )
        # torch.optim.lr_scheduler.OneCycleLR(optimizer, )

    return optimizer, lr_scheduler
