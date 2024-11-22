import torch
from torch import nn
import torchvision
from bable.models.base_siamese_model import BaseSiameseModel


class DRN(BaseSiameseModel):
    def __init__(self, args):

        extractor = torchvision.models.vgg16_bn()
        extractor.classifier[0] = torch.nn.Linear(in_features=8192, out_features=4096, bias=True)
        extractor.avgpool = torch.nn.Identity()
        extractor.classifier[6] = nn.Linear(4096, 1)

        super(DRN, self).__init__(extractor)
