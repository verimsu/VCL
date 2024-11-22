"""
@author: Jun Wang
@date: 20201016
@contact: jun21wangustc@gmail.com
"""

import torch
import torchvision

class ModelLoader:
    """Load a model by network and weights file.

    Attributes: 
        model(object): the model definition file.
    """
    def __init__(self):
        backbone = torchvision.models.vgg16_bn(pretrained=True)
        del backbone.classifier[6]
        backbone.classifier[0] = torch.nn.Linear(in_features=8192, out_features=4096, bias=True)
        backbone.avgpool = torch.nn.Identity()
        self.model = backbone

    def load_model_default(self, model_path):
        """The default method to load a model.
        
        Args:
            model_path(str): the path of the weight file.
        
        Returns:
            model(object): initialized model.
        """
        self.model.load_state_dict(torch.load(model_path), strict=True) 
        model = self.model.cuda()
        return model

    def load_model(self, model_path):
        """The custom method to load a model.
        
        Args:
            model_path(str): the path of the weight file.
        
        Returns:
            model(object): initialized model.
        """
        self.model.load_state_dict(torch.load(model_path), strict=True) 
        model = self.model.cuda()
        return model
