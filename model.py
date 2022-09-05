import timm
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, model_name, target_size, pretrained=True, exportable=True):
        super(CustomModel, self).__init__()
        self.model = timm.create_model(model_name, target_size=target_size, pretrained=pretrained, exportable=exportable)
    
    def forward(self, x):
        output = self.model(x)
        return output