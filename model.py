import timm
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, model_name, target_size, pretrained=True, exportable=True):
        super(CustomModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, exportable=exportable)
        self.n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(self.n_features, target_size)
    
    def forward(self, x):
        output = self.model(x)
        return output