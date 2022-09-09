import timm
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, model_name, target_size, pretrained=True, exportable=True):
        """
        Parameters :-

        model_name : name of model from Timm
        target_size : number of classes
        pretrained : Want ImageNet pretrained model?
        exportable : Want to export your model to Onnx in future?
        """
        super(CustomModel, self).__init__()
        self.model = timm.create_model(model_name, num_classes=target_size, pretrained=pretrained, exportable=exportable)
    
    def forward(self, x):
        output = self.model(x)
        return output