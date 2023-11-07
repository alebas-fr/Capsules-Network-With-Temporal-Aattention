import torch.nn as nn
import torch.nn.functional as F

from models.attention import EncoderSelfAttention
from models.features_caps import features_extraction,Resnet


class _GestureTransformer(nn.Module):
    """Multi Modal model for gesture recognition on 3 or 1 channel"""
    def __init__(self,device,backbone,in_planes: int, out_planes: int,
                 pretrained: bool = False,input_dim=512,layers_to_unfreeze=1,layers_to_delete=2,
                 **kwargs):
        super(_GestureTransformer, self).__init__()

        self.in_planes = in_planes
        self.device = device
        self.conv_name = backbone
        self.conv_model = None
        self.out_caps = out_planes
        if out_planes == 28: # For SHREC28 we reduce the number to 14
            self.out_caps = 14


        if self.conv_name.lower()=="resnet":
            self.conv_model = Resnet(pretrained,layers_to_unfreeze,layers_to_delete,in_planes)
        else:
            raise NotImplementedError("The model {} is not supported!".format(self.conv_name))
        self.conv_model.to(device)
        self.features = features_extraction(self.conv_model,in_planes)

        self.self_attention = EncoderSelfAttention(device,input_dim,64,64, **kwargs)

        self.pool = nn.AdaptiveAvgPool2d((1,input_dim))
        self.classifier = nn.Linear(input_dim, out_planes)

    def forward(self, x):
        shape = x.shape
        x = self.features(x)
        x = x.view(shape[0], shape[1] // self.in_planes,-1)
        x = self.self_attention(x)

        x = self.pool(x).squeeze(dim=1)
        x = self.classifier(x)
        return x

def GestureTransoformer (device,backbone,in_planes: int=3, n_classes: int=25,
                        pretrained: bool = False,layers_to_delete=2,input_dim=512,**kwargs):
    model = _GestureTransformer(device,backbone,in_planes,out_planes = n_classes,input_dim=input_dim,pretrained=pretrained,layers_to_delete=layers_to_delete,**kwargs)
    return model
