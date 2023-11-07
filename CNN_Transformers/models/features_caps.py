import torch 
from torch import nn 
from torch.utils.data import DataLoader 
from torchvision import datasets 
from torchvision.transforms import ToTensor 
import torchvision
from torchsummary import summary
import torch.nn.functional as F
import math


def Resnet(pretrain=True,layers_to_unfreeze=1,layers_to_delete=2,in_planes=3):
    """
    param:
    pretrain: Define if we load a pretrained model from ImageNet
    layers_to_unfreeze: Define the number of layers that we want to train at the end of the Resnet
    layers_to_delete: Define the numbers of layers that we want to delete
    in_planes: Define the numbers of input channels of images (supported values: 1,2 or 3)

    return: The Resnet model
    """
    resnet = torchvision.models.resnet18(pretrained=pretrain)
    # Create a new model cause we don't want the pooling operation at the end and the classifier
    model = nn.Sequential()
    number_of_layers = len(list(resnet.children())) - layers_to_delete # In practice it remove the pooling operation and the classifier

    if number_of_layers<layers_to_unfreeze:
        layers_to_unfreeze = number_of_layers
    layers_to_freeze = number_of_layers - layers_to_unfreeze
    i=0
    for child in resnet.children():
        # For the first layers we create a new weight if in_planes is not 3 cause ResNet is pretrain on image with 3 channels there is no version for 1 channel
        if i==0 and in_planes<3:
            if i<layers_to_freeze: # Define if we freeze this layer or no
                for param in child.parameters():
                    param.requires_grad = False # Freeze the layers by passing requires_grad attribute to False
            w = child._parameters['weight'].data # Get the weight for 3 channels data
            child._modules['0'] = nn.Conv2d(in_planes, 64, kernel_size=3, padding=1) # Define the new conv layer
            if in_planes == 1:
                child._parameters['weight'].data = w.mean(dim=1, keepdim=True) # If the number of channels is 1 we made the mean of channels to set the new weight
            else:
                child._parameters['weight'].data = w[:, :-1] * 1.5

        if i<layers_to_freeze: # Define if we freeze this layer or no
            for param in child.parameters():
                param.requires_grad = False # Freeze the layers by passing requires_grad attribute to False
        if i<number_of_layers: # To define if we keep this layer or not
            model.append(child) 
        i+=1
    return model



class features_extraction(nn.Module):
    """
    param:
    conv_model: The convolution model used before capsules for the moment only ResNet is supported
    in_planes: Numbers of channels for the image
    output_caps: Numbers of capsule at the output of primary capsules
    primary_dims: Dimension of the primary capsules
    output_dims: Dimension of the dense caps
    n_classes: Number of Gesture to detect. In this part it correspond to the number of capsule at the output of dense caps
    n_routing: Number of iteration in the routing by agreement
    input_dim: Dimension of features after the resnet
    """
    def __init__(self,conv_model,in_planes: int):
        super().__init__()
        self.conv_model = conv_model
        self.in_planes = in_planes
        self.pooling = nn.AdaptiveAvgPool2d((1,1))
    def forward(self,input):
        shape = input.size()
        x = input.view(-1,self.in_planes,shape[-2],shape[-1])
        x = self.conv_model(x)
        x = self.pooling(x)
        return x
