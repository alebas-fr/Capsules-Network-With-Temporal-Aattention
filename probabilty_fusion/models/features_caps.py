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

def efficient_net(pretrain=True,layers_to_unfreeze=1,layers_to_delete=2,in_planes=3):
    """
    param:
    pretrain: Define if we load a pretrained model from ImageNet
    layers_to_unfreeze: Define the number of layers that we want to train at the end of the Resnet
    layers_to_delete: Define the numbers of layers that we want to delete 
    in_planes: Define the numbers of input channels of images (supported values: 1,2 or 3)
    return: The efficient_net model
    """
    efficient_net = torchvision.models.efficientnet_b3(pretrained=pretrain)
    efficient_net = efficient_net.features
    # Create a new model cause we don't want the pooling operation at the end and the classifier
    model = nn.Sequential()
    number_of_layers = len(list(efficient_net.children())) # In practice it remove the pooling operation and the classifier

    if number_of_layers<layers_to_unfreeze:
        layers_to_unfreeze = number_of_layers
    layers_to_freeze = number_of_layers - layers_to_unfreeze
    i=0
    for child in efficient_net.children():
        # For the first layers we create a new weight if in_planes is not 3 cause ResNet is pretrain on image with 3 channels there is no version for 1 channel
        if i==0 and in_planes<3:
            if i<layers_to_freeze: # Define if we freeze this layer or no
                for param in child.parameters():
                    param.requires_grad = False # Freeze the layers by passing requires_grad attribute to False
            j = 0 
            for c_child in child.children():
                if j==0:
                    w = c_child._parameters['weight'].data
                    c_child._modules['0'] = nn.Conv2d(in_planes,40,3,stride=2,bias=False)
                    c_child._parameters['weight'].data = w.mean(dim=1, keepdim=True)
                j+=1

        if i<layers_to_freeze: # Define if we freeze this layer or no
            for param in child.parameters():
                param.requires_grad = False # Freeze the layers by passing requires_grad attribute to False
        if i<number_of_layers: # To define if we keep this layer or not
            model.append(child) 
        i+=1
    return model

# Squash function
def squash(x):
    """ This function computes the squash of a vector
    param:
    x: The vector to squash
    return: The squashed vector
    """
    lengths2 = x.pow(2).sum(dim=2) 
    lengths = lengths2.sqrt() 
    x = x * (lengths2 / (1 + lengths2) / lengths).view(x.size(0), x.size(1), 1)
    return x


class PrimaryCapsLayer(nn.Module):
    """
    param: 
    input_channels: Number of channels that we get of previous conv layer
    output_caps: Number of caps that we wants at the output
    output_dim: Dimension of the caps
    kernel_size: Kernel for the conv layer
    stride: stride for the conv layer
    conv: The convolution layer that we apply to 
    """
    def __init__(self, input_channels, output_caps, output_dim, kernel_size, stride):
        super(PrimaryCapsLayer, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_caps * output_dim, kernel_size=kernel_size, stride=stride)
        self.input_channels = input_channels
        self.output_caps = output_caps
        self.output_dim = output_dim

    def forward(self, input):
        """
        input is the convolution layer from previous layer
        """
        out = self.conv(input)
        N, C, H, W = out.size() # Get the param for caps
        """
        N is the number of capsule
        C is the size of vector inside capsules
        H is the height
        W is the width
        """
        
        # Resize the tensor in order to have the vectors in order to have differents capsules with the good number of vector and the good size
        out = out.view(N, self.output_caps, self.output_dim, H, W)
        
        # output must have shape N x OUT_CAPS x OUT_DIM

        out = out.permute(0, 1, 3, 4, 2).contiguous() 
        # Remplace the width by output dims in order to flatten the vectors in only one dimension of shape (W*H*output_caps).
        # Set the shape of return tensor to (N,W*H*output_caps,output_dims)
        out = out.view(out.size(0), -1, out.size(4)) 
        out = squash(out) #Apply the squash function
        return out

class AgreementRouting(nn.Module):
    """
    param:
    n_iterations: Number of iteration
    input_caps: Inputs capsules
    output_caps: Outputs capsules
    b: Weight of routing 
    """
    def __init__(self, input_caps, output_caps, n_iterations):
        super(AgreementRouting, self).__init__()
        self.n_iterations = n_iterations
        self.b = nn.Parameter(torch.zeros((input_caps, output_caps)))

    def forward(self, u_predict):
        """
        u_predict: vector predict by Dense Caps
        """
        batch_size, input_caps, output_caps, output_dim = u_predict.size() 
        c = F.softmax(self.b,dim=1)
        s = (c.unsqueeze(2) * u_predict).sum(dim=1)
        v = squash(s)

        if self.n_iterations > 0:
            b_batch = self.b.expand((batch_size, input_caps, output_caps))
            for r in range(self.n_iterations):
                v = v.unsqueeze(1)
                b_batch = b_batch + (u_predict * v).sum(-1)
                c = F.softmax(b_batch.view(-1, output_caps),dim=1).view(-1, input_caps, output_caps, 1)
                s = (c * u_predict).sum(dim=1)
                v = squash(s)

        return v

class CapsLayer(nn.Module):
    """
    input_caps: Number of capsules from previous layer
    input_dim: Dimension of vector from previous layer
    output_caps: Number of capsule that we have at the end of the layer
    output_dim: Dimension of vector of each capsule at the end of the layer
    routing_module: Routing Class
    weights: Routing Weights 
    """
    def __init__(self, input_caps, input_dim, output_caps, output_dim, routing_module):
        super(CapsLayer, self).__init__()
        self.input_dim = input_dim
        self.input_caps = input_caps
        self.output_dim = output_dim
        self.output_caps = output_caps
        self.weights = nn.Parameter(torch.Tensor(input_caps, input_dim, output_caps * output_dim))
        self.routing_module = routing_module
        self.reset_parameters()

    def reset_parameters(self):
        """
        Set the weight with random values at the begining
        """
        stdv = 1. / math.sqrt(self.input_caps)
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, caps_output):
        """
        caps_output : Capsules output from previous layers
        """
        # Add a dimension to compute matmul with weight
        caps_output = caps_output.unsqueeze(2)
        u_predict = caps_output.matmul(self.weights)
        # Resize the u_predict for routing
        u_predict = u_predict.view(u_predict.size(0), self.input_caps, self.output_caps, self.output_dim)
        v = self.routing_module(u_predict)
        return v

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
    def __init__(self,conv_model,in_planes: int,output_caps = 8,primary_dims=16,output_dims = 32,n_classes=12,n_routing = 3,input_dim=512):
        super().__init__()
        self.conv_model = conv_model
        self.primary_caps = PrimaryCapsLayer(input_dim,output_caps,primary_dims,3,1)
        self.in_planes = in_planes
        self.n_prim_caps = 4*4*output_caps # 4*4 correspond to the widht and height at the output of conv operation in primary caps
        self.routing = AgreementRouting(self.n_prim_caps,n_classes,n_routing)
        self.dense_caps = CapsLayer(self.n_prim_caps,primary_dims,n_classes,output_dims,self.routing)
    def forward(self,input):
        shape = input.size()
        x = input.view(-1,self.in_planes,shape[-2],shape[-1])
        x = self.conv_model(x)
        x = self.primary_caps(x)
        x = self.dense_caps(x)
        x = x.view(shape[0],shape[1],-1)
        return x
