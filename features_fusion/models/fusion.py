import torch
import torch.nn as nn
from torchsummary import summary
from models.model_utilizer import ModuleUtilizer
from models.temporal import GestureTransoformer,_GestureTransformer

def remove_Linear(net):
    i = 0
    for child in net.children():
        for param in child.parameters():
            param.requires_grad = False
    new_net = nn.Sequential()
    new_net.append(net.features)
    new_net.append(net.self_attention)
    new_net.append(net.pool)
    return new_net

class Fusion(nn.Module):
    def __init__(self,device,backbone,data_types,model_utility,n_classes: int=12,
                 pretrained: bool = False,layers_to_unfreeze=1,n_caps=8,caps_dims=16,output_dims=32,
                 **kwargs):
        super(Fusion, self).__init__()
        self.model_utility = model_utility
        self.device = device
        self.data_types = data_types
        self.Caps_and_Transformers = nn.ModuleList()
        i = 0
        in_planes = 0
        self.in_planes = 0
        while i<len(self.data_types):
            data_type = self.data_types[i].lower()
            if data_type in ["depth", "ir"]:
                in_planes = 1
                self.in_planes+=1
            else:
                in_planes = 3
                self.in_planes+=3
            net = _GestureTransformer(device,backbone,in_planes,out_planes = n_classes,caps_dims=caps_dims,output_dims=output_dims,pretrained=pretrained,**kwargs)
            net,iters,epoch,optim_dict = self.model_utility.load_net_modality(net,data_type)
            net = remove_Linear(net)
            self.Caps_and_Transformers.append(net)
            i+=1
        if n_classes == 28: # For SHREC28 we reduce to 14.
            n_classes=14
        self.fc1 = nn.Linear(len(self.data_types)*n_classes*output_dims,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,n_classes)


    def forward(self, x):
        i = 0
        chanel_prec = 0
        x_cat = None
        x = x.view(-1,x.shape[1]//self.in_planes,self.in_planes,x.shape[-2],x.shape[-1])
        while i<len(self.data_types):
            if self.data_types[i] in ["depth", "ir"]:
                x_current = x[:,:,chanel_prec:chanel_prec+1]
                chanel_prec+=1
            else:
                x_current = x[:,:,chanel_prec:chanel_prec+3]
                chanel_prec+=3
            print(x_current.shape)
            x_current = self.Caps_and_Transformers[i](x_current)
            if i==0:
                x_cat = x_current
            else:
                x_cat = torch.cat((x_cat,x_current),dim=-1)
            i+=1
        x_cat = x_cat.squeeze(dim=1)
        x = self.fc1(x_cat)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

