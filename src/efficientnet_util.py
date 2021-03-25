
from efficientnet_pytorch import EfficientNet
import torch
from torch import nn


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)
    elif type(m) == nn.BatchNorm1d:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def get_EfficientNet(name:str="efficientnet-b0", pretrained:bool=True, n_class:int=None, dropout_p:float=0.5):
    assert n_class != None, "Please specify the number of output classes `n_class`"
    if pretrained == True:
        print(f"Getting pretrained {name}")
        m = EfficientNet.from_pretrained(name)
    else:
        print(f"Getting random initialized {name}")
        m = EfficientNet.from_name(name)
    
    n_in = m._fc.in_features
    m._fc = nn.Sequential(
        nn.Dropout(p=dropout_p), 
        nn.Linear(n_in, n_class))
    m._fc.apply(init_weights)
    return m
