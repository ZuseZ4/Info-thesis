import os
import glob
import numpy as np
from tqdm import tqdm
from PIL import Image

from loader.MaskDataset import MaskDataset

import torch
import torch.nn as nn
from torch.nn.functional import normalize
from torch import optim

import torchvision



class Res50(nn.Module):
    def __init__(self,  pretrained=True):
        super(Res50, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        #self.conv = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)
        self.maskLayer = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
    #def __apply_mask(self, image, mask):
    #    mask = mask.reshape(1,1,384,288)
    #    mask = self.conv(mask)
    #    return image + mask
            
    def forward(self, images, masks):
        prime_img, sym_img, diff_img = images
        prime_mask, sym_mask, diff_mask = masks

        y=MaskLayer(Mask)
        x = net.conv1(image)
        x = net.bn1(x)+y
        x = net.relu(x)
        x = net.maxpool(x)
        x = net.layer1(x)
        x = net.layer2(x)
        x = net.layer3(x)
        x = net.layer4(x)
        x = torch.mean(torch.mean(x, dim=2), dim=2)

        
        prime_in = self.__apply_mask(prime_img, prime_mask)
        diff_in  = self.__apply_mask(diff_img, diff_mask)
        sym_in   = self.__apply_mask(sym_img, sym_mask)
        
        prime_out = self.model(prime_in).reshape(1000)
        sym_out   = self.model(sym_in).reshape(1000)
        diff_out  = self.model(diff_in).reshape(1000)
        
        return (prime_out, sym_out, diff_out)
        
        

"""
net=torchvision.models.resnet50(pretrained=True)

y=MaskLayer(Mask)
x = net.conv1(image)
x = net.bn1(x)+y
x = net.relu(x)
x = net.maxpool(x)
x = net.layer1(x)
x = net.layer2(x)
x = net.layer3(x)
x = net.layer4(x)
x = torch.mean(torch.mean(x, dim=2), dim=2)

finalrsults=finalLinearLayer(x)
"""
# Make sure that the first and last added layer are part of the training / backprop path
       
        
        

training_dir = "/home/zuse/prog/CS_BA_Data/training"
m = MaskDataset(training_dir)
model = Res50()

prime, sym, diff = model.forward(*m.get())


prime = normalize(prime, dim=0)
sym   = normalize(sym, dim=0)
diff  = normalize(diff, dim=0)


t1 = torch.dot(prime, sym)
t2 = torch.dot(prime, diff)


output = torch.tensor((t1,t2), dtype=torch.float, requires_grad=True)
target = torch.tensor([1,0], dtype=torch.float)


criterion = nn.BCEWithLogitsLoss()
loss = criterion(output, target)
print("loss",loss, loss.shape)
loss.backward()
