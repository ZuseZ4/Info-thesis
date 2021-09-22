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
        self.net = torchvision.models.resnet50(pretrained=True)
        #self.conv = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)
        self.maskLayer = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.finalLinearLayer = torch.nn.Linear(2048, 500) # find out first param
        

    def forward_single(self, image, mask):
        
        y=self.maskLayer(mask.reshape([1, 1, 384, 288]))
        x = self.net.conv1(image.reshape([1, 3, 384, 288]))
        x = self.net.bn1(x)+y
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        x = torch.mean(torch.mean(x, dim=2), dim=2)

        finalresults=self.finalLinearLayer(x)
        print(finalresults.shape)
        
        return finalresults.reshape(500)
    
    
    def forward(self, images, masks):
        prime_img, sym_img, diff_img = images
        prime_mask, sym_mask, diff_mask = masks

        
        prime_out = self.forward_single(prime_img, prime_mask)
        diff_out  = self.forward_single(diff_img, diff_mask)
        sym_out   = self.forward_single(sym_img, sym_mask)
        
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
 
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive
        
        
# Generated Dataset
# img_dir = "/home/zuse/prog/CS_BA_Data/training/img"
# mask_dir = "/home/zuse/prog/CS_BA_Data/training/mask"
# image_ending = ".png"

# OpenSurface Dataset
img_dir = "/home/zuse/prog/CS_BA_Data/OpenSurfaces/OpenSurfaceMaterialsSmall/Images"
mask_dir = "/home/zuse/prog/CS_BA_Data/OpenSurfaces/OpenSurfaceMaterialsSmall/TrainLabelsSplit"
image_ending = ".jpg"

m = MaskDataset(img_dir, mask_dir, image_ending)
net = Res50().cuda()
optimizer = optim.Adam(net.parameters(),lr = 0.0005 )
criterion = ContrastiveLoss()

prime, sym, diff = net.forward(*m.get())


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
