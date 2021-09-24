import os
import glob
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from loader.MaskDataset import MaskDataset

from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from torch.nn.functional import normalize
from torch import optim

import torchvision



class Res50(nn.Module):
    def __init__(self,  pretrained=True):
        super(Res50, self).__init__()
        self.net = torchvision.models.resnet50(pretrained=True)
        self.maskLayer = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.finalLinearLayer = torch.nn.Linear(2048, 500) # find out first param
        

    def forward_single(self, image, mask):
        y = self.maskLayer(mask)

        x = self.net.conv1(image)
        x = self.net.bn1(x)+y
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        x = torch.mean(torch.mean(x, dim=2), dim=2)

        finalresults=self.finalLinearLayer(x)
        
        # Make sure that the first and last added layer are part of the training / backprop path
        return finalresults
    
    
    def forward(self, images, masks):
        prime_img, sym_img, diff_img = images
        prime_mask, sym_mask, diff_mask = masks

        
        prime_out = self.forward_single(prime_img, prime_mask)
        diff_out  = self.forward_single(diff_img, diff_mask)
        sym_out   = self.forward_single(sym_img, sym_mask)
        
        return (prime_out, sym_out, diff_out)
        
        

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
        

# OpenSurface Dataset
class Config():
    train_batch_size = 4
    train_number_epochs = 100
    img_dir = "/home/zuse/prog/CS_BA_Data/OpenSurfaces/OpenSurfaceMaterialsSmall/Images"
    train_mask_dir = "/home/zuse/prog/CS_BA_Data/OpenSurfaces/OpenSurfaceMaterialsSmall/TrainLabelsSplit"
    test_mask_dir  = "/home/zuse/prog/CS_BA_Data/OpenSurfaces/OpenSurfaceMaterialsSmall/TestLabelsSplit"
    weights_folder = "/home/zuse/prog/CS_BA_Data/nn_savedir"
    image_ending = ".jpg"

def get_weights_path(num):
    return os.path.join(cfg.weights_folder, str(num)+".pt")

def test(cfg, weight):
    test_dataset = MaskDataset(cfg.img_dir, cfg.test_mask_dir, cfg.image_ending)
    bs = 2
    test_dataloader = DataLoader(test_dataset, bs, shuffle=True)
    nn_path = get_weights_path(weight)

    net = Res50().cuda()
    net.load_state_dict(torch.load(nn_path))
    net.eval()
    correct = 0
    n = 2000
    for i, data in enumerate(test_dataloader, 0):
        if i > n:
            break
        if i % 100 = 0:
            print(i, correct, i*bs-correct)
        (im0, im1, im2), (mask0, mask1, mask2) = data
        images, masks = (im0.cuda(), im1.cuda(), im2.cuda()), (mask0.cuda(), mask1.cuda(), mask2.cuda())
        (o1, o2, o3) = net(images, masks)
        (o1, o2, o3) = (o1.reshape(bs,1,500), o2.reshape(bs,1,500), o3.reshape(bs,1,500))
        d_sym  = torch.cdist(o1, o2, p=2).reshape(bs,1)
        d_other = torch.cdist(o1, o3, p=2).reshape(bs,1)
        for i in range(bs):
            diff = d_sym[i] - d_other[i]
            # print("{}\t sym: {}\t other: {}\t diff {}\t correct: {}".format(i, round(d_sym,2), round(d_other,2), round(diff,2), diff<0))
            if diff < 0:
                correct += 1
        #print("sym:",d_sym, "\tother:",d_diff, "\tdiff:", d_diff-d_sym )
    print("correct:", correct, "false:", n*bs-correct)

# Generated Dataset
# img_dir = "/home/zuse/prog/CS_BA_Data/training/img"
# mask_dir = "/home/zuse/prog/CS_BA_Data/training/mask"
# image_ending = ".png"

cfg = Config()

test(cfg, 10000)

train_dataset = MaskDataset(cfg.img_dir, cfg.train_mask_dir, cfg.image_ending)
train_dataloader = DataLoader(train_dataset, cfg.train_batch_size, shuffle=True)


net = Res50().cuda()
nn_path = get_weights_path(5000)
net.load_state_dict(torch.load(nn_path))
optimizer = optim.Adam(net.parameters(),lr = 0.0005 )
# criterion = ContrastiveLoss()
triplet_criterion = torch.nn.TripletMarginLoss()


counter = []
loss_history = [] 
iteration_number= 0
dataset_len = len(train_dataset)

for epoch in range(0,Config.train_number_epochs):
    for i, data in enumerate(train_dataloader,0):
        #(im0, im1, im2), (mask0, mask1, mask2) = train_dataloader.get(i)
        (im0, im1, im2), (mask0, mask1, mask2) = data
        images, masks = (im0.cuda(), im1.cuda(), im2.cuda()), (mask0.cuda(), mask1.cuda(), mask2.cuda()) 
        (o1, o2, o3) = net(images, masks)
        optimizer.zero_grad()
        # print(o1.shape, o2.shape, o3.shape)
        loss_triplet = triplet_criterion(o1,o2,o3)
        loss_triplet.backward()
        optimizer.step()
        if i %10 == 0 :
            # loss = loss_triplet.data[0] # for multiple dims
            loss = loss_triplet.item()
            print("Epoch number {}, {}/{}\n Current loss {}\n".format(epoch,i,dataset_len,loss))
            iteration_number +=10
            counter.append(iteration_number)
            loss_history.append(loss)
        if i %2000 == 0:
            save_path = os.path.join(cfg.weights_folder, str(i)+".pt")
            torch.save(net.state_dict(),save_path)
    test(cfg, int(40000 / bs))
plt.plot(iteration,loss)
plt.show()

# OpenSurface Dataset
class Config():
    train_batch_size = 4
    train_number_epochs = 100
    img_dir = "/home/zuse/prog/CS_BA_Data/OpenSurfaces/OpenSurfaceMaterialsSmall/Images"
    train_mask_dir = "/home/zuse/prog/CS_BA_Data/OpenSurfaces/OpenSurfaceMaterialsSmall/TrainLabelsSplit"
    test_mask_dir  = "/home/zuse/prog/CS_BA_Data/OpenSurfaces/OpenSurfaceMaterialsSmall/TestLabelsSplit"
    weights_folder = "/home/zuse/prog/CS_BA_Data/nn_savedir"
    image_ending = ".jpg"

def get_weights_path(num):
    return os.path.join(cfg.weights_folder, str(num)+".pt")

def test(cfg, weight):
    test_dataset = MaskDataset(cfg.img_dir, cfg.test_mask_dir, cfg.image_ending)
    bs = 2
    test_dataloader = DataLoader(test_dataset, bs, shuffle=True)
    nn_path = get_weights_path(weight)

    net = Res50().cuda()
    net.load_state_dict(torch.load(nn_path))
    net.eval()
    correct = 0
    n = 2000
    for i, data in enumerate(test_dataloader, 0):
        if i > n:
            break
        if i % 100 = 0:
            print(i, correct, i*bs-correct)
        (im0, im1, im2), (mask0, mask1, mask2) = data
        images, masks = (im0.cuda(), im1.cuda(), im2.cuda()), (mask0.cuda(), mask1.cuda(), mask2.cuda())
        (o1, o2, o3) = net(images, masks)
        (o1, o2, o3) = (o1.reshape(bs,1,500), o2.reshape(bs,1,500), o3.reshape(bs,1,500))
        d_sym  = torch.cdist(o1, o2, p=2).reshape(bs,1)
        d_other = torch.cdist(o1, o3, p=2).reshape(bs,1)
        for i in range(bs):
            diff = d_sym[i] - d_other[i]
            # print("{}\t sym: {}\t other: {}\t diff {}\t correct: {}".format(i, round(d_sym,2), round(d_other,2), round(diff,2), diff<0))
            if diff < 0:
                correct += 1
        #print("sym:",d_sym, "\tother:",d_diff, "\tdiff:", d_diff-d_sym )
    print("correct:", correct, "false:", n*bs-correct)

# Generated Dataset
# img_dir = "/home/zuse/prog/CS_BA_Data/training/img"
# mask_dir = "/home/zuse/prog/CS_BA_Data/training/mask"
# image_ending = ".png"

cfg = Config()

test(cfg, 10000)

train_dataset = MaskDataset(cfg.img_dir, cfg.train_mask_dir, cfg.image_ending)
train_dataloader = DataLoader(train_dataset, cfg.train_batch_size, shuffle=True)


net = Res50().cuda()
nn_path = get_weights_path(5000)
net.load_state_dict(torch.load(nn_path))
optimizer = optim.Adam(net.parameters(),lr = 0.0005 )
# criterion = ContrastiveLoss()
triplet_criterion = torch.nn.TripletMarginLoss()


counter = []
loss_history = [] 
iteration_number= 0
dataset_len = len(train_dataset)

for epoch in range(0,Config.train_number_epochs):
    for i, data in enumerate(train_dataloader,0):
        #(im0, im1, im2), (mask0, mask1, mask2) = train_dataloader.get(i)
        (im0, im1, im2), (mask0, mask1, mask2) = data
        images, masks = (im0.cuda(), im1.cuda(), im2.cuda()), (mask0.cuda(), mask1.cuda(), mask2.cuda()) 
        (o1, o2, o3) = net(images, masks)
        optimizer.zero_grad()
        # print(o1.shape, o2.shape, o3.shape)
        loss_triplet = triplet_criterion(o1,o2,o3)
        loss_triplet.backward()
        optimizer.step()
        if i %10 == 0 :
            # loss = loss_triplet.data[0] # for multiple dims
            loss = loss_triplet.item()
            print("Epoch number {}, {}/{}\n Current loss {}\n".format(epoch,i,dataset_len,loss))
            iteration_number +=10
            counter.append(iteration_number)
            loss_history.append(loss)
        if i %2000 == 0:
            save_path = os.path.join(cfg.weights_folder, str(i)+".pt")
            torch.save(net.state_dict(),save_path)
    test(cfg, int(40000 / bs))



"""
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
"""

