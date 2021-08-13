import os
import glob
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms, utils

deeplab = models.segmentation.deeplabv3_resnet50(pretrained=0, progress=1, num_classes=6) # 5+bg

class MatSegModel(nn.Module):
    def __init__(self):
        super(MatSegModel,self).__init__()
        self.dl = deeplab
        
    def forward(self, x):
        y = self.dl(x)['out']
        return y

class SegDataset(Dataset):
    
    def __init__(self, training_path):
        self.training_path = training_path
        image_path = os.path.join(training_path, "*")
        self.trainingDirs = glob.glob(image_path)
        self.trainingDirs.sort()
        print(len(self.trainingDirs))
        self.classes = ["bg.png", "Asphalt001.png", "Bark006.png", "Bricks031.png", "Cardboard004.png", "Chainmail002.png"]

    def __getitem__(self, index):
        
        preprocess = transforms.Compose([
                                    transforms.Resize((384,288), 2),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        img_path = os.path.join(self.trainingDirs[index], "input.png")
        X = Image.open(img_path).convert('RGB')
        X = preprocess(X)
        
        trfresize = transforms.Resize((384, 288), 2)
        trftensor = transforms.ToTensor()
        
        masks = []
        for class_name in self.classes:
            mask_path = os.path.join(self.trainingDirs[index], class_name)
            if not os.path.isfile(mask_path):
                yimg = Image.new('L', size=(1920,1080))
                #print("not found", mask_path)
            else:
                yimg = Image.open(mask_path).convert('L')

            y1 = trftensor(trfresize(yimg))
            y1 = y1.type(torch.BoolTensor)
            masks.append(y1)

        y = torch.cat(masks, dim=0)
        
        return X, y
            
    def __len__(self):
        return len(self.trainingDirs)

def meanIOU(target, predicted):
    if target.shape != predicted.shape:
        print("target has dimension", target.shape, ", predicted values have shape", predicted.shape)
        return
        
    if target.dim() != 4:
        print("target has dim", target.dim(), ", Must be 4.")
        return
    
    iousum = 0
    for i in range(target.shape[0]):
        target_arr = target[i, :, :, :].clone().detach().cpu().numpy().argmax(0)
        predicted_arr = predicted[i, :, :, :].clone().detach().cpu().numpy().argmax(0)
        
        intersection = np.logical_and(target_arr, predicted_arr).sum()
        union = np.logical_or(target_arr, predicted_arr).sum()
        if union == 0:
            iou_score = 0
        else :
            iou_score = intersection / union
        iousum +=iou_score
        
    miou = iousum/target.shape[0]
    return miou

def pixelAcc(target, predicted):    
    if target.shape != predicted.shape:
        print("target has dimension", target.shape, ", predicted values have shape", predicted.shape)
        return
        
    if target.dim() != 4:
        print("target has dim", target.dim(), ", Must be 4.")
        return
    
    accsum=0
    for i in range(target.shape[0]):
        target_arr = target[i, :, :, :].clone().detach().cpu().numpy().argmax(0)
        predicted_arr = predicted[i, :, :, :].clone().detach().cpu().numpy().argmax(0)
        
        same = (target_arr == predicted_arr).sum()
        a, b = target_arr.shape
        total = a*b
        accsum += same/total
    
    pixelAccuracy = accsum/target.shape[0]        
    return pixelAccuracy

def training_loop(n_epochs, optimizer, lr_scheduler, model, loss_fn, train_loader, val_loader, lastCkptPath = None):
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"
    dev = "cuda:0"
    device = torch.device(dev)
    
    tr_loss_arr = []
    val_loss_arr = []
    meanioutrain = []
    pixelacctrain = []
    meanioutest = []
    pixelacctest = []
    prevEpoch = 0
    
    if lastCkptPath != None :
        checkpoint = torch.load(lastCkptPath)
        prevEpoch = checkpoint['epoch'].model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.stainte.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
                    tr_loss_arr =  checkpoint['Training Loss']
        val_loss_arr =  checkpoint['Validation Loss']
        meanioutrain =  checkpoint['MeanIOU train']
        pixelacctrain =  checkpoint['PixelAcc train']
        meanioutest =  checkpoint['MeanIOU test']
        pixelacctest =  checkpoint['PixelAcc test']
        print("loaded model, ", checkpoint['description'], "at epoch", prevEpoch)
    model.to(device)
    
    for epoch in range(0, n_epochs):
        train_loss = 0.0
        pixelacc = 0
        meaniou = 0
        
        pbar = tqdm(train_loader, total = len(train_loader))
        #pbar = tqdm(train_loader, total = train_len)
        for X, y in pbar:
            torch.cuda.empty_cache()
            model.train()
            X = X.to(device).float()
            y = y.to(device).float()
            ypred = model(X)
            loss = loss_fn(ypred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            tr_loss_arr.append(loss.item())
            meanioutrain.append(meanIOU(y, ypred))
            pixelacctrain.append(pixelAcc(y, ypred))
            pbar.set_postfix({'Epoch':epoch+1+prevEpoch, 
                              'Training Loss': np.mean(tr_loss_arr),
                              'Mean IOU': np.mean(meanioutrain),
                              'Pixel Acc': np.mean(pixelacctrain)
                             })
            
        with torch.no_grad():
            
            val_loss = 0
            pbar = tqdm(val_loader, total = len(val_loader))
            #pbar = tqdm(val_loader, total = val_len)
            for X, y in pbar:
                torch.cuda.empty_cache()
                X = X.to(device).float()
                y = y.to(device).float()
                model.eval()
                ypred = model(X)
                
                val_loss_arr.append(loss_fn(ypred, y).item())
                pixelacctest.append(pixelAcc(y, ypred))
                meanioutest.append(meanIOU(y, ypred))
                
                pbar.set_postfix({'Epoch':epoch+1+prevEpoch, 
                                  'Validation Loss': np.mean(val_loss_arr),
                                  'Mean IOU': np.mean(meanioutest),
                                  'Pixel Acc': np.mean(pixelacctest)
                                 })
        
        
        
        checkpoint = {
            'epoch':epoch+1+prevEpoch,
            'description':"add your description",
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'Training Loss': tr_loss_arr,
            'Validation Loss':val_loss_arr,
            'MeanIOU train':meanioutrain, 
            'PixelAcc train':pixelacctrain, 
            'MeanIOU test':meanioutest, 
            'PixelAcc test':pixelacctest
        }
        torch.save(checkpoint, 'checkpoints/checkpointhandseg'+str(epoch+1+prevEpoch)+'.pt')
        lr_scheduler.step()
        
    return tr_loss_arr, val_loss_arr, meanioutrain, pixelacctrain, meanioutest, pixelacctest

megaDataset = SegDataset('/home/zuse/prog/CS_BA_Data/training')
print(megaDataset)
#print(len(megaDataset))

#TTR is Train Test Ratio
def trainTestSplit(dataset, TTR):
    trainDataset = torch.utils.data.Subset(dataset, range(0, int(TTR * len(dataset))))
    valDataset = torch.utils.data.Subset(dataset, range(int(TTR*len(dataset)), len(dataset)))
    return trainDataset, valDataset
  
batchSize = 8
trainDataset, valDataset = trainTestSplit(megaDataset, 0.9)
print("dataset lengths", len(trainDataset), len(valDataset))
trainLoader = DataLoader(trainDataset, batch_size = batchSize, shuffle=True, drop_last=True)
valLoader = DataLoader(valDataset, batch_size = batchSize, shuffle=True, drop_last=True)

model = MatSegModel()
optimizer = optim.Adam(model.parameters(), lr=0.00005)
loss_fn = nn.BCEWithLogitsLoss ()
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.8)


retval = training_loop(3, 
                       optimizer, 
                       lr_scheduler, 
                       model, 
                       loss_fn, 
                       trainLoader, 
                       valLoader, 
                       None)
