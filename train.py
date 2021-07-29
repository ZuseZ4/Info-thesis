
from torchvision import models

deeplab = models.segmentation.deeplabv3_resnet50(pretrained=0, progress=1, num_classes=7) # 6+bg

class MatSegModel(nn.Module):
    def __init__(self):
        super(MatSegModel,self).__init__()
        self.dl = deeplab
        
    def forward(self, x):
        y = self.dl(x)['out']
        return y

class SegDataset(Dataset):
    
    def __init__(self, training_path):
        image_path = os.path.join(training_path, "*")
        self.trainingDirs = glob.glob(training_dir)
        self.trainingDirs.sort()
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
        
        empty_mask = torch.zeros((384,288), dtype=torch.bool)

        masks = []
        for class_name in self.classes:
            mask_path = os.path.join(training_path, index, class_name)
            if not os.path.isfile(mask_path):
                masks.append(empty_mask)
            else:
                yimg = Image.open(mask_path).convert('L')
                y1 = trftensor(trfresize(yimg))
                y1 = y1.type(torch.BoolTensor)
                masks.append(y1)


        y = torch.cat(masks, dim=0)
        
        return X, y
            
    def __len__(self):
        return len(self.trainingDirs)

retval = training_loop(3, 
                       optimizer, 
                       lr_scheduler, 
                       model, 
                       loss_fn, 
                       trainLoader, 
                       valLoader)
