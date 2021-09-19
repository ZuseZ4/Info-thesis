import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms, utils
from torchvision.transforms import InterpolationMode
from PIL import Image
import os
import sys
import glob
import random

# expected layout:
# base_dir/training/img/<num>.png
# base_dir/training/masks/<surface_name>/<img_num>_<counter_per_img_and_surface_combination>.png

# Core idea:
# We pick two classes.
# From the first one we pick two masks (and their linked input images)
# From the second class we pick one mask (and the related input image)
# Now we have one equal and one different image
class MaskDataset(Dataset):
    
    def __init__(self, training_path):
        self.img_dir = os.path.join(training_path, "img")
        self.masks_dir = os.path.join(training_path, "masks")
        self.mask_subdirs = [f for f in os.scandir(self.masks_dir) if f.is_dir()]
        self.surface_names = [f.name for f in self.mask_subdirs]
        self.paths = [ [mask.path for mask in os.scandir(class_dir.path)] for class_dir in self.mask_subdirs]
        self.examples_per_surface = [len(path) for path in self.paths]
        self.num_classes = len(self.surface_names)
        self.len = sum(self.examples_per_surface)
        print("Initialized", self.len, "objects from", self.num_classes, "classes.")

    # Just for debug
    def get(self):
        return self.__getitem__(random.randrange(0, self.__len__()))

    def __getitem__(self, index):

        # First we need to get our primary and secondary class
        primary_class = 0
        while index >= self.examples_per_surface[primary_class]:
            primary_class += 1
            index -= self.examples_per_surface[primary_class]
        secondary_class = random.randrange(0,self.num_classes-1)
        if secondary_class >= primary_class: # we account for not picking our primary class twice
            secondary_class += 1
            
        # Now we have to pick two masks / images from our primary class, one from the secondary
        # We only pick our first primary img based on the index.
        # The other two masks / images are picked randomly from their selected class
        primary_mask = self.paths[primary_class][index]
        sym_mask = self.paths[primary_class][random.randrange(0, self.examples_per_surface[primary_class])]
        diff_mask = self.paths[secondary_class][random.randrange(0, self.examples_per_surface[secondary_class])]
        #print(primary_mask + "\n" + sym_mask + "\n" + diff_mask)
        
        # Now we have to get the corresponding input images for these masks
        primary_num = primary_mask.split(os.path.sep)[-1].split("_")[0] + ".png"
        sym_num = sym_mask.split(os.path.sep)[-1].split("_")[0] + ".png"
        diff_num = diff_mask.split(os.path.sep)[-1].split("_")[0] + ".png"
        primary_img = os.path.join(self.img_dir,primary_num)
        sym_img = os.path.join(self.img_dir,sym_num)
        diff_img = os.path.join(self.img_dir,diff_num)
        #print(primary_img + "\n" + sym_img + "\n" + diff_img)  
        
        # Now we just have to replace all the paths by the actual images / masks 
        primary_mask = Image.open(primary_mask)
        sym_mask = Image.open(sym_mask)
        diff_mask = Image.open(diff_mask)
        
        # TODO: Images should be generated as RGB's by blender, not RGBA. Fix it there
        primary_img = Image.open(primary_img).convert("RGB")
        sym_img = Image.open(sym_img).convert("RGB")
        diff_img = Image.open(diff_img).convert("RGB")
        #print("mask mode: ", primary_mask.mode, " image mode: ", primary_img.mode) # 1, RGB

        
        preprocess_normalize = transforms.Compose([
                                    transforms.Resize((384,288), InterpolationMode.BILINEAR),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        preprocess           = transforms.Compose([
                                    transforms.Resize((384,288), InterpolationMode.BILINEAR),
                                    transforms.ToTensor()])

        primary_mask = preprocess(primary_mask)
        sym_mask = preprocess(sym_mask)
        diff_mask = preprocess(diff_mask)
        
        primary_img = preprocess_normalize(primary_img)
        sym_img = preprocess_normalize(sym_img)
        diff_img = preprocess_normalize(diff_img)
 
        images = (primary_img, sym_img, diff_img)
        masks = (primary_mask, sym_mask, diff_mask)
        return images, masks 
            
    def __len__(self):
        return self.len
