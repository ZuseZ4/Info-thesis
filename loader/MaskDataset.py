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
    
    def __init__(self, img_dir, mask_dir, image_ending=".png"):
        self.img_dir = img_dir
        self.masks_dir = mask_dir
        self.mask_subdirs = [f for f in os.scandir(self.masks_dir) if f.is_dir()]
        self.image_ending = image_ending
        self.surface_names = [f.name for f in self.mask_subdirs]
        self.paths = [ [mask.path for mask in os.scandir(class_dir.path)] for class_dir in self.mask_subdirs]
        self.examples_per_surface = [len(path) for path in self.paths]
        self.num_classes = len(self.surface_names)
        self.len = sum(self.examples_per_surface)

        # We might have classes without examples, strip them:
        for i in range(self.num_classes-1, -1, -1):
            if self.examples_per_surface[i] == 0:
                self.paths.pop(i)
                self.examples_per_surface.pop(i)
                self.surface_names.pop(i)
                self.mask_subdirs.pop(i)
        self.num_classes = len(self.surface_names)

        print("Initialized", self.len, "objects from", self.num_classes, "classes.")
        print(self.examples_per_surface)

    # Just for debug
    def get(self, i = None):
        if i == None:
            return self.__getitem__(random.randrange(0, self.__len__()))
        else:
            return self.__getitem__(i)

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
        primary_mask_path = self.paths[primary_class][index]
        sym_mask_path     = self.paths[primary_class][random.randrange(0, self.examples_per_surface[primary_class])]
        diff_mask_path    = self.paths[secondary_class][random.randrange(0, self.examples_per_surface[secondary_class])]
        # print(primary_mask_path + "\n" + sym_mask_path + "\n" + diff_mask_path)
        
        # Now we have to get the corresponding input images for these masks
        # We remove the image ening and add the ending which our masks will have
        # We remove _X enumerations which show up if we have multiple masks based of the same class for one image
        primary_name = os.path.basename(primary_mask_path).split(".")[0].split("_")[0] + self.image_ending
        sym_name     = os.path.basename(sym_mask_path).split(".")[0].split("_")[0] + self.image_ending
        diff_name    = os.path.basename(diff_mask_path).split(".")[0].split("_")[0] + self.image_ending

        primary_img_path = os.path.join(self.img_dir, primary_name)
        sym_img_path     = os.path.join(self.img_dir, sym_name    )
        diff_img_path    = os.path.join(self.img_dir, diff_name   )

        
        # Now we just have to replace all the paths by the actual images / masks 
        primary_mask = Image.open(primary_mask_path)
        sym_mask = Image.open(sym_mask_path)
        diff_mask = Image.open(diff_mask_path)
        
        # I told Blender to output RGB instead of RGBA, so that's just making sure.
        primary_img = Image.open(primary_img_path).convert("RGB")
        sym_img = Image.open(sym_img_path).convert("RGB")
        diff_img = Image.open(diff_img_path).convert("RGB")
        #print("mask mode: ", primary_mask.mode, " image mode: ", primary_img.mode) # 1, RGB

        factor = 1.5
        
        preprocess_normalize = transforms.Compose([
                                    transforms.Resize((int(384*factor),int(288*factor)), InterpolationMode.BILINEAR),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        preprocess           = transforms.Compose([
                                    transforms.Resize((int(384*factor),int(288*factor)), InterpolationMode.BILINEAR),
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
