import OpenEXR as exr
import numpy
import numpy as np
import os
import shutil
import glob
import sys
import Imath
from pathlib import Path
from PIL import Image as im
from PIL import ImageDraw

def read_exr_channel(filepath: os.path, channel):
    exrfile = exr.InputFile(filepath)
    raw_bytes = exrfile.channel(channel, Imath.PixelType(Imath.PixelType.FLOAT))
    depth_vector = numpy.frombuffer(raw_bytes, dtype=numpy.float32)
    height = exrfile.header()['displayWindow'].max.y + 1 - exrfile.header()['displayWindow'].min.y
    width = exrfile.header()['displayWindow'].max.x + 1 - exrfile.header()['displayWindow'].min.x
    depth_map = numpy.reshape(depth_vector, (height, width))
    return depth_map

# white=255, black=0
# closer objects have lower depth values
def create_and_store_mask(base_depth, img_depth, full_depth, path):
    f = lambda x: 255 if (x>=0)  else 0
    filter_diff = np.vectorize(f)
    attention = lambda x: 1 if (x>0)  else 0
    filter_attention = np.vectorize(attention)

    attention_area = filter_attention(base_depth - img_depth) # ones where the object is, zero elsewhere
        
    bw_img = filter_diff(full_depth-img_depth) * attention_area
    bw_img = bw_img.astype(np.uint8)
    
    print("bw-image: ", bw_img.shape, bw_img.dtype, type(bw_img))
    diff_img = im.fromarray(bw_img)
    diff_img = diff_img.convert('1')
    if not diff_img.getbbox():
        print("completely hidden")
        if os.path.isfile(path): # it's completely black. See https://stackoverflow.com/a/14041871/6153287
            os.remove(path) # usually we would overwrite it, but we return early. However we don't want to keep empty masks
        return
    diff_img.save(path)


def generate_masks(render_path):
    
    scene_folders = [f.path for f in os.scandir(render_path) if f.is_dir()]
    for scene_dir in scene_folders:

        # empty img for comparison
        base_img = os.path.join(scene_dir, "render_empty.exr")
        if not os.path.isfile(base_img):
            continue
        base_depth = read_exr_channel(base_img, 'Z')

        # complete img for comparison
        full_img_path = os.path.join(scene_dir, "render_full.exr")
        if not os.path.isfile(full_img_path):
            continue
        full_depth = read_exr_channel(full_img_path, 'Z')
        
        
        exr_files = glob.glob(os.path.join(scene_dir, "*.exr"))
        print(exr_files)
        for exr_file in exr_files:
            if exr_file.endswith("render_full.exr") or exr_file.endswith("render_empty.exr"):
                print("returning early")
                continue # We already have specific handling for that
            out_path = os.path.join(scene_dir, exr_file.split(".")[0] + "_mask.png")
            print(out_path)
            img_depth = read_exr_channel(exr_file, 'Z')
            create_and_store_mask(base_depth, img_depth, full_depth, out_path)

def exists_or_create(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def move_train_data(src,dst, pbr_dir):
    image_dir = os.path.join(dst, "img")
    mask_dir = os.path.join(dst, "masks")
    exists_or_create(dst)
    exists_or_create(image_dir)
    exists_or_create(mask_dir)
    
    pbr_materials = [f.name for f in os.scandir(pbr_dir) if f.is_dir()]
    for mat in pbr_materials:
        mat_dir = os.path.join(mask_dir, mat)
        if not os.path.isdir(mat_dir):
            os.mkdir(mat_dir)
    
    scene_folders = [f.path for f in os.scandir(src) if f.is_dir()]
    for scene_dir in scene_folders:
        full_img_path = os.path.join(scene_dir, "render_full.png")
        if not os.path.isfile(full_img_path):
            print("skipping folder, didn't found full img")
            continue # we then can't use the rest 
        dir_num = scene_dir.split(os.path.sep)[-1]
        full_img_dst = os.path.join(image_dir,  dir_num + ".png")
        os.rename(full_img_path, full_img_dst)
        
        masks = glob.glob(os.path.join(scene_dir, "*_mask.png"))
        for mask in masks:
            tmp = mask.split("_mask.png")[0].split(os.path.sep)[-1]
            pbr, num = tmp.rsplit("_",1)
            mask_dst = os.path.join(mask_dir, pbr, dir_num + "_" + num + ".png")
            os.rename(mask, mask_dst)
    

base_path   = "/home/zuse/prog/CS_BA_Data"
render_path = os.path.join(base_path, "render_output")
sample_path = os.path.join(base_path, "SamplePBR")
training_path = os.path.join(base_path, "training")

#generate_masks(render_path)
move_train_data(render_path, training_path, sample_path)



