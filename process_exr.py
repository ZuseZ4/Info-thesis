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
    diff_img.convert('1').save(path)


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
                continue # We already have specific handling for that
            out_path = os.path.join(scene_dir, exr_file.split(".")[0] + "_mask.png")
            print(out_path)
            img_depth = read_exr_channel(exr_file, 'Z')
            create_and_store_mask(base_depth, img_depth, full_depth, out_path)


base_path   = "/home/zuse/prog/CS_BA_Data"
render_path = os.path.join(base_path, "render_output")
sample_path = os.path.join(base_path, "SamplePBR")

generate_masks(render_path)
