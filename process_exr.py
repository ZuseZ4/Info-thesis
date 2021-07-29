import OpenEXR as exr
import numpy
import numpy as np
import os
import shutil
import glob
import sys
import Imath
from pathlib import Path
import pylab # for colors
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

def store_array_as_img(img, path, inv):
    if inv:
        f = lambda x: 0 if (x>0)  else 255
    else:
        f = lambda x: 255 if (x>0)  else 0
    filter_diff = np.vectorize(f)
    bw_img = filter_diff(img)
    bw_img = bw_img.astype(np.uint8)
    print("bw-image: ", bw_img.shape, bw_img.dtype, type(bw_img))
    diff_img = im.fromarray(bw_img)
    diff_img.save(path)


def get_labels_color_map(labels):
    cm = pylab.get_cmap('gist_rainbow')
    colors = []
    n = len(labels)
    for i in range(n):
        colors.append(cm(1.*i/n))  # color will now be an RGBA tuple
    label_colors = {}
    for i in range(n):
        c = colors[i]
        label_colors[labels[i]]= (int(c[0] *255),int(c[1] *255),int(c[2] *255),int(c[3] *255))
    return label_colors

def generate_masks(render_path, mask_path):
    base_img = os.path.join(render_path, "render_empty.exr")
    base_depth = read_exr_channel(base_img, 'Z')
    
    scene_folders = [f.path for f in os.scandir(render_path) if f.is_dir()]
    for scene_dir in scene_folders:

        label_path = os.path.join(scene_dir, "labels.txt")
        f = open(label_path)
        lines = f.readlines()

        print(scene_dir)
        folder_num = os.path.basename(scene_dir)
        mask_dir = os.path.join(mask_path, folder_num)
        if not os.path.isdir(mask_dir):
            os.mkdir(mask_dir)
        
        i = 0
        bg_init = False
        bg = None
        # damn, still not done. Overlapping single masks could exist
        # We need to make sure to only add the front obj to the bitmask
        print("isfile", os.path.join(scene_dir, "render_img_"+str(i)+".exr"), os.path.isfile(os.path.join(scene_dir, "render_img_"+str(i)+".exr")))
        while os.path.isfile(os.path.join(scene_dir, "render_img_"+str(i)+".exr")):
            img_path = os.path.join(scene_dir, "render_img_"+str(i)+".exr")
            if not os.path.isfile(img_path):
                sys.exit(42)
            out_path = os.path.join(mask_dir, lines[i].split()[0] + ".png")
            img_depth = read_exr_channel(img_path, 'Z')
            diff = base_depth - img_depth
            store_array_as_img(diff, out_path, False)
            if not bg_init:
                bg = diff
                bg_init = True
            else:
                bg += diff
            i += 1
        print(type(bg))
        store_array_as_img(bg, os.path.join(mask_path, folder_num, "bg.png"), True)

def draw_bb(base_path, label_path):
    img_path = os.path.join(base_path, "label.png")
    img = im.open(img_path)
    draw = ImageDraw.Draw(img)
    f = open(label_path)
    lines = f.readlines()
    coordinates = []
    for line in lines:
        entries = line.split()
        for i, entry in enumerate(entries):
            if i == 0:
                continue
            coordinates.append( entry.split(",") )
    for c in coordinates:
        (x,y,w,h) = (int(c[0]), int(c[1]), int(c[2]), int(c[3]))
        draw.rectangle([(x,y),(x+w,y+h)], outline=(0,0,0))
    img.save(os.path.join(base_path, "labels_w_box.png"))

def copy_input_images(render_path, input_img_dir):
    folder_searchpath = os.path.join(render_path, "*")
    render_folders = glob.glob(folder_searchpath)
    for folder in render_folders:
        if not os.path.isdir(folder):
            continue
        folder_num = os.path.basename(folder)
        input_img_path = os.path.join(render_path, str(folder_num), "render_full.png")
        output_img_path = os.path.join(input_img_dir, str(folder_num), "input.png")
        if os.path.isfile(input_img_path):
            shutil.copyfile(input_img_path, output_img_path)
        else:
            print("skipping ", folder_num, " ", input_img_path)


base_path   = "/media/MX500/CS_BA_Data"
render_path = os.path.join(base_path, "render_output")
#label_path = os.path.join(some_path, "labels.txt")
input_img_dir = os.path.join(base_path, "training")
sample_path = os.path.join(base_path, "SamplePBR")
mask_path   = os.path.join(base_path, "masks")

generate_masks(render_path, input_img_dir)
copy_input_images(render_path, input_img_dir)
#draw_bb(render_path, label_path)
