import OpenEXR as exr
import numpy
import numpy as np
import os
import glob
import sys
import Imath
import pylab # for colors
from PIL import Image as im
#from os import Path

def read_exr_channel(filepath: os.path, channel):
    exrfile = exr.InputFile(filepath)
    raw_bytes = exrfile.channel(channel, Imath.PixelType(Imath.PixelType.FLOAT))
    depth_vector = numpy.frombuffer(raw_bytes, dtype=numpy.float32)
    height = exrfile.header()['displayWindow'].max.y + 1 - exrfile.header()['displayWindow'].min.y
    width = exrfile.header()['displayWindow'].max.x + 1 - exrfile.header()['displayWindow'].min.x
    depth_map = numpy.reshape(depth_vector, (height, width))
    return depth_map

def read_exr_file(filepath: os.path):
    channels = ['A','R','G','B','Z']
    img = []
    for channel in channels:
        exrfile = exr.InputFile(filepath)
        raw_bytes = exrfile.channel(channel, Imath.PixelType(Imath.PixelType.FLOAT))
        depth_vector = numpy.frombuffer(raw_bytes, dtype=numpy.float32)
        height = exrfile.header()['displayWindow'].max.y + 1 - exrfile.header()['displayWindow'].min.y
        width = exrfile.header()['displayWindow'].max.x + 1 - exrfile.header()['displayWindow'].min.x
        depth_map = numpy.reshape(depth_vector, (height, width))
        img.append(depth_map)
    return np.array(img)

def safe_array_as_img(img, path):
    #f = lambda x: 255 if (x>0)  else 0
    f = lambda x: 0 if (x>0)  else 255
    filter_diff = np.vectorize(f)
    bw_img = filter_diff(img)
    bw_img = bw_img.astype(np.uint8)
    #print("bw-image: ", bw_img.shape, bw_img.dtype, type(bw_img))
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


def get_merged_label_img(depth_images, colors, out_path):
    depth_arr = np.array(depth_images)
    pos = np.argmax(depth_arr, axis=0).astype(np.uint8)
    #print("pos info ", pos.shape, pos.max(), pos)
    f = lambda x: colors[x]
    filter_diff = np.vectorize(f)
    label_img = np.array(filter_diff(pos))
    label_img = np.moveaxis(label_img, 0, 2)
    label_img = label_img.astype(np.uint8)
    tmp = im.fromarray(label_img, mode='RGBA')
    tmp.save(out_path)

def create_labels(base_path, label_path):
    sample_path = os.path.join(base_path, "..", "SamplePBR")
    merge_label_path = os.path.join(base_path, "label.png")

    # total_x is based on the amount of surfaces on which we work 
    total_labels = [ f.name for f in os.scandir(sample_path) if f.is_dir() ]
    total_label_colors = get_labels_color_map(total_labels)
    print(total_label_colors)

    img_path_full = os.path.join(base_path, "render_full.exr")
    img_path_empty = os.path.join(base_path, "render_empty.exr")
    img_paths = glob.glob(base_path + "/*.exr")
    base_img_full = read_exr_file(img_path_full)
    base_img_empty = read_exr_file(img_path_empty)
    depth_map_full = base_img_full[4]
    depth_map_empty = base_img_empty[4]
    depth_map_background = depth_map_empty - depth_map_full
    safe_array_as_img(depth_map_background, os.path.join(base_path, "background.png"))

    print("depth_map: ", depth_map_full.shape, depth_map_full.min(), depth_map_full.max())
    out_shape = (3, *depth_map_full.shape)
    print("out_shape: ", out_shape)

    # scene_x is based on the ones existing in the current scene. Might not cover all surfaces
    scene_images = []
    scene_depth_images = []
    scene_colors = []

    img_paths = glob.glob(os.path.join(base_path, "render_img_*.exr"))
    for i, img_path in enumerate(img_paths):
        out_path = os.path.join(base_path, str(i) + ".png")
        print(out_path, str(i))
        img = read_exr_file(img_path)
        scene_images.append(img)
        #scene_depth_images.append(img[4])
        f = open(label_path, 'r')
        labels_str = f.readlines()
        label = labels_str[i].split()[0]
        scene_colors.append(total_label_colors[label])
        diff = img[4] - depth_map_full
        scene_depth_images.append(diff)
        safe_array_as_img(diff, out_path)

    scene_colors.append((255,255,255,255)) # background color
    scene_depth_images.append(np.ones(depth_map_full.shape)) #

    print(scene_colors)
    get_merged_label_img(scene_depth_images, scene_colors, merge_label_path)

base_path = "/media/MX500/CS_BA_Data/render_output"
label_path = os.path.join(base_path, "labels.txt")

create_labels(base_path, label_path)



