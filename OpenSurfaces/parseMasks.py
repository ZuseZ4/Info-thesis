
##########################################################################################################################################################################

from PIL import Image
import numpy as np
import os
import multiprocessing

import OpenSurfacesClasses


def process_mask(AnnotationDir, OutputDir, dic, mask):
    input_path = os.path.join(AnnotationDir, mask)
    I = np.array(Image.open(input_path))
    for i in range(1, NumClasses+1):
        label = dic[i].replace(" / ", " or ")
        output_path = os.path.join(OutputDir, label, mask)
        out = Image.fromarray((I == i))
        if not out.getbbox():
            continue # img is completely black, thus class not present in image
        out.save(output_path)

#...........................................Input Parameters.................................................

#AnnotationDir="OpenSurfaceMaterialsSmall/TestLabels/"
#OutputDir=r"OpenSurfaceMaterialsSmall/TestLabelsSplit/"

AnnotationDir="OpenSurfaceMaterialsSmall/TrainLabels/"
OutputDir=r"OpenSurfaceMaterialsSmall/TrainLabelsSplit/"

ImageDir="OpenSurfaceMaterialsSmall/Images/"

if not os.path.isdir(OutputDir):
    os.mkdir(OutputDir)
NumClasses=44 # Number of classes if  -1 read num classes from the reader
BackgroundClass=0 # Marking for background/unknown class that will be ignored


dic=OpenSurfacesClasses.CreateMaterialDict()

for i in range(1,NumClasses+1):
    label = dic[i]
    label_dir = os.path.join(OutputDir, label).replace(" / ", " or ")
    if not os.path.isdir(label_dir):
        os.mkdir(label_dir)

mask_images = [f.name for f in os.scandir(AnnotationDir) if f.is_file()]
print(len(mask_images))
    
n_cpus = multiprocessing.cpu_count()
print ("multiprocessing: using %s processes" % n_cpus)
pool = multiprocessing.Pool(n_cpus)
for i in range(len(mask_images)):
    print(i)
    pool.apply_async(process_mask, args = (AnnotationDir, OutputDir, dic, mask_images[i]))
pool.close()
pool.join()
