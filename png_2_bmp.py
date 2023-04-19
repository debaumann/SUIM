from __future__ import print_function, division
import os
import ntpath
import numpy as np
from PIL import Image
from os.path import join, exists
from utils.data_utils import getPaths



convert_dir = "data/seg_masks"
out_dir = "data/train_val/masks//"

for p in getPaths(convert_dir):
        # read and scale inputs
        
        img = Image.open(p).resize((853, 480))
        
        print(np.shape(img))

        img_name = ntpath.basename(p).split('.')[0] + '.bmp'
        Image.fromarray(np.uint8(img)).save(out_dir+img_name)
        