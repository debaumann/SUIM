from __future__ import print_function, division
import os
import ntpath
import numpy as np
from PIL import Image
from os.path import join, exists
import onnxruntime as rt
# local libs
from models.unet import UNet0
from utils.data_utils import getPaths





sess_options = rt.SessionOptions()
sess_options.graph_optimization_level =rt.GraphOptimizationLevel.ORT_ENABLE_ALL

session = rt.InferenceSession("models/suimnet_may.onnx", sess_options=sess_options, providers=['CUDAExecutionProvider'])
test_dir = "data/selected_1/"

## sample and ckpt dir
eval_dir = "data/onnx/"



im_h, im_w = 240, 320
def testGenerator():
    # test all images in the directory
    assert exists(test_dir), "local image path doesnt exist"
    imgs = []
    for p in getPaths(test_dir):
        # read and scale inputs
        img = Image.open(p).resize((im_w, im_h))
        img = np.array(img, dtype=np.float32)/255.
        img = np.expand_dims(img, axis=0)
        print(np.shape(img))
        # inference
        out_img = session.run(None, {'args_0': img})
        # thresholding
        #visualize 
        out_img = np.array(out_img)
        print(np.shape(out_img))
        m=np.array(out_img[0,0,:,:,0])
        m = m+ np.array(out_img[0,0,:,:,1])
   
        m[m>0.5] = 1.
        m[m<=0.5] = 0.
        
        print(np.shape(m))

        
        # get filename
        img_name = ntpath.basename(p).split('.')[0] + '.bmp'
        # save individual output masks
        Image.fromarray(np.uint8(m*255)).save(eval_dir+img_name)
     
testGenerator()