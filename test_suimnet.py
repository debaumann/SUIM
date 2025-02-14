"""
# Test script for the SUIM-Net
    # for 5 object categories: HD, FV, RO, RI, WR 
# Paper: https://arxiv.org/pdf/2004.01241.pdf  
"""
from __future__ import print_function, division
import os
import ntpath
import numpy as np
from PIL import Image
from os.path import join, exists
# local libs
from models.suim_net import SUIM_Net
from utils.data_utils import getPaths

## experiment directories
#test_dir = "/mnt/data1/ImageSeg/suim/TEST/images/"
test_dir = "data/test/images/italy"

## sample and ckpt dir
samples_dir = "data/test/output/"
eval_dir = "data/test/output/eval_run_lr001/"
RO_dir = samples_dir + "RO/"
FB_dir = samples_dir + "FV/"
WR_dir = samples_dir + "WR/"
HD_dir = samples_dir + "HD/"
RI_dir = samples_dir + "RI/" 
if not exists(samples_dir): os.makedirs(samples_dir)
if not exists(RO_dir): os.makedirs(RO_dir)
if not exists(FB_dir): os.makedirs(FB_dir)
if not exists(WR_dir): os.makedirs(WR_dir)
if not exists(HD_dir): os.makedirs(HD_dir)
if not exists(RI_dir): os.makedirs(RI_dir)

## input/output shapes
base_ = 'RSB' # or 'RSB'
if base_=='RSB':
    im_res_ = (320, 240, 3) 
    ckpt_name = "suimnet_rsb_500_24_lr_001.hdf5"
else: 
    im_res_ = (320, 256, 3)
    ckpt_name = "suimnet_vgg5.hdf5"
suimnet = SUIM_Net(base=base_, im_res=im_res_, n_classes=2)
model = suimnet.model
print (model.summary())
model.load_weights(join("ckpt/saved/", ckpt_name))
print("works here")


im_h, im_w = im_res_[1], im_res_[0]
def testGenerator():
    # test all images in the directory
    assert exists(test_dir), "local image path doesnt exist"
    imgs = []
    for p in getPaths(test_dir):
        # read and scale inputs
        img = Image.open(p).resize((im_w, im_h))
        img = np.array(img)/255.
        img = np.expand_dims(img, axis=0)
        
        # inference
        out_img = model.predict(img)
        # thresholding
        out_img[out_img>0.5] = 1.
        out_img[out_img<=0.5] = 0.
        print ("tested: {0}".format(p))
        #visualize 
        print(np.shape(out_img))
        m=np.array(out_img[0,:,:,0])
        m = m+ np.array(out_img[0,:,:,1])
        # m = m+ np.array(out_img[0,:,:,2])
        # m = m+ np.array(out_img[0,:,:,3])
        # m = m+ np.array(out_img[0,:,:,4])
        m[m>0.5] = 1.
        m[m<=0.5] = 0.
        
        print(np.shape(m))

        
        # get filename
        img_name = ntpath.basename(p).split('.')[0] + '.bmp'
        # save individual output masks
        Image.fromarray(np.uint8(m*255)).save(eval_dir+img_name)
        # ROs = np.reshape(out_img[0,:,:,0], (im_h, im_w))
        # FVs = np.reshape(out_img[0,:,:,1], (im_h, im_w))
        # HDs = np.reshape(out_img[0,:,:,2], (im_h, im_w))
        # RIs = np.reshape(out_img[0,:,:,3], (im_h, im_w))
        # WRs = np.reshape(out_img[0,:,:,4], (im_h, im_w))
        # Image.fromarray(np.uint8(ROs*255.)).save(RO_dir+img_name)
        # Image.fromarray(np.uint8(FVs*255.)).save(FB_dir+img_name)
        # Image.fromarray(np.uint8(HDs*255.)).save(HD_dir+img_name)
        # Image.fromarray(np.uint8(RIs*255.)).save(RI_dir+img_name)
        # Image.fromarray(np.uint8(WRs*255.)).save(WR_dir+img_name)

# test images
testGenerator()


