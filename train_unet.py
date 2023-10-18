"""
# Training pipeline of UNet on SUIM 
"""
from __future__ import print_function, division
import numpy as np
import os
from os.path import join, exists
from keras import callbacks
# local libs
from models.unet import UNet0
from utils.data_utils import trainDataGenerator
import wandb
from wandb.keras import WandbMetricsLogger

## dataset directory
dataset_name = "suim"
train_dir = "data/train_val/"
wandb.init(config={"bs":15})
## ckpt directory
ckpt_dir = "ckpt/"
im_res_ = (320, 240, 3)
ckpt_name = "unet_ba.hdf5"
model_ckpt_name = join(ckpt_dir, ckpt_name)
if not exists(ckpt_dir): os.makedirs(ckpt_dir)

## initialize model
model = UNet0(input_size=(im_res_[1], im_res_[0], 3), no_of_class=2)
print (model.summary())
## load saved model
#model.load_weights(join("ckpt/saved/", "***.hdf5"))

class WandbCallback(callbacks.Callback):
    def __init__(self,data_generator, num_samples):
        self.data_generator=data_generator
        self.num_samples = num_samples
    def on_epoch_end(self, epoch,logs={}):
        images, masks= next(self.data_generator)
        predicted_masks= self.model.predict(images)        
        
        
        for i in range(self.num_samples):
            out_img = predicted_masks[i]
            out_img[out_img>0.5] = 1.
            out_img[out_img<=0.5] = 0.
            
            #visualize 
            m=np.array(out_img[:,:,0])
            m = m+ np.array(out_img[:,:,1])
            # m = m+ np.array(out_img[0,:,:,2])
            # m = m+ np.array(out_img[0,:,:,3])
            # m = m+ np.array(out_img[0,:,:,4])
            m[m>0.5] = 1.
            m[m<=0.5] = 0.
            mar = np.dstack((m*255,m*255,m*255))
            masks_ar= np.dstack((masks[i]*255,masks[i]*255,masks[i]*255))
            img_array = np.hstack((mar, masks_ar, images[i]*255))
            wandb.log({
                "prediction": wandb.Image(img_array, caption="left: pred_mask / mid: gt / right: og_image ")  
            })

batch_size = 4
num_epochs = 50
# setup data generator
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

model_checkpoint = callbacks.ModelCheckpoint(model_ckpt_name, 
                                   monitor = 'loss', 
                                   verbose = 1, mode= 'auto',
                                   save_weights_only = True,
                                   save_best_only = True)

# data generator
train_gen = trainDataGenerator(batch_size, # batch_size 
                              train_dir,# train-data dir
                              "images", # image_folder 
                              "masks", # mask_folder
                              data_gen_args, # aug_dict
                              image_color_mode="rgb", 
                              mask_color_mode="rgb",
                              target_size = (im_res_[1], im_res_[0]))
#callback images
img_gen = trainDataGenerator(batch_size, # batch_size 
                              train_dir,# train-data dir
                              "images", # image_folder 
                              "masks", # mask_folder
                              data_gen_args, # aug_dict
                              image_color_mode="rgb", 
                              mask_color_mode="rgb",
                              target_size = (im_res_[1], im_res_[0]))

## fit model
model.fit(train_gen, 
                    steps_per_epoch = 300,
                    epochs = num_epochs,
                    callbacks = [model_checkpoint,WandbMetricsLogger(),WandbCallback(img_gen,2)])
# batch_size = 2
# num_epochs = 50
# # setup data generator
# data_gen_args = dict(rotation_range=0.2,
#                     width_shift_range=0.05,
#                     height_shift_range=0.05,
#                     shear_range=0.05,
#                     zoom_range=0.05,
#                     horizontal_flip=True,
#                     fill_mode='nearest')

# model_checkpoint = callbacks.ModelCheckpoint(model_ckpt_name, 
#                                    monitor = 'loss', 
#                                    verbose = 1, mode= 'auto',
#                                    save_weights_only = True,
#                                    save_best_only = True)

# # data generator
# train_gen = trainDataGenerator(batch_size, # batch_size 
#                               train_dir,# train-data dir
#                               "images", # image_folder 
#                               "masks", # mask_folder
#                               data_gen_args, # aug_dict
#                               image_color_mode="rgb", 
#                               mask_color_mode="rgb",
#                               target_size = (im_res_[1], im_res_[0]))

# ## fit model
# model.fit(train_gen, 
#                     steps_per_epoch = 10,
#                     epochs = num_epochs,
#                     callbacks = [model_checkpoint])

