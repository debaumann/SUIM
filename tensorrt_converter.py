import tensorrt as trt
import numpy as np 
from PIL import Image
import tensorflow as tf
from utils.data_utils import getPaths

test_dir = "data/selected_1/"

SAVED_MODEL_DIR= "models/suimnet_may.h5"
print(tf.__version__)
converter = trt.TrtGraphConverterV2(
   input_saved_model_dir=SAVED_MODEL_DIR,
   precision_mode=trt.TrtPrecisionMode.INT8
)


im_res_ = (320, 240, 3)
im_h, im_w = im_res_[1], im_res_[0]
BATCH_SIZE = 8

def representative_dataset():
    imgs = []
    for p in getPaths(test_dir):
        # read and scale inputs
        img = Image.open(p).resize((im_w, im_h))
        img = np.array(img)/255.
        img = img.astype(np.float32)
        imgs.append(img)
    imgs = np.array(imgs)
    print(imgs.shape)
    img = tf.data.Dataset.from_tensor_slices(imgs).batch(1)
    for i in img.take(BATCH_SIZE):
        yield [i]
#gotta be able to push 

trt_func = converter.convert()
converter.summary()
print(converter.summary())