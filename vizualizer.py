import os
import numpy as np
from PIL import Image
from os.path import join, exists
import matplotlib.pyplot as plt
from pathlib import Path


test_dir = "data/test/images/frontyard_4/"
eval_rsb= "data/test/output/eval_frontyard/frontyard_4/eval_rsb/"
eval_vgg= "data/test/output/eval_frontyard/frontyard_4/eval_vgg/"

len = (len([entry for entry in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, entry))]))
f, axarr = plt.subplots(10,21, sharex=True,sharey=True, figsize=(21,10))
jpgs = sorted(list(Path(test_dir).glob('*.jpg')), key=lambda i: (os.path.splitext(os.path.basename(i))[0]))
rsbs = sorted(list(Path(eval_rsb).glob('*.bmp')), key=lambda i: (os.path.splitext(os.path.basename(i))[0]))
vggs = sorted(list(Path(eval_vgg).glob('*.bmp')), key=lambda i: (os.path.splitext(os.path.basename(i))[0]))
print(list)
count = 0
for p in jpgs:
        # read and scale inputs
        img = Image.open(p).resize((320, 240))
        img = np.array(img)/255.
        if count <= 9:
            axarr[count,0].imshow(img)
        elif count > 9 and count <= 19:
                axarr[count-10,3].imshow(img)
        elif count >= 20 and count<=29 :
                axarr[count -20,6].imshow(img)
        elif count >= 30 and count<=39 :
                axarr[count -30,9].imshow(img)
        elif count >= 40 and count<=49 :
                axarr[count -40,12].imshow(img)
        elif count >= 50 and count<=59 :
                axarr[count -50,15].imshow(img)
        elif count >= 60 and count<=69 :
                axarr[count -60,18].imshow(img)
        count+=1 
                
count=0
for p in rsbs:
        # read and scale inputs
        img = Image.open(p)
        if count <= 9:
            axarr[count,1].imshow(img)
        if count > 9 and count <= 19:
                axarr[count-10,4].imshow(img)
        elif count >= 20 and count<=29 :
                axarr[count-20,7].imshow(img)
        elif count >= 30 and count<=39 :
                axarr[count -30,10].imshow(img)
        elif count >= 40 and count<=49 :
                axarr[count -40,13].imshow(img)
        elif count >= 50 and count<=59 :
                axarr[count -50,16].imshow(img)
        elif count >= 60 and count<=69 :
                axarr[count -60,19].imshow(img)
        count+=1     
                
count=0
for p in vggs:
        # read and scale inputs
        img = Image.open(p)
        if count <= 9:
            axarr[count,2].imshow(img)
        if count > 9 and count <= 19:
                axarr[count-10,5].imshow(img)
        elif count >= 20 and count<=29 :
                axarr[count-20,8].imshow(img)
        elif count >= 30 and count<=39 :
                axarr[count -30,11].imshow(img)
        elif count >= 40 and count<=49 :
                axarr[count -40,14].imshow(img)
        elif count >= 50 and count<=59 :
                axarr[count -50,17].imshow(img)
        elif count >= 60 and count<=69 :
                axarr[count -60,20].imshow(img)
        count+=1
cols=['Original', 'RSB Mask', 'VGG Mask','Original', 'RSB Mask', 'VGG Mask','Original', 'RSB Mask', 'VGG Mask','Original', 'RSB Mask', 'VGG Mask']         
for ax, col in zip(axarr[0], cols):
    ax.set_title(col)
f.tight_layout()
plt.show()