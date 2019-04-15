import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
from data1 import *
from metric import *
import time
import cv2
import os
from symbol_unet import Segnet
ctx = mx.cpu()
# Initialize the network and also the pre-trained parameters
net = Segnet()
net.hybridize()


## Models trained with different iterations are provided,
## (i.e., segnet-0100.params -- segnet-0900.params), try to load different
## models and see the difference
net.load_params('model/segnet-0200.params', ctx=ctx)


# Define the preprocessing steps for the testing data
my_val_aug = Compose([
    #RandomCrop(500),
    Resize(500,500),
    #RandomAffine(),
    ToNDArray(),
    Normalize(mx.nd.array([107]), mx.nd.array([1]))
])

# create the dataset for testing
my_val = MyDataSet('data', 'test', None)
pred_idx = 0

[img, lbl] = my_val.__getitem__(pred_idx)
img2, lbl2 = my_val_aug(img, lbl)

imgfill = np.ones((img.shape[0],np.int_(img.shape[1]*0.1),3))*255

#a_in = a.asnumpy().transpose((1,2,0)) + 107
lbl_label = cv2.cvtColor(lbl*255,cv2.COLOR_GRAY2RGB)
im_show = np.hstack((img,imgfill))
im_show = np.hstack((im_show,lbl_label))

output = net(img2.expand_dims(axis=0))
output = output.asnumpy()
pred = np.uint8(np.argmax(output,axis=1))
b_pred = cv2.cvtColor(cv2.resize(pred[0]*255,(img.shape[1],img.shape[0]),0, 0, cv2.INTER_NEAREST),cv2.COLOR_GRAY2RGB)

im_show = np.hstack((im_show,imgfill))
im_show = np.hstack((im_show,b_pred))

plt.imshow(im_show/255.)
plt.show()