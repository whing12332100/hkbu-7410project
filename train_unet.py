import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
from data1 import *
from metric import *
import time
from symbol_unet import Segnet
import cv2
import os
# Let us define the data processing procedures
my_train_aug_tmp = Compose([
    RandomCrop(500),
    #RandomAffine(),
    Resize(500,500)
    ])

# We are now creating the customize dataset via our own class
my_train_tmp = MyDataSet('data', 'train', my_train_aug_tmp)
# Let us visualize the image after augmentation
# Feel free to change
img_idx  = 1
[a,b] = my_train_tmp.__getitem__(img_idx)
#a = a.asnumpy()#.transpose((1,2,0)) + 107
plt.imshow(a)
plt.show()

# Please feel free to change the data processing procedures and the img_idx to see the difference
# Try to uncomment the RandomAffine()
# Try to change the img_idx
# Define the preprocessing procedures,
# For detail implementation, please refer to 'data.py'
my_train_aug = Compose([
    RandomCrop(500),
    Resize(500,500),
    #RandomAffine(),
    ToNDArray(), # the network  created by mxnet accepts their mxnet ndarray as input only
    Normalize(mx.nd.array([107]), mx.nd.array([1])) # perform the image wise normalization
    ])

# Creeate the dataset
my_train = MyDataSet('data', 'train', my_train_aug)

# Create training data loader
train_loader = DataLoader(my_train, batch_size=1, shuffle=True, last_batch='rollover')

unet = Segnet()
unet.hybridize()

# Instead of random intialize the parameters of U-Net as above,
# We can also load the parameters that is stored in the system
unet.load_params('model/segnet-0900.params', ctx=mx.cpu())

x_u = mx.sym.var('data')
sym_u = unet(x_u)



class PolyScheduler(mx.lr_scheduler.LRScheduler):
    def __init__(self, base_lr, lr_power, total_steps):
        super(PolyScheduler, self).__init__(base_lr=base_lr)
        self.lr_power = lr_power
        self.total_steps = total_steps

    def __call__(self, num_update):
        lr = self.base_lr * ((1 - float(num_update) / self.total_steps) ** self.lr_power)
        return lr


num_steps = len(my_train) / 16
trainer = mx.gluon.Trainer(unet.collect_params(), 'sgd', {
    'learning_rate': 0.5,
    'wd': 0.0005,
    'momentum': 0.9,
    'lr_scheduler': PolyScheduler(1.0, 0.9, num_steps * 1200)
})

criterion = mx.gluon.loss.SoftmaxCrossEntropyLoss(axis=1)
metrics = [SegMetric()]


ctx = [mx.cpu()]

num_epochs = 2
for epoch in range(num_epochs):
    t0 = time.time()
    total_loss = 0
    for m in metrics:
        m.reset()
    for data, label in train_loader:
        batch_size = data.shape[0]
        dlist = mx.gluon.utils.split_and_load(data, ctx)
        llist = mx.gluon.utils.split_and_load(label, ctx)
        mlist = [y != 255 for y in llist]
        with mx.autograd.record():
            # losses = [criterion(net(X), y, m) for X, y in zip(dlist, llist, mlist)]
            preds = [unet(X) for X in dlist]
            losses = []
            for i in range(len(preds)):
                l = criterion(preds[i], llist[i], mlist[i])
                losses.append(l)
        for l in losses:
            l.backward()
        total_loss += sum([l.sum().asscalar() for l in losses])
        trainer.step(batch_size)
        for m in metrics:
            m.update(labels=llist, preds=preds)
        for m in metrics:
            name, value = m.get()
        t1 = time.time()
        print(epoch, t1 - t0, total_loss, name, value)
        break

    if epoch % 100 == 1:
        unet.export('model/segnet')
        unet.save_parameters('model/segnet-%04d.params' % epoch)

    print(epoch, t1 - t0, total_loss, name, value)
