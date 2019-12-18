#%%
import mxnet
from mxnet import gluon
from mxnet.gluon import nn
#import mxnet.ndarray as F
import numpy as np
from matplotlib import pyplot as plt
from mxnet import autograd as ag

#%%
print(help(nn))

# %%

dshape = np.memmap('MxNet/Data/mnist.mm', dtype=np.int32, shape=(4,))
dshape
datashape = tuple(dshape)
del dshape
# %%
mmdata = np.memmap('MxNet/Data/mnist.mm', dtype=np.float32, shape=datashape, offset=4)
data = mmdata.copy()
del mmdata
data = np.moveaxis(data, -1, 1)
data.shape
# %%
#plt.imshow(data[1,:,:, 0])

# %%
raw_labels = np.fromfile('MxNet/Data/train-labels-idx1-ubyte',
                    dtype=np.uint8)
labels = raw_labels[8:]

# %%
i=-300
plt.imshow(data[i,:,:, 0])
labels[i]

# %%
class CNN(gluon.HybridBlock):
    def __init__(self, **kwargs):
        super(CNN, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = nn.Conv2D(16, kernel_size=(5,5))
            self.bn1 = nn.BatchNorm(axis=1, scale=True)
            self.pool1 = nn.MaxPool2D(pool_size=(2,2), strides=(2,2))
            self.conv2 = nn.Conv2D(32, kernel_size=(3,3))
            self.bn2 = nn.BatchNorm(axis=1, scale=True)
            self.pool2 = nn.MaxPool2D(pool_size=(2,2), strides=(2,2))
            self.conv3 = nn.Conv2D(48, kernel_size=(3,3))
            self.bn3 = nn.BatchNorm(axis=1, scale=True)
            self.pool3 = nn.MaxPool2D(pool_size=(2,2), strides=(2,2))
            self.fc1 = nn.Dense(512)
            self.fc2 = nn.Dense(128)
            self.fc3 = nn.Dense(10)

    def hybrid_forward(self, F, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.reshape((0, -1))
        x = F.relu(self.fc1(x))
        ox = F.relu(self.fc2(x))
        self.totsne = ox
        x = self.fc3(ox)
        return x

    def get_vis():
        return self.totsne

cnn = CNN()   
cnn.hybridize()
# %%
cnn

# %%
gpus = mxnet.test_utils.list_gpus()
gpus

# %%
ctx = [mxnet.gpu()] if gpus else [mxnet.cpu(0), mxnet.cpu(1)]
ctx

# %%
cnn.initialize(mxnet.init.Xavier(magnitude=2.24), ctx=ctx)
trainer = gluon.Trainer(cnn.collect_params(), 'sgd', {'learning_rate':0.02})

#%%

tdata = data[0:-5000]
tlabels = labels[0:-5000]
vdata = data[-5000:]
vlabels = labels[-5000:]

batch_size = 100
train_data = mxnet.io.NDArrayIter(tdata, tlabels, batch_size, shuffle=True)
val_data = mxnet.io.NDArrayIter(vdata, vlabels, batch_size)



# %%
%%time
metric = mxnet.metric.Accuracy()
loss = gluon.loss.SoftmaxCrossEntropyLoss()
nepochs=20
for i in range(nepochs):
    train_data.reset()
    for batch in train_data:
        xd = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx,
                                        batch_axis=0)
        yd = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx,
                            batch_axis=0)
        outputs = list()
        with ag.record():
            for x, y in zip(xd, yd):
                z = cnn(x)
                zloss = loss(z, y)
                zloss.backward()
                outputs.append(z)
        
        metric.update(yd, outputs)
        trainer.step(batch.data[0].shape[0])
    
    name, acc = metric.get()
    metric.reset()
    print('acc at epoch %d: %s=%f'%(i, name, acc))

    cnn.save_parameters("saved")    
#%%
import mxnet.ndarray as ndarray

vcnn = CNN()
vcnn.load_parameters("saved", ctx=ctx[0])
nv = ndarray.array(vdata, ctx=ctx[0])
res = vcnn(nv)
res0 = vcnn.totsne
res0
#%%
val_data.reset()
metric = mxnet.metric.Accuracy()
vislist = list()
for batch in val_data:
    vx = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx,
                                    batch_axis=0)
    vy = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx,
                                    batch_axis=0)

    output = list()
    for x in vx:
        output.append(vcnn(x))
        vislist.append(vcnn.totsne)
    metric.update(vy, output)

print('val acc: %s=%f'%metric.get())

#%%
v0 = vislist[0]

# %%
vs = np.concatenate(vislist, axis=0)
print(vs.shape)
 

#%%


# %%
import umap
u = umap.UMAP()
f = u.fit_transform(res0)
# %%

# %%
import pandas as pd
df = pd.DataFrame({'d1':f[:,0], 'd2':f[:,1], 't':vlabels})

# %%
import seaborn as sns
plt.figure(figsize=(8,8))
sns.scatterplot(x='d1', y='d2', hue='t', data=df,
                palette=sns.color_palette(), s=4,
                linewidth=0, alpha=.75)

# %%
