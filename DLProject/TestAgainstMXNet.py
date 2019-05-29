import os
import torch
import numpy as np
import mxnet as mx
from torch import nn
from time import time
from pprint import pprint
from torch.autograd import Variable
from mxnet.initializer import Initializer
from DeformConv2DTorch import DeformConv2D
from keras.models import Sequential
from keras.layers import Conv2D
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input
from keras.backend import tf
from keras import backend as K
from keras import optimizers
from DeformConv2DKeras import Conv2DOffset

bs, inC, ouC, H, W = 1, 1, 1, 4, 5
kH, kW = 3, 3
padding = 1

# ---------------------------------------
use_gpu = torch.cuda.is_available()
gpu_device = 0
if use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    print("Using gpu{}".format(os.getenv("CUDA_VISIBLE_DEVICES")))
# ---------------------------------------
##################### SETUP OUR INITIAL INPUTS ####################
raw_inputs = np.random.rand(bs, inC, H, W).astype(np.float32)
raw_labels = np.random.rand(bs, ouC, (H+2*padding-2)//1, (W+2*padding-2)//1).astype(np.float32)
# weights for conv offsets.
offset_weights = np.random.rand(18, inC, 3, 3).astype(np.float32)
# weights for deformable convolution.
conv_weights = np.random.rand(ouC, inC, 3, 3).astype(np.float32)
print('\ninputs:')
pprint(raw_inputs)
print('\nlabels:')
pprint(raw_labels)
print('\nconv weights:')
pprint(conv_weights)

###################################################################
######################## Our PyTorch Model ########################
class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.conv_offset = nn.Conv2d(in_channels=inC, out_channels=18, kernel_size=3, padding=padding, bias=None)
        self.deform_conv = DeformConv2D(inc=inC, outc=ouC, padding=padding)

    def forward(self, x):
        offsets = self.conv_offset(x)
        out = self.deform_conv(x, offsets)
        return out

model = TestModel()

pt_inputs = Variable(torch.from_numpy(raw_inputs).cuda(), requires_grad=True)
pt_labels = Variable(torch.from_numpy(raw_labels).cuda(), requires_grad=False)

optimizer = torch.optim.SGD([{'params': model.parameters()}], lr=1e-1)
loss_fn = torch.nn.MSELoss(reduce=True)

def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        m.weight.data = torch.from_numpy(conv_weights)
        if m.bias is not None:
            m.bias.data = torch.FloatTensor(m.bias.shape[0]).zero_()

def init_offsets_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        m.weight.data = torch.from_numpy(offset_weights)
        if m.bias is not None:
            m.bias.data = torch.FloatTensor(m.bias.shape[0]).zero_()

model.deform_conv.apply(init_weights)
model.conv_offset.apply(init_offsets_weights)
if use_gpu:
    model.cuda()

output = model(pt_inputs)
pprint(output)

for i in range(10):
    output = model(pt_inputs)
    loss = loss_fn(output, pt_labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print("\n\nOur pytorch output is:\n\n")
pprint(model(pt_inputs))
###################################################################
######################### Our Keras Model #########################

def TestModelK(input_tensor=None):
        
    #self.conv_offset = nn.Conv2d(in_channels=inC, out_channels=18, kernel_size=3, padding=padding, bias=None)
    #self.deform_conv = DeformConv2D(inc=inC, outc=ouC, padding=padding)
    if input_tensor is None:
        img_input = Input(shape=(H, W, inC))
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=inC)
        else:
            img_input = input_tensor

    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    x = Conv2DOffset(inC, (3, 3), input_shape=(H, W, inC))(img_input)
    x = Conv2D(ouC, kernel_size=(3, 3), strides=(3, 3), padding='valid', use_bias=False)(x)

    model = Model(img_input, x)
    return model

modelK = TestModelK()

k_inputs = np.transpose(raw_inputs, (0, 2, 3, 1))
k_labels = np.transpose(raw_labels, (0, 2, 3, 1))
#k_conv_weights = np.transpose(conv_weights)
#k_offset_weights = np.transpose(offset_weights)

k_offset_weights = np.expand_dims(offset_weights, axis=0)
k_offset_weights = np.transpose(k_offset_weights, (0, 3, 4, 2, 1))

k_conv_weights = np.expand_dims(conv_weights, axis=0)
k_conv_weights = np.transpose(k_conv_weights, (0, 3, 4, 2, 1))

modelK.layers[1].set_weights(k_offset_weights)
modelK.layers[2].set_weights(k_conv_weights)


Koptim = optimizers.SGD(lr=0.1, nesterov=False)
modelK.compile(loss='mean_squared_error', optimizer=Koptim)

Koutput = modelK.predict(k_inputs)
pprint(Koutput)


modelK.fit(k_inputs, k_labels, batch_size=1, epochs=10, verbose=1)

print("\n\nOur keras output is:\n\n")
Koutput = modelK.predict(k_inputs)
pprint(Koutput)



######################### Our MXNet Model #########################
# trainiter
train_iter = mx.io.NDArrayIter(raw_inputs, raw_labels, 1, shuffle=True, data_name='data', label_name='label')

# # symbol
inputs = mx.symbol.Variable('data')
labels = mx.symbol.Variable('label')
offsets = mx.symbol.Convolution(data=inputs, kernel=(3, 3), pad=(padding, padding), num_filter=18, name='offset', no_bias=True)
net = mx.symbol.contrib.DeformableConvolution(data=inputs, offset=offsets, kernel=(3, 3), pad=(padding, padding), num_filter=ouC, name='deform', no_bias=True)
outputs = mx.symbol.MakeLoss(data=mx.symbol.mean((net-labels)**2))

mod = mx.mod.Module(symbol=outputs,
                    context=mx.gpu(),
                    data_names=['data'],
                    label_names=['label'])

mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
mod.init_params(initializer=mx.initializer.Load({'deform_weight': mx.nd.array(conv_weights),
                                                 'offset_weight': mx.nd.array(offset_weights)}))
mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.1),))



mx_inputs = mx.nd.array(raw_inputs, ctx=mx.gpu())
conv_weights = mx.nd.array(conv_weights, ctx=mx.gpu())
offset_weights = mx.nd.array(offset_weights, ctx=mx.gpu())
offset = mx.ndarray.Convolution(data=mx_inputs, weight=offset_weights, kernel=(3, 3), pad=(padding, padding), num_filter=18, name='offset', no_bias=True)
outputs = mx.ndarray.contrib.DeformableConvolution(
    data=mx_inputs, offset=offset, weight=conv_weights, kernel=(3, 3), pad=(padding, padding), num_filter=ouC, name='deform', no_bias=True)
pprint(outputs)

for i in range(10):
    train_iter.reset()
    for batch in train_iter:
        # get outputs
#         infer_outputs = mx.mod.Module(symbol=net,
#                                      context=mx.gpu(),
#                                      data_names=['data'])
#         infer_outputs.bind(data_shapes=train_iter.provide_data)
#         infer_outputs.set_params(arg_params=mod.get_params()[0], aux_params=mod.get_params()[1], allow_extra=True)
#         outputs_value = infer_outputs.predict(train_iter)

        mod.forward(batch, is_train=True)  # compute predictions
        mod.backward()  # compute gradients
        mod.update()  # update parameters

mx_inputs = mx.nd.array(raw_inputs, ctx=mx.gpu())
mx_labels = mx.nd.array(raw_labels, ctx=mx.gpu())
conv_weights = mod.get_params()[0]['deform_weight'].as_in_context(mx.gpu())
offset_weights = mod.get_params()[0]['offset_weight'].as_in_context(mx.gpu())
offset = mx.ndarray.Convolution(data=mx_inputs, weight=offset_weights, kernel=(3, 3), pad=(padding, padding), num_filter=18, name='offset', no_bias=True)
outputs = mx.ndarray.contrib.DeformableConvolution(data=mx_inputs, offset=offset, weight=conv_weights, kernel=(3, 3), pad=(padding, padding), num_filter=ouC, name='deform', no_bias=True)
print("\n\nReference MXNet output is:\n\n")
pprint(outputs)
###################################################################
