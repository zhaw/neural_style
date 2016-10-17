import mxnet as mx
import numpy as np

def block(data, num_filter, name):
    data2 = conv(data, num_filter, 1, name=name)
    data2 = mx.sym.Convolution(data=data2, num_filter=num_filter, kernel=(3,3), pad=(1,1), name='%s_conv1'%name)
    data2 = mx.sym.BatchNorm(data=data2, momentum=0.9, name='%s_bn1'%name)
    return mx.sym.Activation(data=data+data2, act_type='relu')

def conv(data, num_filter, stride, name):
    data = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(3,3), pad=(1,1), stride=(stride, stride), name='%s_conv'%name)
    data = mx.sym.BatchNorm(data=data, momentum=0.9, name='%s_conv'%name)
    data = mx.sym.Activation(data=data, act_type='relu')
    return data

def generator_symbol():
    data = mx.sym.Variable('data')
    data = mx.sym.Convolution(data=data, num_filter=32, kernel=(9,9), pad=(4,4), name='conv0')
    data = mx.sym.BatchNorm(data=data, name='bn0')
    data = mx.sym.Activation(data=data, act_type='relu')
    data = conv(data, 64, 2, name='downsample0')
    data = conv(data, 128, 2, name='downsample1')
    data = block(data, 128, name='block0')
    data = block(data, 128, name='block1')
    data = block(data, 128, name='block2')
    data = block(data, 128, name='block3')
    data = block(data, 128, name='block4')
    data = mx.sym.Deconvolution(data=data, kernel=(4,4), pad=(0,0), stride=(2,2), num_filter=64, name='deconv0')
    data = mx.sym.BatchNorm(data=data, momentum=0.9, name='dcbn0')
    data = mx.sym.Activation(data=data, act_type='relu')
    data = mx.sym.Deconvolution(data=data, kernel=(4,4), pad=(0,0), stride=(2,2), num_filter=32, name='deconv1')
    data = mx.sym.BatchNorm(data=data, momentum=0.9, name='dcbn1')
    data = mx.sym.Activation(data=data, act_type='relu')
    data = mx.sym.Convolution(data=data, num_filter=3, kernel=(9,9), pad=(1,1), name='lastconv')
    return data 


def descriptor_symbol(style_layers=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'], content_layer='relu4_2'):
    data = mx.symbol.Variable('data')
    conv1_1 = mx.symbol.Convolution(name='conv1_1', data=data , num_filter=64, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu1_1 = mx.symbol.Activation(name='relu1_1', data=conv1_1 , act_type='relu')
    conv1_2 = mx.symbol.Convolution(name='conv1_2', data=relu1_1 , num_filter=64, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu1_2 = mx.symbol.Activation(name='relu1_2', data=conv1_2 , act_type='relu')
    pool1 = mx.symbol.Pooling(name='pool1', data=relu1_2 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')
    conv2_1 = mx.symbol.Convolution(name='conv2_1', data=pool1 , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu2_1 = mx.symbol.Activation(name='relu2_1', data=conv2_1 , act_type='relu')
    conv2_2 = mx.symbol.Convolution(name='conv2_2', data=relu2_1 , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu2_2 = mx.symbol.Activation(name='relu2_2', data=conv2_2 , act_type='relu')
    pool2 = mx.symbol.Pooling(name='pool2', data=relu2_2 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')
    conv3_1 = mx.symbol.Convolution(name='conv3_1', data=pool2 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu3_1 = mx.symbol.Activation(name='relu3_1', data=conv3_1 , act_type='relu')
    conv3_2 = mx.symbol.Convolution(name='conv3_2', data=relu3_1 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu3_2 = mx.symbol.Activation(name='relu3_2', data=conv3_2 , act_type='relu')
    conv3_3 = mx.symbol.Convolution(name='conv3_3', data=relu3_2 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu3_3 = mx.symbol.Activation(name='relu3_3', data=conv3_3 , act_type='relu')
    conv3_4 = mx.symbol.Convolution(name='conv3_4', data=relu3_3 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu3_4 = mx.symbol.Activation(name='relu3_4', data=conv3_4 , act_type='relu')
    pool3 = mx.symbol.Pooling(name='pool3', data=relu3_4 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')
    conv4_1 = mx.symbol.Convolution(name='conv4_1', data=pool3 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu4_1 = mx.symbol.Activation(name='relu4_1', data=conv4_1 , act_type='relu')
    conv4_2 = mx.symbol.Convolution(name='conv4_2', data=relu4_1 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu4_2 = mx.symbol.Activation(name='relu4_2', data=conv4_2 , act_type='relu')
    conv4_3 = mx.symbol.Convolution(name='conv4_3', data=relu4_2 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu4_3 = mx.symbol.Activation(name='relu4_3', data=conv4_3 , act_type='relu')
    conv4_4 = mx.symbol.Convolution(name='conv4_4', data=relu4_3 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu4_4 = mx.symbol.Activation(name='relu4_4', data=conv4_4 , act_type='relu')
    pool4 = mx.symbol.Pooling(name='pool4', data=relu4_4 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')
    conv5_1 = mx.symbol.Convolution(name='conv5_1', data=pool4 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu5_1 = mx.symbol.Activation(name='relu5_1', data=conv5_1 , act_type='relu')
    style_out = mx.sym.Group([x for x in map(eval, style_layers)])
    return mx.sym.Group([style_out, eval(content_layer)])

