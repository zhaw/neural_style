import mxnet as mx
import numpy as np

def block(data, num_filter, name):
    data = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(3,3), pad=(1,1), name='%s_conv1'%name)
    data = mx.sym.BatchNorm(data=data, momentum=0., name='%s_batchnorm1'%name)
    data = mx.sym.LeakyReLU(data=data, slope=0.1)
    data = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(3,3), pad=(1,1), name='%s_conv2'%name)
    data = mx.sym.BatchNorm(data=data, momentum=0., name='%s_batchnorm2'%name)
    data = mx.sym.LeakyReLU(data=data, slope=0.1)
    data = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1,1), pad=(0,0), name='%s_conv3'%name)
    data = mx.sym.BatchNorm(data=data, momentum=0., name='%s_batchnorm3'%name)
    data = mx.sym.LeakyReLU(data=data, slope=0.1)
    return data


def join(data, data_low, num_filter, name):
    data_low = mx.sym.UpSampling(data_low, scale=2, num_filter=num_filter, sample_type='nearest', num_args=1)
#     data_low = mx.sym.Deconvolution(data_low, kernel=(2,2), stride=(2,2), num_filter=num_filter, name='%s_upsample_low'%name)
    data_low = mx.sym.BatchNorm(data=data_low, momentum=0, name='%s_batchnorm_low'%name)
    data = mx.sym.BatchNorm(data=data, momentum=0, name='%s_batchnorm'%name)
    out = mx.sym.Concat(data, data_low)
    return out


def generator_symbol(m, task):
    Z = []
    for i in range(m):
        if task == 'texture':
            Z.append(block(mx.sym.Variable('z_%d'%i), 8, name='block%d'%i))
        else:
            noise = mx.sym.Variable('znoise_%d'%i)
            im = mx.sym.Variable('zim_%d'%i)
            Z.append(block(mx.sym.Concat(noise, im), 8, name='block%d'%i))
#            for j in range(i):
#                im = block(im, 8, name='block%d'%(m*i+j))
#            Z.append(block(im, 8, name='block%d'%i))
    for i in range(1,m):
        Z[i] = block(join(Z[i], Z[i-1], i*8, name='join%d'%i), 8*(i+1), name='blockjoin%d'%i)
    out = mx.sym.Convolution(data=Z[-1], num_filter=3, kernel=(1,1), pad=(0,0), name='blockout')
#     out = mx.sym.BatchNorm(data=out, name='blockoutbn')
#     out = mx.sym.LeakyReLU(data=out, slope=0.1)
    return out


def descriptor_symbol(style_layers=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'], content_layer='relu4_2', task='style'):
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
#     style_out = mx.sym.Group([mx.symbol.Dropout(data=x, p=dp) for x in map(eval, style_layers)])
    style_out = mx.sym.Group([x for x in map(eval, style_layers)])
    if task != 'style':
        return style_out
    return mx.sym.Group([style_out, eval(content_layer)])

