import os
import time
import mxnet as mx
import numpy as np
import symbol
import cPickle as pickle
from skimage import io, transform


def crop_img(im, size):
    im = io.imread(im)
    if im.shape[0]*size[1] > im.shape[1]*size[0]:
        c = (im.shape[0]-1.*im.shape[1]/size[1]*size[0]) / 2
        c = int(c)
        im = im[c:-(1+c),:,:]
    else:
        c = (im.shape[1]-1.*im.shape[0]/size[0]*size[1]) / 2
        c = int(c)
        im = im[:,c:-(1+c),:]
    im = transform.resize(im, size)
    im *= 255
    return im

def preprocess_img(im, size):
    if type(size) == int:
        size = (size, size)
    im = crop_img(im, size)
    im = im.astype(np.float32)
    im = np.swapaxes(im, 0, 2)
    im = np.swapaxes(im, 1, 2)
    im[0,:] -= 123.68
    im[1,:] -= 116.779
    im[2,:] -= 103.939
    im = np.expand_dims(im, 0)
    return im

def postprocess_img(im):
    im = im[0]
    im[0,:] += 123.68
    im[1,:] += 116.779
    im[2,:] += 103.939
    im = np.swapaxes(im, 0, 2)
    im = np.swapaxes(im, 0, 1)
    im[im<0] = 0
    im[im>255] = 255
    return im.astype(np.uint8)

class Maker():
    def __init__(self, model_prefix, output_shape, task):
        self.task = task
        s1, s0 = output_shape
        s0 = s0//32*32
        s1 = s1//32*32
        self.s0 = s0
        self.s1 = s1
        if task == 'texture':
            self.m = 5
            generator = symbol.generator_symbol(self.m, task)
            args = mx.nd.load('%s_args.nd'%model_prefix)
            for i in range(self.m):
                args['z_%d'%i] = mx.nd.zeros([1,3,s0/16*2**i,s1/16*2**i], mx.gpu())
        else:
            self.m = 5
            generator = symbol.generator_symbol(self.m, task)
            args = mx.nd.load('%s_args.nd'%model_prefix)
            for i in range(self.m):
                args['znoise_%d'%i] = mx.nd.zeros([1,3,s0/16*2**i,s1/16*2**i], mx.gpu())
                args['zim_%d'%i] = mx.nd.zeros([1,3,s0/16*2**i,s1/16*2**i], mx.gpu())
        self.gene_executor = generator.bind(ctx=mx.gpu(), args=args, aux_states=mx.nd.load('%s_auxs.nd'%model_prefix))

    def generate(self, save_path, content_path=''):
        if self.task == 'texture':
            for i in range(self.m):
                self.gene_executor.arg_dict['z_%d'%i][:] = mx.random.uniform(-128,128,[1,3,self.s0/16*2**i,self.s1/16*2**i])
            self.gene_executor.forward(is_train=True)
            out = self.gene_executor.outputs[0].asnumpy()
            im = postprocess_img(out)
            io.imsave(save_path, im)
        else:
            for i in range(self.m):
                self.gene_executor.arg_dict['znoise_%d'%i][:] = mx.random.uniform(-10,10,[1,3,self.s0/16*2**i,self.s1/16*2**i])
                self.gene_executor.arg_dict['zim_%d'%i][:] = preprocess_img(content_path, (self.s0/16*2**i,self.s1/16*2**i))
            self.gene_executor.forward(is_train=True)
            out = self.gene_executor.outputs[0].asnumpy()
            im = postprocess_img(out)
            io.imsave(save_path, im)

