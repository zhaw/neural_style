import os
import time
import mxnet as mx
import numpy as np
import symbol
import cPickle as pickle
from matplotlib import pyplot as plt
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
    new_img = np.zeros([size[0]+128,size[1]+128,3])
    new_img[64:-64,64:-64,:] = im
    new_img[:64,:,:] = new_img[128:64:-1,:,:]
    new_img[-64:,:,:] = new_img[-64:-128:-1,:,:]
    new_img[:,:64,:] = new_img[:,128:64:-1,:]
    new_img[:,-64:,:] = new_img[:,-64:-128:-1,:]
    new_img = new_img.astype(np.float32)
    new_img = np.swapaxes(new_img, 0, 2)
    new_img = np.swapaxes(new_img, 1, 2)
    new_img[0,:] -= 123.68
    new_img[1,:] -= 116.779
    new_img[2,:] -= 103.939
    new_img = np.expand_dims(new_img, 0)
    return new_img 

def postprocess_img(im):
    im = im[0]
    im[0,:] += 123.68
    im[1,:] += 116.779
    im[2,:] += 103.939
    im = np.swapaxes(im, 0, 2)
    im = np.swapaxes(im, 0, 1)
    im[im<0] = 0
    im[im>255] = 255
    return im[64:-64,64:-64,:].astype(np.uint8)

class Maker():
    def __init__(self, model_prefix, output_shape):
        s1, s0 = output_shape
        s0 = s0//32*32
        s1 = s1//32*32
        self.s0 = s0
        self.s1 = s1
        generator = symbol.generator_symbol()
        args = mx.nd.load('%s_args.nd'%model_prefix)
        auxs = mx.nd.load('%s_auxs.nd'%model_prefix)
        args['data'] = mx.nd.zeros([1,3,s0+128,s1+128], mx.gpu())
        self.gene_executor = generator.bind(ctx=mx.gpu(), args=args, aux_states=auxs)

    def generate(self, save_path, content_path):
        self.gene_executor.arg_dict['data'][:] = preprocess_img(content_path, (self.s0, self.s1))
        self.gene_executor.forward(is_train=True)
        out = self.gene_executor.outputs[0].asnumpy()
        im = postprocess_img(out)
        io.imsave(save_path, im)
