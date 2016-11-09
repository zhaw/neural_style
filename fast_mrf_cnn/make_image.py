import os
import time
import mxnet as mx
import numpy as np
import symbol
import cPickle as pickle
from matplotlib import pyplot as plt
from skimage import io, transform

VGGPATH = '../vgg19.params'

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
    def __init__(self, model_prefix, output_shape):
        vgg_symbol = symbol.descriptor_symbol(1)
        arg_names = vgg_symbol.list_arguments()
        arg_dict = {}
        pretrained = mx.nd.load(VGGPATH)
        for name in arg_names:
            if name == "data":
                continue
            key = "arg:" + name
            if key in pretrained:
                arg_dict[name] = pretrained[key].copyto(mx.gpu())
        del pretrained
        s1, s0 = output_shape
        s0 = s0//32*32
        s1 = s1//32*32
        self.s0 = s0
        self.s1 = s1
        arg_dict['data'] = mx.nd.zeros([1,3,s0,s1], mx.gpu())
        self.vgg_executor = vgg_symbol.bind(ctx=mx.gpu(), args=arg_dict)
        decoder = symbol.decoder_symbol()
        args = mx.nd.load('%s_decoder_args.nd'%model_prefix)
        auxs = mx.nd.load('%s_decoder_auxs.nd'%model_prefix)
        args['data'] = self.vgg_executor.outputs[0]
        self.deco_executor = decoder.bind(ctx=mx.gpu(), args=args, aux_states=auxs)

    def generate(self, save_path, content_path):
        self.vgg_executor.arg_dict['data'][:] = preprocess_img(content_path, (self.s0, self.s1))
        self.vgg_executor.forward()
        self.deco_executor.forward(is_train=True)
        out = self.deco_executor.outputs[0].asnumpy()
        im = postprocess_img(out)
        io.imsave(save_path, im)
