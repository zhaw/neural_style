import os
import mxnet as mx
import numpy as np
import cPickle as pickle
from skimage import io, transform


def crop_img(im, shape):
    if type(im) == str:
        im = io.imread(im)
    factor = 1. * shape[0] / shape[1]
    if im.shape[0] > im.shape[1]*factor:
        c = (im.shape[0]-int(factor*im.shape[1])) / 2
        im = im[c:c+int(im.shape[1]*factor),:,:]
    else:
        c = (im.shape[1]-int(im.shape[0]/factor)) / 2
        im = im[:,c:c+int(im.shape[0]/factor),:]
    im = transform.resize(im, shape)
    im *= 255
    im = im.astype(np.uint8)
    return im

def preprocess_img(im, size):
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

def make_image(img, style, save_name, size=[]):
    with open('models/model%s.pkl'%style) as f:
        args, auxs, symbol = pickle.load(f)
    if type(img) == str:
        im = io.imread(img)
    else:
        im = img
    if not size:
        size = (im.shape[0]/32*32, im.shape[1]/32*32)
    for i in range(6):
        args['zim_%d'%i] = mx.nd.zeros([1,3,size[0]/2**(5-i), size[1]/2**(5-i)], mx.gpu())
        if 'znoise_%d'%i in args:
            args['znoise_%d'%i] = mx.random.uniform(-250, 250 ,[1,1,size[0]/2**(5-i),size[1]/2**(5-i)], mx.gpu())
    gene_executor = symbol.bind(ctx=mx.gpu(), args=args, aux_states=auxs)
    for i in range(6):
        gene_executor.arg_dict['zim_%d'%i][:] = preprocess_img(img, (size[0]/2**(5-i), size[1]/2**(5-i)))
    gene_executor.forward(is_train=True)
    out = gene_executor.outputs[0].asnumpy()
    im = postprocess_img(out)
    io.imsave(save_name, im)

def test():
    for img in os.listdir('test_pics'):
        for style in map(str, range(1,20)):
            print img, style
            make_image('test_pics/%s'%img, style, 'out/%s_%s'%(style, img))

def test_small():
    for img in os.listdir('test_pics'):
        for style in map(str, range(1,20)):
            print img, style
            make_image('test_pics/%s'%img, style, 'out/small_%s_%s'%(style, img), (512,512))
