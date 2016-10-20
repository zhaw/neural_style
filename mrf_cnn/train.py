import time
import random
import os
import mxnet as mx
import numpy as np
np.set_printoptions(precision=2)
import argparse
import symbol

from skimage import io, transform

parser = argparse.ArgumentParser(description='mrf neural style')

parser.add_argument('--content-image', type=str)
parser.add_argument('--style-image', type=str)
parser.add_argument('--content-weight', type=float)
parser.add_argument('--style-weight', nargs='+', type=float)
parser.add_argument('--tv-weight', type=float)
parser.add_argument('--epochs', type=int)
parser.add_argument('--size', nargs='+', type=int)
parser.add_argument('--style-size', nargs='+', type=int)
parser.add_argument('--lr', type=float)
parser.add_argument('--output', type=str)
parser.add_argument('--num-res', type=int)
parser.add_argument('--num-rotation', type=int)
parser.add_argument('--num-scale', type=int)
parser.add_argument('--stride', type=int)
parser.add_argument('--patch-size', type=int)

args = parser.parse_args()
VGGPATH = '../vgg19.params'

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

def preprocess_img(im):
    im = im.astype(np.float32)
    im = np.swapaxes(im, 0, 2)
    im = np.swapaxes(im, 1, 2)
    im[0,:] -= 123.68
    im[1,:] -= 116.779
    im[2,:] -= 103.939
    im = np.expand_dims(im, 0)
    return im

def get_mrf_executor(layer_shape, patch_shape):
    patch_size = patch_shape[-1]
    data = mx.sym.Variable('conv')
    weight = mx.sym.Variable('weight')
    dist = mx.sym.Convolution(data=data, weight=weight, kernel=(patch_size, patch_size), num_filter=patch_shape[0], no_bias=True)
    dist_executor = dist.bind(args={'conv': mx.nd.zeros(layer_shape, mx.gpu()), 'weight': mx.nd.zeros(patch_shape, mx.gpu())}, ctx=mx.gpu())
    return dist_executor

def get_tv_grad_executor(img, ctx, tv_weight):
    nchannel = img.shape[1]
    simg = mx.sym.Variable("img")
    skernel = mx.sym.Variable("kernel")
    channels = mx.sym.SliceChannel(simg, num_outputs=nchannel)
    out = mx.sym.Concat(*[
        mx.sym.Convolution(data=channels[i], weight=skernel,
                           num_filter=1,
                           kernel=(3, 3), pad=(1,1),
                           no_bias=True, stride=(1,1))
        for i in range(nchannel)])
    kernel = mx.nd.array(np.array([[0, -1, 0],
                                   [-1, 4, -1],
                                   [0, -1, 0]])
                         .reshape((1, 1, 3, 3)),
                         ctx) / 8.0
    out = out * tv_weight
    return out.bind(ctx, args={"img": img,
                               "kernel": kernel})


vgg_symbol = symbol.symbol()
arg_names = vgg_symbol.list_arguments()
arg_dict = {}
pretrained = mx.nd.load(VGGPATH)
for name in arg_names:
    if name == "data":
        continue
    key = "arg:" + name
    if key in pretrained:
        arg_dict[name] = pretrained[key].copyto(mx.gpu())
img = None
size = [args.size[1]//2**args.num_res, args.size[0]//2**args.num_res] 
style_size = [args.style_size[1]//2**args.num_res, args.style_size[0]//2**args.num_res]
args.epochs /= 2**args.num_res
args.lr /= 2**args.num_res
rotations = [15*i for i in range(-args.num_rotation, args.num_rotation+1)]
scales = [1.05**i for i in range(-args.num_scale, args.num_scale+1)]
for res in range(args.num_res):
    size[0] *= 2
    size[1] *= 2
    style_size[0] *= 2
    style_size[1] *= 2
    args.epochs *= 2
    args.lr *= 4
    if res > 0:
        args.stride *= 2
    if min(size) <= 64:
        continue
# extract patches
    style_img = crop_img(args.style_image, style_size) 
    patches0 = []
    patches1 = []
    for s in scales:
        scaled = transform.rescale(style_img, s)
        arg_dict['data'] = mx.nd.zeros([len(rotations),3,scaled.shape[0],scaled.shape[1]], mx.gpu())
        for r in range(len(rotations)):
            arg_dict['data'][r:r+1] = preprocess_img(transform.rotate(scaled, rotations[r]))
        vgg_executor = vgg_symbol.bind(ctx=mx.gpu(), args=arg_dict, grad_req='null')
        vgg_executor.forward()
        layer0 = vgg_executor.outputs[0].asnumpy()
        layer1 = vgg_executor.outputs[1].asnumpy()
        for ii in range(0, layer0.shape[2]-args.patch_size+1, args.stride):
            for jj in range(0, layer0.shape[3]-args.patch_size+1, args.stride):
                for r in range(len(rotations)):
                    patches0.append(layer0[r,:,ii:ii+args.patch_size,jj:jj+args.patch_size])
        for ii in range(0, layer1.shape[2]-args.patch_size+1, args.stride):
            for jj in range(0, layer1.shape[3]-args.patch_size+1, args.stride):
                for r in range(len(rotations)):
                    patches1.append(layer1[r,:,ii:ii+args.patch_size,jj:jj+args.patch_size])
    patches0 = np.array(patches0)
    patches1 = np.array(patches1)
    norm0 = np.linalg.norm(np.reshape(patches0, [patches0.shape[0], np.prod(patches0.shape[1:])]), axis=1)
    norm0 = np.reshape(norm0, [norm0.shape[0],1,1,1])
    norm1 = np.linalg.norm(np.reshape(patches1, [patches1.shape[0], np.prod(patches1.shape[1:])]), axis=1)
    norm1 = np.reshape(norm1, [norm1.shape[0],1,1,1])
    patches0_normed = patches0 / norm0
    patches1_normed = patches1 / norm1
# compute content
    content_img = crop_img(args.content_image, size)
    arg_dict['data'] = mx.nd.zeros([1,3,size[0],size[1]], mx.gpu())
    arg_dict['data'][:] = preprocess_img(content_img)
    vgg_executor = vgg_symbol.bind(ctx=mx.gpu(), args=arg_dict, grad_req='null')
    vgg_executor.forward()
    target_content = vgg_executor.outputs[-1].copyto(mx.gpu())
    content_grad = mx.nd.empty(vgg_executor.outputs[-1].shape, mx.gpu())
# get mrf executor
    mrf_executor0 = get_mrf_executor(vgg_executor.outputs[0].shape, patches0.shape)
    mrf_executor1 = get_mrf_executor(vgg_executor.outputs[1].shape, patches1.shape)
    mrf_grad0 = mx.nd.empty(vgg_executor.outputs[0].shape, mx.gpu())
    mrf_grad1 = mx.nd.empty(vgg_executor.outputs[1].shape, mx.gpu())
    mrf_executor0.arg_dict['weight'][:] = patches0_normed
    mrf_executor1.arg_dict['weight'][:] = patches1_normed
    vgg_executor.outputs[0].copyto(mrf_executor0.arg_dict['conv'])
    vgg_executor.outputs[1].copyto(mrf_executor1.arg_dict['conv'])
    mrf_executor0.forward()
    mrf_executor1.forward()
    nn0 = mx.nd.argmax_channel(mrf_executor0.outputs[0]).asnumpy()[0].astype(np.int)
    nn1 = mx.nd.argmax_channel(mrf_executor1.outputs[0]).asnumpy()[0].astype(np.int)

    target_patch0 = np.zeros([1]+list(vgg_executor.outputs[0].shape[1:]))
    target_patch1 = np.zeros([1]+list(vgg_executor.outputs[1].shape[1:]))
    count0 = np.zeros_like(target_patch0)
    count1 = np.zeros_like(target_patch1)
    for i1 in range(0, target_patch0.shape[2]-args.patch_size+1, 2):
        for i2 in range(0, target_patch0.shape[3]-args.patch_size+1, 2):
            target_patch0[0,:,i1:i1+args.patch_size,i2:i2+args.patch_size] += patches0[nn0[i1,i2],:,:,:]
            count0[0,:,i1:i1+args.patch_size,i2:i2+args.patch_size] += 1
    for i1 in range(0, target_patch1.shape[2]-args.patch_size+1, 2):
        for i2 in range(0, target_patch1.shape[3]-args.patch_size+1, 2):
            target_patch1[0,:,i1:i1+args.patch_size,i2:i2+args.patch_size] += patches1[nn1[i1,i2],:,:,:]
            count1[0,:,i1:i1+args.patch_size,i2:i2+args.patch_size] += 1
    count0[count0==0] = 1
    count1[count1==0] = 1
    target_patch0 /= count0
    target_patch1 /= count1
    target_patch0 = mx.nd.array(target_patch0, mx.gpu())
    target_patch1 = mx.nd.array(target_patch1, mx.gpu())
# optimize
    if img != None:
        img = transform.resize(img, (size[0],size[1]))*255
        img = preprocess_img(img)
    else:
        img = crop_img(args.content_image, size)
        img = preprocess_img(img)
#        img = mx.random.uniform(-10,10,[1,3,size[0],size[1]], mx.gpu())
    arg_dict['data'][:] = img
    grad_dict = {"data": arg_dict["data"].copyto(mx.gpu())}
    vgg_executor = vgg_symbol.bind(ctx=mx.gpu(), args=arg_dict, args_grad=grad_dict, grad_req='write')
    tv_grad_executor = get_tv_grad_executor(vgg_executor.arg_dict['data'], mx.gpu(), args.tv_weight) 
    optimizer = mx.optimizer.SGD(learning_rate=args.lr, wd=0e-0, momentum=0.9)
    optim_state = optimizer.create_state(0, arg_dict['data'])

    for epoch in range(args.epochs):
        loss = [0, 0, 0]
        vgg_executor.forward(is_train=True)
        tv_grad_executor.forward()
        mrf_grad0[:] = 2*args.style_weight[0] * (vgg_executor.outputs[0]-target_patch0) / np.prod(vgg_executor.outputs[0].shape)
        mrf_grad1[:] = 2*args.style_weight[1] * (vgg_executor.outputs[1]-target_patch1) / np.prod(vgg_executor.outputs[1].shape)
        content_grad[:] = 2*args.content_weight * (vgg_executor.outputs[-1]-target_content) / np.prod(vgg_executor.outputs[-1].shape)
        if epoch % 10 == 0:
            loss[0] = np.square(mrf_grad0.asnumpy()).sum() / (2*args.style_weight[0]) * np.prod(vgg_executor.outputs[0].shape)
            loss[1] = np.square(mrf_grad1.asnumpy()).sum() / (2*args.style_weight[1]) * np.prod(vgg_executor.outputs[1].shape)
            loss[2] = np.square(content_grad.asnumpy()).sum() / (2*args.content_weight) * np.prod(vgg_executor.outputs[-1].shape)
            print loss, sum(loss)
        vgg_executor.backward([mrf_grad0, mrf_grad1, content_grad])
        optimizer.update(0, vgg_executor.arg_dict['data'], vgg_executor.grad_dict['data']+tv_grad_executor.outputs[0], optim_state)
        if epoch % 50 == 49:
            img = postprocess_img(vgg_executor.arg_dict['data'].asnumpy())
            io.imsave('tmp%d_%d.jpg'%(res, epoch), img)

    img = postprocess_img(vgg_executor.arg_dict['data'].asnumpy())
    io.imsave('tmp%d.jpg'%res, img)
io.imsave(args.output, img)
    





