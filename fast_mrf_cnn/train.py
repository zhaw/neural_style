import time
import random
import os
import mxnet as mx
import numpy as np
np.set_printoptions(precision=2)
import argparse
import symbol

from skimage import io, transform, exposure, color

parser = argparse.ArgumentParser(description='mrf neural style')

parser.add_argument('--style-image', type=str)
parser.add_argument('--content-weight', nargs='+', type=float)
parser.add_argument('--style-weight', nargs='+', type=float)
parser.add_argument('--tv-weight', type=float)
parser.add_argument('--num-image', type=int)
parser.add_argument('--epochs', type=int)
parser.add_argument('--style-size', nargs='+', type=int)
parser.add_argument('--lr', type=float)
parser.add_argument('--model-name', type=str)
parser.add_argument('--num-res', type=int)
parser.add_argument('--num-rotation', type=int)
parser.add_argument('--num-scale', type=int)
parser.add_argument('--stride', type=int)
parser.add_argument('--patch-size', type=int)

args = parser.parse_args()
VGGPATH = '../vgg19.params'
COCOPATH = '/home/zw/dataset/mscoco'

try:
    os.mkdir(args.model_name)
    os.mkdir('%s/data'%args.model_name)
    os.mkdir('%s/output'%args.model_name)
except:
    pass

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
    if len(im.shape) == 2:
        im = color.gray2rgb(im)
    if im.shape[0]*size[1] > im.shape[1]*size[0]:
        c = (im.shape[0]-1.*im.shape[1]/size[1]*size[0]) / 2
        c = int(c)
        im = im[c:-(1+c),:,:]
    else:
        c = (im.shape[1]-1.*im.shape[0]/size[0]*size[1]) / 2
        c = int(c)
        im = im[:,c:-(1+c),:]
    im = transform.resize(im, size)
    im = exposure.equalize_adapthist(im, kernel_size=(16,16), clip_limit=0.01)
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

def get_mrf_executor(layer, patch_shape):
    patch_size = patch_shape[-1]
    data = mx.sym.Variable('conv')
    weight = mx.sym.Variable('weight')
    dist = mx.sym.Convolution(data=data, weight=weight, kernel=(patch_size, patch_size), num_filter=patch_shape[0], no_bias=True)
    dist_executor = dist.bind(args={'conv': layer, 'weight': mx.nd.zeros(patch_shape, mx.gpu())}, ctx=mx.gpu())
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


vgg_symbol = symbol.descriptor_symbol(args.num_res)
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
img = None
args.style_size[0] = args.style_size[0] // 4 * 4
args.style_size[1] = args.style_size[1] // 4 * 4
size = [512, 512]
args.style_size = args.style_size[::-1]
rotations = [15*i for i in range(-args.num_rotation, args.num_rotation+1)]
scales = [1.05**i for i in range(-args.num_scale, args.num_scale+1)]


# extract patches
style_img = crop_img(args.style_image, args.style_size) 
patches = [[] for i in range(args.num_res)]
patches_normed = []
for s in scales:
    scaled = transform.rescale(style_img, s)
    arg_dict['data'] = mx.nd.zeros([len(rotations),3,scaled.shape[0],scaled.shape[1]], mx.gpu())
    for r in range(len(rotations)):
        arg_dict['data'][r:r+1] = preprocess_img(transform.rotate(scaled, rotations[r], mode='reflect'))
    vgg_executor = vgg_symbol.bind(ctx=mx.gpu(), args=arg_dict, grad_req='null')
    vgg_executor.forward()
    for l in range(args.num_res):
        tmp = vgg_executor.outputs[l].asnumpy()
        for ii in range(0, vgg_executor.outputs[l].shape[2]-args.patch_size+1, args.stride):
            for jj in range(0, vgg_executor.outputs[l].shape[3]-args.patch_size+1, args.stride):
                for r in range(len(rotations)):
                    patches[l].append(tmp[r,:,ii:ii+args.patch_size,jj:jj+args.patch_size])
for l in range(args.num_res):
    patches[l] = np.array(patches[l])
    tmp = np.linalg.norm(np.reshape(patches[l], [patches[l].shape[0], np.prod(patches[l].shape[1:])]), axis=1)
    norm = np.reshape(tmp, [tmp.shape[0],1,1,1])
    patches_normed.append(patches[l]/norm)
    patches[l] = mx.nd.array(patches[l], mx.gpu())

arg_dict['data'] = mx.nd.zeros([1,3,size[0],size[1]], mx.gpu())
grad_dict = {"data": arg_dict["data"].copyto(mx.gpu())}
vgg_executor = vgg_symbol.bind(ctx=mx.gpu(), args=arg_dict, args_grad=grad_dict, grad_req='write')

tv_grad_executor = get_tv_grad_executor(vgg_executor.arg_dict['data'], mx.gpu(), args.tv_weight) 
optimizer = mx.optimizer.SGD(learning_rate=args.lr, wd=0e-0, momentum=0.9)

# get mrf executor
mrf_executors = []
target_patch = []
for l in range(args.num_res):
    mrf_executors.append(get_mrf_executor(vgg_executor.outputs[l], patches[l].shape))
    mrf_executors[l].arg_dict['weight'][:] = patches_normed[l]
# get assign executor
pcs = []
ass_executors = []
nns = []
for l in range(args.num_res):
    pc = np.zeros(vgg_executor.outputs[l].shape)
    for i1 in range(0, vgg_executor.outputs[l].shape[2]-args.patch_size+1):
        for i2 in range(0, vgg_executor.outputs[l].shape[3]-args.patch_size+1):
            pc[0,:,i1:i1+args.patch_size,i2:i2+args.patch_size] += 1
    pc = mx.nd.array(pc, mx.gpu())
    nn = mx.nd.zeros([vgg_executor.outputs[l].shape[2]-args.patch_size+1, vgg_executor.outputs[l].shape[3]-args.patch_size+1], mx.gpu())
    assign_symbol = symbol.assign_symbol()
    assign_executor = assign_symbol.bind(args={'source':patches[l], 'nn':nn}, ctx=mx.gpu())
    assign_executor.forward()
    pcs.append(pc)
    ass_executors.append(assign_executor)
    nns.append(nn)

img_list = os.listdir(COCOPATH)
for idx in range(args.num_image):
#    break
    optim_state = optimizer.create_state(0, arg_dict['data'])
    selected = np.random.randint(0, len(img_list))
    img = crop_img(os.path.join(COCOPATH, img_list[selected]), size)
    io.imsave('%s/data/image%dx.jpg'%(args.model_name, idx), img/255)
    img = preprocess_img(img)
    vgg_executor.arg_dict['data'][:] = img
    vgg_executor.forward()
    original_content = []
    for l in range(args.num_res):
        original_content.append(vgg_executor.outputs[l].copyto(mx.gpu()))
    for epoch in range(args.epochs):
        vgg_executor.forward(is_train=True)
        if epoch % 10 == 0:
            for l in range(args.num_res):
                mrf_executors[l].forward()
                nns[l][:] = mx.nd.argmax_channel(mrf_executors[l].outputs[0])[0]
                ass_executors[l].outputs[0][:] = 0
                ass_executors[l].forward()
                ass_executors[l].outputs[0][:] /= pcs[l]
                # compute target layer
                ass_executors[l].outputs[0][:] *= args.style_weight[l]
                ass_executors[l].outputs[0][:] += args.content_weight[l]*original_content[l]
                ass_executors[l].outputs[0][:] *= 1./(args.style_weight[l]+args.content_weight[l])
        tv_grad_executor.forward()
        if epoch > args.epochs - 30:
            for l in range(1,args.num_res):
                vgg_executor.outputs[l][:] = 0
            vgg_executor.outputs[0][:] -= ass_executors[0].outputs[0] # grad
            vgg_executor.outputs[0][:] *= (args.style_weight[0]+args.content_weight[0]) / np.prod(vgg_executor.outputs[0].shape)
        else:
            for l in range(args.num_res):
                vgg_executor.outputs[l][:] -= ass_executors[l].outputs[0] # grad
                vgg_executor.outputs[l][:] *= (args.style_weight[l]+args.content_weight[l]) / np.prod(vgg_executor.outputs[l].shape)

        vgg_executor.backward(vgg_executor.outputs)
        optimizer.update(0, vgg_executor.arg_dict['data'], vgg_executor.grad_dict['data']+tv_grad_executor.outputs[0], optim_state)

    img = postprocess_img(vgg_executor.arg_dict['data'].asnumpy())
    io.imsave('%s/data/image%d.jpg'%(args.model_name, idx), img)

# Train a generative network
vgg_symbol = symbol.descriptor_symbol(1)
vgg_executor = vgg_symbol.bind(ctx=mx.gpu(), args=arg_dict, args_grad=grad_dict, grad_req='write')
decoder = symbol.decoder_symbol()
arg_shapes, output_shapes, aux_shapes = decoder.infer_shape(data=vgg_executor.outputs[0].shape)
arg_names = decoder.list_arguments()
arg_dict = dict(zip(arg_names, [mx.nd.zeros(shape, ctx=mx.gpu()) for shape in arg_shapes]))
aux_names = decoder.list_auxiliary_states()
aux_dict = dict(zip(aux_names, [mx.nd.zeros(shape, ctx=mx.gpu()) for shape in aux_shapes]))
grad_dict = {}
for k in arg_dict:
    if k != 'data':
        grad_dict[k] = arg_dict[k].copyto(mx.gpu())
initializer = mx.init.Normal(1e-3)
for name in arg_names:
    if name != 'data':
        initializer(name, arg_dict[name])
deco_executor = decoder.bind(ctx=mx.gpu(), args=arg_dict, args_grad=grad_dict, aux_states=aux_dict, grad_req='write')


optimizer = mx.optimizer.SGD(learning_rate=1e-10, wd=0e-0, momentum=0.9)
optim_states = []
for i, var in enumerate(deco_executor.grad_dict):
    if var != 'data':
        optim_states.append(optimizer.create_state(i, deco_executor.arg_dict[var]))
    else:
        optim_states.append([])
img_grad = mx.nd.empty((1,3,512,512), mx.gpu())

argstosave = {}
auxstosave = {}

for k in deco_executor.arg_dict:
    if k != 'data':
        argstosave[k] = deco_executor.arg_dict[k]
for k in deco_executor.aux_dict:
    auxstosave[k] = deco_executor.aux_dict[k]

contents = []
targets = []
for idx in range(args.num_image):
    contents.append(crop_img('%s/data/image%dx.jpg'%(args.model_name, idx), [512,512]))
    targets.append(crop_img('%s/data/image%d.jpg'%(args.model_name, idx), [512,512]))

for idx in range(100*args.num_image):
    selected = np.random.randint(0, args.num_image)
    im_content = preprocess_img(contents[selected])
    im_target = preprocess_img(targets[selected])

    vgg_executor.arg_dict['data'][:] = im_target
    vgg_executor.forward()
    target_content = vgg_executor.outputs[0].copyto(mx.gpu())
    vgg_executor.arg_dict['data'][:] = im_content
    vgg_executor.forward()
    deco_executor.arg_dict['data'][:] = vgg_executor.outputs[0]
    deco_executor.forward(is_train=True)
    deco_executor.outputs[0].copyto(vgg_executor.arg_dict['data'])
    if idx % 250 == 249:
        io.imsave('%s/output/out%d.jpg'%(args.model_name, idx), postprocess_img(deco_executor.outputs[0].asnumpy()))
    tv_grad_executor.forward()
    vgg_executor.forward(is_train=True)
    layer_shape = vgg_executor.outputs[0].shape
    layer_size = np.prod(layer_shape)
    vgg_executor.backward([vgg_executor.outputs[0]-target_content])
    deco_executor.backward([vgg_executor.grad_dict['data']+tv_grad_executor.outputs[0]])

    for i, var in enumerate(deco_executor.grad_dict):
        if var != 'data':
            optimizer.update(i, deco_executor.arg_dict[var], deco_executor.grad_dict[var], optim_states[i])
    if idx % 500 == 499:
        mx.nd.save('%s/%s_decoder_args.nd'%(args.model_name, args.model_name), argstosave)
        mx.nd.save('%s/%s_decoder_auxs.nd'%(args.model_name, args.model_name), auxstosave)

mx.nd.save('%s/%s_decoder_args.nd'%(args.model_name, args.model_name), argstosave)
mx.nd.save('%s/%s_decoder_auxs.nd'%(args.model_name, args.model_name), auxstosave)
