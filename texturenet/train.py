import time
import random
import os
import mxnet as mx
import numpy as np
np.set_printoptions(precision=2)
import symbol
from skimage import io, transform

VGGPATH = '../vgg19.params'
MSCOCOPATH = '/home/zw/dataset/mscoco'

def postprocess_img(im):
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

def get_gram_executor(out_shapes, weights=[1,1,1,1,1]):
    gram_executors = []
    for i in range(len(weights)):
        shape = out_shapes[i]
        data = mx.sym.Variable('gram_data')
        flat = mx.sym.Reshape(data, shape=(int(shape[1]), int(np.prod(shape[2:]))))
        gram = mx.sym.FullyConnected(flat, flat, no_bias=True, num_hidden=shape[1]) # data shape: batchsize*n_in, weight shape: n_out*n_in
        normed = gram/np.prod(shape[1:])/shape[1]*np.sqrt(weights[i])
        gram_executors.append(normed.bind(ctx=mx.gpu(), args={'gram_data':mx.nd.zeros(shape, mx.gpu())}, args_grad={'gram_data':mx.nd.zeros(shape, mx.gpu())}, grad_req='write'))
    return gram_executors

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
    return out.bind(ctx, args={"img": img, "kernel": kernel})

def init_executor(m, batch_size, weights=[1,2,1,2], style_layers=['relu1_1','relu2_1','relu3_1','relu4_1'], content_layer='relu4_2', task='style'):
    size = 8*2**m
    initializer = mx.init.Xavier(factor_type='in', magnitude=2.34)
#     initializer = mx.init.Normal(1e-8)
    descriptor = symbol.descriptor_symbol(content_layer=content_layer, style_layers=style_layers, task=task)
    arg_shapes, output_shapes, aux_shapes = descriptor.infer_shape(data=(batch_size, 3, size, size))
    arg_names = descriptor.list_arguments()
    arg_dict = dict(zip(arg_names, [mx.nd.zeros(shape, ctx=mx.gpu()) for shape in arg_shapes]))
    grad_dict = {"data": arg_dict["data"].copyto(mx.gpu())}
    pretrained = mx.nd.load(VGGPATH)
    for name in arg_names:
        if name == "data":
            continue
        key = "arg:" + name
        if key in pretrained:
            pretrained[key].copyto(arg_dict[name])
    desc_executor = descriptor.bind(ctx=mx.gpu(), args=arg_dict, args_grad=grad_dict, grad_req='write')
    gram_executors = get_gram_executor(descriptor.infer_shape(data=(1,3,8*2**m,8*2**m))[1], weights=weights)
    generator = symbol.generator_symbol(m, task)
    if task == 'texture':
        z_shape = dict([('z_%d'%i, (batch_size,3,16*2**i,16*2**i)) for i in range(m)])
    else:
        z_shape = dict(
                [('zim_%d'%i, (batch_size,3,16*2**i,16*2**i)) for i in range(m)]
                + [('znoise_%d'%i, (batch_size,3,16*2**i,16*2**i)) for i in range(m)]
                )
    arg_shapes, output_shapes, aux_shapes = generator.infer_shape(**z_shape)
    arg_names = generator.list_arguments()
    arg_dict = dict(zip(arg_names, [mx.nd.zeros(shape, ctx=mx.gpu()) for shape in arg_shapes]))
    aux_names = generator.list_auxiliary_states()
    aux_dict = dict(zip(aux_names, [mx.nd.zeros(shape, ctx=mx.gpu()) for shape in aux_shapes]))
    grad_dict = {}
    for k in arg_dict:
        if k.startswith('block') or k.startswith('join'):
            grad_dict[k] = arg_dict[k].copyto(mx.gpu())
    for name in arg_names:
        if not name.startswith('z'):
            initializer(name, arg_dict[name])
    gene_executor = generator.bind(ctx=mx.gpu(), args=arg_dict, args_grad=grad_dict, aux_states=aux_dict, grad_req='write')
    return desc_executor, gram_executors, gene_executor

def train_texture(img_path, model_prefix, style_layer=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'], weights=[1,2,1,2], max_epoch=2000, lr=1e-2):
    '''
    img_path:       path to texture image
    model_prefix:   model save path
    style_layer:    layer to use
    weights:        weights for these layers
    max_epoch:      train on how many samples
    lr:             learning rate
    '''
    m = 5
    size = 8*2**m
    batch_size = 1
    desc_executor, gram_executors, gene_executor = init_executor(m, batch_size, task='texture', weights=weights, style_layers=style_layer)
    optimizer = mx.optimizer.Adam(learning_rate=lr, wd=0)
    optim_states = []
    for i, var in enumerate(gene_executor.grad_dict):
        if not var.startswith('z'):
            optim_states.append(optimizer.create_state(i, gene_executor.arg_dict[var]))
        else:
            optim_states.append([])
    im = preprocess_img(img_path, size)
    target_grams = [mx.nd.zeros(x.outputs[0].shape, mx.gpu()) for x in gram_executors]
    im = mx.nd.array(im, mx.gpu())
    im.copyto(desc_executor.arg_dict['data'][:1])
    desc_executor.forward()
    gram_diff = [mx.nd.empty(x.outputs[0].shape, mx.gpu()) for x in gram_executors]
    gram_grad = [mx.nd.empty(x.shape, mx.gpu()) for x in desc_executor.outputs[:len(style_layer)]]
    for j in range(len(target_grams)):
        desc_executor.outputs[j][0:1].copyto(gram_executors[j].arg_dict['gram_data'])
        gram_executors[j].forward()
        target_grams[j][:] = gram_executors[j].outputs[0]
    old_loss = 0
    for epoch in range(max_epoch):
        if epoch in [249, 499, 999]:
            optimizer.lr /= 5
        if epoch % 20 == 0:
            print epoch,
        for j in range(m):
            for i in range(batch_size):
                gene_executor.arg_dict['z_%d'%j][i:i+1][:] = mx.random.uniform(-128,128,[1,3,16*2**j,16*2**j], mx.gpu())
        gene_executor.forward(is_train=True)
        gene_executor.outputs[0].copyto(desc_executor.arg_dict['data'])
        desc_executor.forward(is_train=True)
        loss = [0 for x in desc_executor.outputs]
        for i in range(batch_size):
            for j in range(len(style_layer)):
                desc_executor.outputs[j][i:i+1].copyto(gram_executors[j].arg_dict['gram_data'])
                gram_executors[j].forward(is_train=True)
                gram_diff[j] = gram_executors[j].outputs[0]-target_grams[j]
                gram_executors[j].backward(gram_diff[j])
                gram_grad[j][i:i+1] = gram_executors[j].grad_dict['gram_data'] / batch_size
                loss[j] += np.sum(np.square(gram_diff[j].asnumpy())) / batch_size
        old_loss = sum(loss)
        if epoch % 20 == 0:
            print 'loss', sum(loss), loss
        desc_executor.backward(gram_grad)
        gene_executor.backward(desc_executor.grad_dict['data'])
        for i, var in enumerate(gene_executor.grad_dict):
            if not var.startswith('z'):
                optimizer.update(i, gene_executor.arg_dict[var], gene_executor.grad_dict[var], optim_states[i])
        if epoch % 500 == 499:
            mx.nd.save('%s_args.nd'%model_prefix, gene_executor.arg_dict)
            mx.nd.save('%s_auxs.nd'%model_prefix, gene_executor.aux_dict)
    mx.nd.save('%s_args.nd'%model_prefix, gene_executor.arg_dict)
    mx.nd.save('%s_auxs.nd'%model_prefix, gene_executor.aux_dict)


def train_style(img_path, model_prefix, style_layer=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'], weights=[1,2,1,2], content_layer='relu4_2', alpha=1, tv_weight=1e-6, max_epoch=2000, lr=1e-2):
    '''
    img_path:       path to style image
    model_prefix:   model save path
    style_layer:    layer to use
    weights:        weights for these layers
    content_layer:  use which layer as content
    alpha:          content weight
    tv_weight:      tv_weight
    max_epoch:      train on how many images
    lr:             learning rate
    '''
    m = 5 # Paper says that m should be 6 when training style but I found that it's very hard to train when m=6
    size = 8*2**m
    batch_size = 1
    desc_executor, gram_executors, gene_executor = init_executor(m, batch_size, task='style', weights=weights, style_layers=style_layer)
    tv_grad_executor = get_tv_grad_executor(desc_executor.arg_dict['data'], mx.gpu(), tv_weight)
    optimizer = mx.optimizer.Adam(learning_rate=lr, wd=0)
    optim_states = []
    for i, var in enumerate(gene_executor.grad_dict):
        if not var.startswith('z'):
            optim_states.append(optimizer.create_state(i, gene_executor.arg_dict[var]))
        else:
            optim_states.append([])
    im = preprocess_img(img_path, size)
    target_grams = [mx.nd.zeros(x.outputs[0].shape, mx.gpu()) for x in gram_executors]
    target_content = mx.nd.zeros(desc_executor.outputs[-1].shape, mx.gpu())
    content_grad = mx.nd.zeros(desc_executor.outputs[-1].shape, mx.gpu())
    im = mx.nd.array(im, mx.gpu())
    im.copyto(desc_executor.arg_dict['data'][:1])
    desc_executor.forward()
    gram_diff = [mx.nd.empty(x.outputs[0].shape, mx.gpu()) for x in gram_executors]
    gram_grad = [mx.nd.empty(x.shape, mx.gpu()) for x in desc_executor.outputs[:len(style_layer)]]
    for j in range(len(target_grams)):
        desc_executor.outputs[j][0:1].copyto(gram_executors[j].arg_dict['gram_data'])
        gram_executors[j].forward()
        target_grams[j][:] = gram_executors[j].outputs[0]

    list_img = os.listdir(MSCOCOPATH)
    for epoch in range(max_epoch):
        if epoch in [249, 499, 999]:
            optimizer.lr /= 5
        if epoch % 20 == 0:
            print epoch,
        for j in range(m):
            for i in range(batch_size):
                gene_executor.arg_dict['znoise_%d'%j][i:i+1][:] = mx.random.uniform(-10,10,[1,3,16*2**j,16*2**j], mx.gpu())
        while True:
#            selected = random.sample(range(len(list_img)), batch_size)
            selected = random.sample(range(16), batch_size) # Paper point that too many training samples will cause divergence.
            try:
                for i in range(batch_size):
                    for j in range(m):
                        gene_executor.arg_dict['zim_%d'%j][i:i+1][:] = preprocess_img(os.path.join(MSCOCOPATH, list_img[selected[i]]), 16*2**j)
                    desc_executor.arg_dict['data'][i:i+1] = gene_executor.arg_dict['zim_%d'%(m-1)]
            except:
                continue
            break
        desc_executor.forward()
        target_content[:] = desc_executor.outputs[-1]
        gene_executor.forward(is_train=True)
        gene_executor.outputs[0].copyto(desc_executor.arg_dict['data'])
        tv_grad_executor.forward()
        desc_executor.forward(is_train=True)
        loss = [0 for x in desc_executor.outputs]
        for i in range(batch_size):
            for j in range(len(style_layer)):
                desc_executor.outputs[j][i:i+1].copyto(gram_executors[j].arg_dict['gram_data'])
                gram_executors[j].forward(is_train=True)
                gram_diff[j] = gram_executors[j].outputs[0]-target_grams[j]
                gram_executors[j].backward(gram_diff[j])
                gram_grad[j][i:i+1] = gram_executors[j].grad_dict['gram_data'] / batch_size
                loss[j] += np.sum(np.square(gram_diff[j].asnumpy())) / batch_size
        layer_shape = desc_executor.outputs[-1].shape
        layer_size = np.prod(layer_shape)
        loss[-1] = alpha * np.sum(np.square((desc_executor.outputs[-1]-target_content).asnumpy())) / layer_size / batch_size
        content_grad[:] = alpha * (desc_executor.outputs[-1]-target_content) / layer_size / batch_size
        if epoch % 20 == 0:
            print 'loss', sum(loss), loss
        desc_executor.backward(gram_grad+[content_grad])
        gene_executor.backward(desc_executor.grad_dict['data']+tv_grad_executor.outputs[0])
        for i, var in enumerate(gene_executor.grad_dict):
            if not var.startswith('z'):
                optimizer.update(i, gene_executor.arg_dict[var], gene_executor.grad_dict[var], optim_states[i])
        if epoch % 500 == 499:
            mx.nd.save('%s_args.nd'%model_prefix, gene_executor.arg_dict)
            mx.nd.save('%s_auxs.nd'%model_prefix, gene_executor.aux_dict)
    mx.nd.save('%s_args.nd'%model_prefix, gene_executor.arg_dict)
    mx.nd.save('%s_auxs.nd'%model_prefix, gene_executor.aux_dict)
