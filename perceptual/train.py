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
#     im = im[0]
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


def get_gram_executor(out_shapes, weights=[1,1,1,1]):
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
    return out.bind(ctx, args={"img": img,
                               "kernel": kernel})


def init_executor(batch_size, weights=[1,1,1,1], style_layers=['relu1_1','relu2_1','relu3_1','relu4_1'], content_layer='relu4_2'):
    size = 256 
    initializer = mx.init.Normal(1e-8)
    descriptor = symbol.descriptor_symbol(content_layer=content_layer, style_layers=style_layers)
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
    gram_executors = get_gram_executor(descriptor.infer_shape(data=(1,3,size,size))[1], weights=weights)
    generator = symbol.generator_symbol()
    arg_shapes, output_shapes, aux_shapes = generator.infer_shape(data=(batch_size,3,size,size))
    arg_names = generator.list_arguments()
    arg_dict = dict(zip(arg_names, [mx.nd.zeros(shape, ctx=mx.gpu()) for shape in arg_shapes]))
    aux_names = generator.list_auxiliary_states()
    aux_dict = dict(zip(aux_names, [mx.nd.zeros(shape, ctx=mx.gpu()) for shape in aux_shapes]))
    grad_dict = {}
    for k in arg_dict:
        if k != 'data':
            grad_dict[k] = arg_dict[k].copyto(mx.gpu())
    for name in arg_names:
        if name != 'data':
            initializer(name, arg_dict[name])
    gene_executor = generator.bind(ctx=mx.gpu(), args=arg_dict, args_grad=grad_dict, aux_states=aux_dict, grad_req='write')
    return desc_executor, gram_executors, gene_executor


def train_style(alpha, model_prefix, img_path, weights=[1,4,1,4], tv_weight=1e-4, max_epoch=1000, lr=1e-2):
    '''
    alpha:          content weight
    model_prefix:   path to save model
    img_path:       path to style image
    weights:        weights of each style layer
    tv_weight:      tv weight
    max_epoch:      train on how many images
    lr:             lr
    '''
    size = 256 
    batch_size = 1 
    style_layer = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
    content_layer = 'relu3_1'
    desc_executor, gram_executors, gene_executor = init_executor(batch_size, weights=weights, style_layers=style_layer, content_layer=content_layer)
    tv_grad_executor = get_tv_grad_executor(desc_executor.arg_dict['data'], mx.gpu(), tv_weight) 
    optimizer = mx.optimizer.Adam(learning_rate=lr, wd=1e-4)
    optim_states = []
    for i, var in enumerate(gene_executor.grad_dict):
        if var != 'data':
            optim_states.append(optimizer.create_state(i, gene_executor.arg_dict[var]))
        else:
            optim_states.append([])
    im = preprocess_img(img_path, size)
    target_grams = [mx.nd.zeros(x.outputs[0].shape, mx.gpu()) for x in gram_executors]
    for xy in range(1):
        for flx in range(1):
            for fly in range(1):
                imt = np.zeros_like(im)
                imt[:] = im
                if xy:
                    imt = np.swapaxes(imt, 2, 3)
                if flx:
                    imt = imt[:,:,::-1,:]
                if fly:
                    imt = imt[:,:,:,::-1]
                imt = mx.nd.array(imt, mx.gpu())
                imt.copyto(desc_executor.arg_dict['data'][:1])
                desc_executor.forward()
                gram_diff = [mx.nd.empty(x.outputs[0].shape, mx.gpu()) for x in gram_executors]
                gram_grad = [mx.nd.empty(x.shape, mx.gpu()) for x in desc_executor.outputs[:len(style_layer)]]
                for j in range(len(target_grams)):
                    desc_executor.outputs[j][0:1].copyto(gram_executors[j].arg_dict['gram_data'])
                    gram_executors[j].forward()
                    target_grams[j][:] += gram_executors[j].outputs[0]
    for j in range(len(target_grams)):
        target_grams[j][:] /= 1
    
    target_content = mx.nd.empty(desc_executor.outputs[len(style_layer)].shape, mx.gpu())
    content_grad = mx.nd.empty(desc_executor.outputs[len(style_layer)].shape, mx.gpu())

    list_img = os.listdir(MSCOCOPATH)
    for epoch in range(max_epoch):
        if epoch % 20 == 0:
            print epoch,
        while True:
            selected = random.sample(range(len(list_img)), batch_size) 
            try:
                for i in range(batch_size):
                    desc_executor.arg_dict['data'][i:i+1] = preprocess_img(os.path.join(MSCOCOPATH, list_img[selected[i]]), size)
            except:
                continue
            break
        desc_executor.forward()
        target_content[:] = desc_executor.outputs[len(style_layer)]
        for i in range(batch_size):
            gene_executor.arg_dict['data'][i:i+1] = preprocess_img(os.path.join(MSCOCOPATH, list_img[selected[i]]), size)
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
        layer_shape = desc_executor.outputs[len(style_layer)].shape
        layer_size = np.prod(layer_shape)
        loss[len(style_layer)] += alpha*np.sum(np.square((desc_executor.outputs[len(style_layer)]-target_content).asnumpy()/np.sqrt(layer_size))) / batch_size
        content_grad = alpha*(desc_executor.outputs[len(style_layer)]-target_content) / layer_size / batch_size
        if epoch % 20 == 0:
            print 'loss', sum(loss), np.array(loss)
        desc_executor.backward(gram_grad+[content_grad])
        gene_executor.backward(desc_executor.grad_dict['data']+tv_grad_executor.outputs[0])
        for i, var in enumerate(gene_executor.grad_dict):
            if var != 'data':
                optimizer.update(i, gene_executor.arg_dict[var], gene_executor.grad_dict[var], optim_states[i])
        if epoch % 500 == 499:
            mx.nd.save('%s_args.nd'%model_prefix, gene_executor.arg_dict)
            mx.nd.save('%s_auxs.nd'%model_prefix, gene_executor.aux_dict)
    mx.nd.save('%s_args.nd'%model_prefix, gene_executor.arg_dict)
    mx.nd.save('%s_auxs.nd'%model_prefix, gene_executor.aux_dict)
