import cv2
import mxnet as mx
import cPickle as pickle
from PIL import Image
from make_image import crop_img, preprocess_img, postprocess_img
from images2gif import writeGif


def make_video(video_name, start, end, interval, style, save_name, size=[], batchsize=1):
    v = cv2.VideoCapture(video_name)
    if not size:
        size = (int(v.get(4))/32*32, int(v.get(3))/32*32)
    else:
        size = (size[0]/32*32, size[1]/32*32)
    with open('models/model%s.pkl'%style) as f:
        args, auxs, symbol = pickle.load(f)
    for i in range(6):
        args['zim_%d'%i] = mx.nd.zeros([batchsize,3,size[0]/2**(5-i), size[1]/2**(5-i)], mx.gpu())
        if 'znoise_%d'%i in args:
            args['znoise_%d'%i] = mx.random.uniform(-250, 250 ,[batchsize,1,size[0]/2**(5-i),size[1]/2**(5-i)], mx.gpu())
    gene_executor = symbol.bind(ctx=mx.gpu(), args=args, aux_states=auxs)
    fps = v.get(5)
    countdown = round(start*fps)
    to_write = []
    buf = []
    for i in range(int(end*fps)):
        print i
        ret, frame = v.read()
        countdown -= 1
        if countdown < 0 and i % interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buf.append(frame)
            if len(buf) == batchsize:
                for j in range(batchsize):
                    for ii in range(6):
                        gene_executor.arg_dict['zim_%d'%ii][j:j+1] = preprocess_img(buf[j], (size[0]/2**(5-ii), size[1]/2**(5-ii)))
                gene_executor.forward(is_train=True)
                out = gene_executor.outputs[0].asnumpy()
                for j in range(batchsize):
                    im = postprocess_img(out[j:j+1])
                    to_write.append(Image.fromarray(im))
                buf = []

    writeGif(save_name, to_write, 1./fps*interval)
    v.release()


