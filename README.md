# Neural Style
style transfer with mxnet

## Requirements 
* Install skimage via pip: ```sudo pip install scikit-image```
* Install [CUDA](https://developer.nvidia.com/cuda-downloads) and Cudnn
* Install [Mxnet](https://github.com/dmlc/mxnet), please set "USE_NVRTC = 1" in config.mk before compiling
* Download pretrained [VGG model](https://github.com/dmlc/web-data/raw/master/mxnet/neural-style/model/vgg19.params) and save it to the root of this repository.
* Download [MSCOCO](http://msvocds.blob.core.windows.net/coco2014/train2014.zip) dataset if you want to train models.


## Usage
Folder mrf_cnn implements [Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis](https://arxiv.org/abs/1601.04589). It is an optimization based method, can provide delicate result but is slow. [Original Torch version](https://github.com/chuanli11/CNNMRF)

Folder perceptual implements [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155). It trains a network to do optimization and is very fast. [Original Torch version](https://github.com/jcjohnson/fast-neural-style)

Folder texturenet implements [Texture Networks: Feed-forward Synthesis of Textures and Stylized Images](http://arxiv.org/abs/1603.03417). It also trains a network to do optimization. [Original Torch version](https://github.com/DmitryUlyanov/texture_nets)

Folder fast_mrf_cnn trains network to do optimization in mrf_cnn. It is very fast and can give result similar to but not as good as mrf_cnn.

Folder old_stuff contains some pretrained texture network models.
