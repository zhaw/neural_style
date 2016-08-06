# Neural Style
fast style transfer with mxnet

## Setup
* Install skimage via pip: ```sudo pip install scikit-image```
* Install [CUDA](https://developer.nvidia.com/cuda-downloads)
* Install [Mxnet](https://github.com/dmlc/mxnet)
* Clone this repo
* Download models from this [page](https://zhaw.github.io/neuralstyleio) and put them in ```models``` folder.

## Usage
* Open python console
```python 
>> import make_image 
>> make_image.make_image(path_to_image, style_name, output_path)
```
This will keep the image with its original size. For example, if you downloaded `model9.pkl`, `make_image.make_image('test_pics/city.jpg', '9', 'out/9_city.jpg')` will give you something like this ![alt text](../pics/9_city.jpg)

If your GPU memory is limited, you can restrict the size of the input image to a fixed value. For example, `make_image.make_image('test_pics/city.jpg', '9', 'out/small_9_city.jpg', (512, 512))` will first crop and resize the input image to 512x512, and give you some thing like this ![alt text](../pics/small_9_city.jpg)

For more result, see this [gallary](https://zhaw.github.io/neuralstyleio) or [this](https://github.com/zhaw/neural_style/tree/pics)

## Speed and Memory
Image with normal size should be transfered in 1~2 second. It takes 1.5G GPU memory to transform a 512x512 image. It takes 4.5G GPU memory to transform a 1920x1080 image. Some models are simpler than others so they use less GPU memory. 

Please notice that the first run will be very slow because it may take a lot of time for mxnet to allocate memory and that's why I recommand using a python console so the GPU memory can be reused. 

## Reference
* This repo is based on [texture net](http://arxiv.org/abs/1603.03417), with changes.
* This [example](https://github.com/dmlc/mxnet/tree/master/example/neural-style) is a good tutorial for building non-feedforward networks with mxnet.

