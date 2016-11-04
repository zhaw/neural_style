# Usage 

In run.py, I implemented [paper](https://arxiv.org/abs/1601.04589), with a little change. If you want to use it, here is an example: ```python run.py --content-image path_to_img --style-image path_to_image --content-weight 1e-1 3e-1 3e-1 --style-weight 1 1 1 --tv-weight 1e-5 --epochs 200 --size 1024 768 --style-size 512 512 --lr 3e3 --output path_to_save --num-res 3 --num-rotation 0 --num-scale 1 --stride 2 --patch-size 3 --noise 0```

Before you start, you need to download [VGG model](https://github.com/dmlc/web-data/raw/master/mxnet/neural-style/model/vgg19.params) and save it to the root of this repository.

## Meaning of Arguments

* content-image: path to content image
* style-image: path to style image
* content-weight: content weight for each layer. Number of layers is equal to num-res
* style-weight: style weight for each layer. Number of layers is equal to num-res
* tv-weight: tv loss weight
* epochs: optimization goes how many steps
* size: desired output shape
* style-size: shape that style image be resized to, this is very import because scale matters a lot in this model
* lr: learning rate
* num-res: depth of pyramid, see symbol.py for detail
* num-rotation: rotate style image how many times
* num-scale: scale style image how many times
* stride: stride when extract MRF patches
* patch-size: size of MRF patches
* noise: add noise to content image to make style stronger

If there are pepper-noise-like bad pixel, increase tv-weight. If output image is too smooth, decrease tv-weight. If you don't have enough GPU memory, set num-rotation and num-scale and num-res to 0 and increase stride. Make size smaller will also reduce GPU memory requirment.


## Results
Content image:
![alt text](https://github.com/zhaw/neural_style/blob/master/images/tubingen.jpg)

Style images:
![alt text](https://github.com/zhaw/neural_style/blob/master/images/image1.jpg)
![alt text](https://github.com/zhaw/neural_style/blob/master/images/image8.jpg)
![alt text](https://github.com/zhaw/neural_style/blob/master/images/image10.jpg)
![alt text](https://github.com/zhaw/neural_style/blob/master/images/image54.jpg)
![alt text](https://github.com/zhaw/neural_style/blob/master/images/image64.jpg)
![alt text](https://github.com/zhaw/neural_style/blob/master/images/image65.jpg)
![alt text](https://github.com/zhaw/neural_style/blob/master/images/font.jpg)
![alt text](https://github.com/zhaw/neural_style/blob/master/images/font.png)
![alt text](https://github.com/zhaw/neural_style/blob/master/images/formula.jpg)

Results:
![alt text](https://github.com/zhaw/neural_style/blob/master/images/out1.jpg)
![alt text](https://github.com/zhaw/neural_style/blob/master/images/out8.jpg)
![alt text](https://github.com/zhaw/neural_style/blob/master/images/out10.jpg)
![alt text](https://github.com/zhaw/neural_style/blob/master/images/out54.jpg)
![alt text](https://github.com/zhaw/neural_style/blob/master/images/out64.jpg)
![alt text](https://github.com/zhaw/neural_style/blob/master/images/out65.jpg)
![alt text](https://github.com/zhaw/neural_style/blob/master/images/outfont.jpg)
![alt text](https://github.com/zhaw/neural_style/blob/master/images/outfont2.jpg)
![alt text](https://github.com/zhaw/neural_style/blob/master/images/outformula.jpg)

Content image:
![alt text](https://github.com/zhaw/neural_style/blob/master/images/pitt.jpg)

Style images:
![alt text](https://github.com/zhaw/neural_style/blob/master/images/leaf.jpg)
![alt text](https://github.com/zhaw/neural_style/blob/master/images/leaf2.jpg)
![alt text](https://github.com/zhaw/neural_style/blob/master/images/bark.jpg)
![alt text](https://github.com/zhaw/neural_style/blob/master/images/sparkler.jpg)

Results:
![alt text](https://github.com/zhaw/neural_style/blob/master/images/outpittleaf.jpg)
![alt text](https://github.com/zhaw/neural_style/blob/master/images/outpittleaf2.jpg)
![alt text](https://github.com/zhaw/neural_style/blob/master/images/outpittbark.jpg)
![alt text](https://github.com/zhaw/neural_style/blob/master/images/outpittsparkler.jpg)


