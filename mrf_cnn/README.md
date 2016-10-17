# Usage 

## Original Version

In train.py, I do the exact same as [paper](https://arxiv.org/abs/1601.04589) describes. If you want to use it, here is an example: ```python train.py --content-image ../images/sunflower.jpg --style-image path_to_your_style_image --content-weight 1e-3 --style-weight 10 1 --tv-weight 1e-4 --epochs 200 --size 1024 768 --style-size 512 512 --lr 1e1 --output path_to_save_output --num-res 3 --num-rotation 1 --num-scale 1 --stride 2 --patch-size 3```

## Modified Version

When transfering style, this method builds an image pyramid and optimize from the top down. Suppose the content image is im_c with shape 512x512 and style im_s also with shape 512x512, we first rescale them down to im_c0(128x128) and im_s0(128x128), use them to compute target content and target style and use im_c0 to initialize output0(128x128). After done optimization at 128x128 level, we move to 256x256 level. We rescale im_c and im_s to 256x256 and use them as target content and target style, and we upsample output0 to 256x256 as the initial value output1. Finally we do the same thing at 512x512 level and get the desired output. For more information, see [paper](https://arxiv.org/abs/1601.04589) section 5. One critical drawback of this method is that it can't handle high resolution image very well. When your image is 256x256, you extract 24x24 patches from it and re-arrange them to build a new image. This can give good result because your patch is big enough to act as a style regularizer. However, when your image is 1024x1024, your 24x24 patches are too small to influence the final result. 

To overcome this drawback, I modified the method a little bit. There are 2 changes. 1. When do image pyramid, we just rescale the content image. 2. Upsampled output from previous pyramid level is not only used as initial value, but also the content image. When use this modified version, content weight should be set higher. For example, ```python train2.py --content-image ../images/sunflower.jpg --style-image path_to_your_style_image --content-weight 1 --style-weight 10 1 --tv-weight 1e-4 --epochs 200 --size 1024 768 --style-size 512 512 --lr 1e1 --output path_to_save_output --num-res 3 --num-rotation 1 --num-scale 1 --stride 2 --patch-size 3```


## Meaning of Arguments

* content-image: path to content image
* style-image: path to style image
* content-weight: content weight
* style-weight: style weight for each layer
* tv-weight: tv loss weight
* epochs: optimization goes how many steps
* size: desired output shape
* style-size: shape that style image be resized to, this is very import because scale matters a lot in this model
* lr: learning rate
* num-res: depth of pyramid
* num-rotation: rotate style image how many times
* num-scale: scale style image how many times
* stride: stride when extract MRF patches
* patch-size: size of MRF patches

If there are pepper-noise-like bad pixel, increase tv-weight. If output image is too smooth, decrease tv-weight. If you don't have enough GPU memory, set num-rotation and num-scale to 0 and increase stride.


## Results
Results from the original version:
![alt text](https://github.com/zhaw/neural_style/blob/master/images/sunflower4a.jpg)
![alt text](https://github.com/zhaw/neural_style/blob/master/images/sunflower7a.jpg)
![alt text](https://github.com/zhaw/neural_style/blob/master/images/sunflower13a.jpg)
![alt text](https://github.com/zhaw/neural_style/blob/master/images/sunflower36a.jpg)
![alt text](https://github.com/zhaw/neural_style/blob/master/images/sunflower56a.jpg)
![alt text](https://github.com/zhaw/neural_style/blob/master/images/sunflower63a.jpg)
![alt text](https://github.com/zhaw/neural_style/blob/master/images/sunflower71a.jpg)

Results from the modified version:
![alt text](https://github.com/zhaw/neural_style/blob/master/images/sunflower4b.jpg)
![alt text](https://github.com/zhaw/neural_style/blob/master/images/sunflower7b.jpg)
![alt text](https://github.com/zhaw/neural_style/blob/master/images/sunflower13b.jpg)
![alt text](https://github.com/zhaw/neural_style/blob/master/images/sunflower36b.jpg)
![alt text](https://github.com/zhaw/neural_style/blob/master/images/sunflower56b.jpg)
![alt text](https://github.com/zhaw/neural_style/blob/master/images/sunflower63b.jpg)
![alt text](https://github.com/zhaw/neural_style/blob/master/images/sunflower71b.jpg)
