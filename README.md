# Neural Style
fast style transfer with mxnet, can do videos

## Setup
* Install skimage via pip: ```sudo pip install scikit-image```
* Install [CUDA](https://developer.nvidia.com/cuda-downloads)
* Install [OpenCV](http://opencv.org) (if you want to transfer video)
* Install [Mxnet](https://github.com/dmlc/mxnet)
* Clone this repo
* Download models from this [page](https://zhaw.github.io/neuralstyleio) and put them in ```models``` folder.

## Usage
* Open python console
```python 
>> import make_image 
>> make_image.make_image(path_to_image, style_name, output_path)
```
This will keep the image with its original size. For example, if you downloaded `model9.pkl`, `make_image.make_image('test_pics/city.jpg', '9', 'out/9_city.jpg')` will give you something like this ![alt text](https://raw.githubusercontent.com/zhaw/neuralstyleio/gh-pages/out/9_city.jpg)

If your GPU memory is limited, you can restrict the size of the input image to a fixed value. For example, `make_image.make_image('test_pics/city.jpg', '9', 'out/small_9_city.jpg', (512, 512))` will first crop and resize the input image to 512x512, and give you some thing like this ![alt text](https://raw.githubusercontent.com/zhaw/neuralstyleio/gh-pages/out/small_9_city.jpg)

For more result, see this [gallary](https://zhaw.github.io/neuralstyleio) or [this](https://github.com/zhaw/neural_style/tree/pics)

## Transfer Video
* Open python console
```python
>> import make_video
>> make_video.make_video(video_name, start_time, end_time, frame_interval, style_name, output_name)
```
For example, if ```myvideo.mp4``` is a 24fps video file, ```make_video.make_video('myvideo.mp4', 10, 30, 3, '5', 'out.gif')``` will select the 10sec~30sec part of the video, and the output file will be in 8fps. If you want to save to video file format, you have to install [ffmpeg](http://ffmpeg.org/) and build opencv with it. Or you can just save with GIF and convert it with other software. This is the [website](https://cloudconvert.com/) I use for converting GIF to MP4.

Here is an example result. It's small and low-fps because otherwise the GIF file will go larger than 50MB. ![alt text](https://raw.githubusercontent.com/zhaw/neuralstyleio/gh-pages/gifs/small.gif)

High resolution result with higher fps can be downloaded [here](https://raw.githubusercontent.com/zhaw/neuralstyleio/gh-pages/gifs/1.mp4) and [here](https://raw.githubusercontent.com/zhaw/neuralstyleio/gh-pages/gifs/15.mp4).


## Speed and Memory
Image with normal size should be transfered in 1~2 second. It takes 1G GPU memory to transform a 512x512 image. It takes 4.5G GPU memory to transform a 1920x1080 image. Some models are simpler than others so they use less GPU memory. 

It takes me 50 seconds to transfer the 30 seconds 800x352 8fps video above.

Please notice that the first run will be very slow because it may take a lot of time for mxnet to allocate memory and that's why I recommand using a python console so the GPU memory can be reused. 

## Reference
* This repo is based on [texture net](http://arxiv.org/abs/1603.03417), with changes.
* This [example](https://github.com/dmlc/mxnet/tree/master/example/neural-style) is a good tutorial for building non-feedforward networks with mxnet.
* images2gif comes from [images2gif](https://pypi.python.org/pypi/images2gif) but the original version has error.
