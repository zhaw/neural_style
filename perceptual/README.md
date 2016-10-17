# Usage 

## Use Trained Models
To use trained models to generate images, first load the model and specify the output shape.
```python
>> import make_image
>> maker = make_image.Maker(model_prefix, output_shape)
>> maker.generate(output_path, content_image_path)
```

For example,
```python
>> import make_image
>> maker = make_image.Maker('models/s4', (512, 512))
>> maker.generate('output.jpg', '../images/sunflower.jpg')
```
will output 
![alt text](https://github.com/zhaw/neural_style/blob/master/images/perceptual_result.jpg)


## Train New Model
First, edit train.py, modify MSCOCOPATH at line11. Then, use function ```train_style``` to train model. The meaning of parameters can be found in docstring.


# Pretrained Models
I trained ~20 models, you can download them [here](http://pan.baidu.com/s/1geZlH31). Notice that the boundary effect is very severe due to the zero padding in convolution layers.

