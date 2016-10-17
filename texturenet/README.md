# Usage 

## Use Trained Models
To use trained models to generate images, first load the model and specify the output shape and task, task is either 'texture' or 'style'.
```python
>> import make_image
>> maker = make_image.Maker(model_prefix, output_shape, task)
>> maker.generate(output_path, content_image_path)
```

For example,
```python
>> import make_image
>> maker = make_image.Maker('models/fire', (512, 512), 'style')
>> maker.generate('output.jpg', '../images/sunflower.jpg')
```
will output 
![alt text](https://github.com/zhaw/neural_style/blob/master/images/texturenet_fire.jpg)


## Train New Model
First, edit train.py, modify MSCOCOPATH at line11. Then, use function ```train_style``` to train style model and ```train_texture``` to train texture model. The meaning of parameters can be found in docstring. Models are very hard to train, maybe I didn't implement well.
