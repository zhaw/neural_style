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
>> maker = make_image.Maker('models/13', (1024, 768))
>> maker.generate('output.jpg', '../images/tubingen.jpg')
```
will output 
![alt text](https://github.com/zhaw/neural_style/blob/master/images/fastmrf13.jpg)


## Train New Model
The basic idea is that we get some styled image using optimization based method first. Then we train a generative network to do it.
If you want to train new model, edit train.py, modify COCOPATH at line31. Then, for example, use this command ```python train.py --style-image path_to_style_image --content-weight 1e-1 3e-1 3e-1 --style-weight 1 1 1 --tv-weight 1e-5 --num-image 200 --epochs 200 --style-size 512 512 --lr 3e3 --model-name path_to_save_model --num-res 3 --num-rotation 2 --num-scale 2 --stride 4 --patch-size 3``` to train models.
```--num-image``` specifies how many training data do we use. Other arguments mean the same thing as they are in folder mrf_cnn.


# Pretrained Models
I trained ~20 models, you can download them [here](https://pan.baidu.com/s/1skMHqYp). As you will see, some give good results and some don't.

