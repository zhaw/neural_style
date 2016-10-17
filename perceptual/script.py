import train

for i,st in enumerate(map(str, [1,6,7,11,12,16,17,19,21,36,38,47,48,52,67,69,72,73,78,83,84,86])):
    train.train_style(img_path='../../perceptual/image%s.jpg'%st, model_prefix='s%d'%i, alpha=2e-1, max_epoch=20000)
