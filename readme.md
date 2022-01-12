# Beyond ImageNet Attack: Towards Crafting Adversarial Examples for Black-box Domains (ICLR 2022)

## Requirement
  - Python 3.7
  - Pytorch 1.8.0
  - torchvision 0.9.0
  - numpy 1.20.2
  - scipy 1.7.0
  - pandas 1.3.0
  - opencv-python 4.5.2.54 
  - joblib 0.14.1
  - Pillow 6.1

## Dataset
- Download the ImageNet training dataset.
  - [ImageNet](http://www.image-net.org/) Training Set.

- Download the testing dataset.
  - [ImageNet](http://www.image-net.org/) Validation Set (我们提供val224和val299).
  - [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
  - [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
  - [FGVC AirCraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)
  - CIFAR-10, CIFAR-100, STL-10 and SVHN can be automatically downloaded via `torchvision.dataset`

`Note: After downloading CUB-200-2011, Standford Cars and FGVC Aircraft, you should set the "self.rawdata_root" (DCL_finegrained/config.py: lines 59-75) to your saved path.`

## Target model
The checkpoint of target model should be put into `model` folder.
- CUB-200-2011, Stanford Cars and FGVC AirCraft can be downloaded from （我们提供）......
- CIFAR-10, CIFAR-100, STL-10 and SVHN can be automatically downloaded 
- ImageNet pre-trained models are available at [torchvision](https://pytorch.org/vision/stable/models.html) 
  
## Pretrained-Generators
Adversarial generators are trained against following four ImageNet pre-trained models.
* VGG19
* VGG16
* ResNet152
* DenseNet169

After finishing training, the resulting generator will be put into `saved_models` folder. You can also download our pretrained-generator from ...... to `saved_models` folder.

## Train
Train the generator using vanilla BIA (RN: False, DA: False)
```python
python train.py --model_type vgg16 --train_dir your_imagenet_path --RN False --DA False
```
`your_imagenet_path` is the path where you download the imagenet training set. 

## Evaluation
Evaluate the performance of vanilla BIA (RN: False, DA: False)
```python
python eval.py --model_type vgg16 --RN False --DA False
```

## Acknowledge
Thank [@aaron-xichen](https://github.com/aaron-xichen/pytorch-playground) and [@Muzammal-Naseer](https://github.com/Muzammal-Naseer/Cross-Domain-Perturbations) for sharing their codes.
