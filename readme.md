# Beyond ImageNet Attack: Towards Crafting Adversarial Examples for Black-box Domains (ICLR'2022) 

This is the **Pytorch code** for our paper [Beyond ImageNet Attack: Towards Crafting Adversarial Examples for Black-box Domains](https://arxiv.org/pdf/2201.11528.pdf)).
In this paper, with only the knowledge of the ImageNet domain, we propose a Beyond ImageNet Attack (BIA) to investigate the transferability towards black-box domains (unknown classification tasks).

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
![images](https://github.com/qilong-zhang/Beyond-ImageNet-Attack/blob/main/images.png)
- Download the ImageNet training dataset.
  - [ImageNet](http://www.image-net.org/) Training Set.

- Download the testing dataset.
  - [ImageNet](http://www.image-net.org/) Validation Set. We also provide precomputed imagenet validation dataset in [here](https://github.com/Alibaba-AAIG/Beyond-ImageNet-Attack/releases/download/precomputed_imagenet_validation/val224_compressed.pkl) (download it to `imagenet` folder and then run `convert.py`, thanks [@aaron-xichen](https://github.com/aaron-xichen/pytorch-playground)).
  - [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
  - [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
  - [FGVC AirCraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)
  - CIFAR-10, CIFAR-100, STL-10 and SVHN can be automatically downloaded via `torchvision.dataset`

Note: After downloading CUB-200-2011, Standford Cars and FGVC Aircraft, you should set the "self.rawdata_root" (DCL_finegrained/config.py: lines 59-75) to your saved path.

## Target model
The checkpoint of target model should be put into `model` folder.
- CUB-200-2011, Stanford Cars and FGVC AirCraft can be downloaded from [here](https://github.com/Alibaba-AAIG/Beyond-ImageNet-Attack/releases/download/Pretrained_DCL_model/model.zip).
- CIFAR-10, CIFAR-100, STL-10 and SVHN can be automatically downloaded. 
- ImageNet pre-trained models are available at [torchvision](https://pytorch.org/vision/stable/models.html). 
  
## Pretrained-Generators
![framework](https://github.com/qilong-zhang/Beyond-ImageNet-Attack/blob/main/framework.png)
Adversarial generators are trained against following four ImageNet pre-trained models.
* VGG19
* VGG16
* ResNet152
* DenseNet169

After finishing training, the resulting generator will be put into `saved_models` folder. You can also download our pretrained-generator from [here](https://github.com/Alibaba-AAIG/Beyond-ImageNet-Attack/releases/download/pretrained_models/saved_models.zip).

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


## Citing this work

If you find this work is useful in your research, please consider citing:

```
@inproceedings{Zhang2022BIA,
  author    = {Qilong Zhang and
               Xiaodan Li and
               Yuefeng Chen and
               Jingkuan Song and
               Lianli Gao and
               Yuan He and
               Hui Xue},
  title     = {Beyond ImageNet Attack: Towards Crafting Adversarial Examples for Black-box Domains},
  Booktitle = {International Conference on Learning Representations},
  year      = {2022}
}
```

## Acknowledge
Thank [@aaron-xichen](https://github.com/aaron-xichen/pytorch-playground), [@Muzammal-Naseer](https://github.com/Muzammal-Naseer/Cross-Domain-Perturbations) and [@JDAI-CV](https://github.com/JDAI-CV/DCL) for sharing their codes.
