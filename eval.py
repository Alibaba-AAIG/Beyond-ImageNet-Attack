from DCL_finegrained import model
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torchvision
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from utee import selector
from tqdm import tqdm
from utee.Normalize import Normalize
from loader_checkpoint import *
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

parser = argparse.ArgumentParser(description='Transfer towards Black-box Domain')
parser.add_argument('--epochs', type=int, default=0, help='Which Saving Instance to Evaluate')
parser.add_argument('--model_type', type=str, default= 'vgg16',  help ='Model against GAN is trained: vgg16, vgg19, res152, dense169')
parser.add_argument('--eps', type=int, default=10, help='Perturbation Budget')
parser.add_argument('--RN', type=lambda x: (str(x).lower() == 'true'), default=False, 
                    help='If true, activating the Random Normalization module in training phase')
parser.add_argument('--DA', type=lambda x: (str(x).lower() == 'true'), default=False, 
                    help='If true, activating the Domain-agnostic Attention module in training phase')
args = parser.parse_args()
print(args)

# pdb.set_trace()
if args.RN and args.DA:
    save_checkpoint_suffix = 'BIA+RN+DA'
elif args.RN:
    save_checkpoint_suffix = 'BIA+RN'
elif args.DA:
    save_checkpoint_suffix = 'BIA+DA'
else:
    save_checkpoint_suffix = 'BIA'  


# Normalize (0-1)
eps = 10.0/255

for domain in ['cifar10', 'cifar100', 'stl10', 'svhn', 'dcl_cub', 'dcl_car', 'dcl_air', 'imagenet','imagenet_incv3'][1:2]:
    print('='*30, '{}'.format(domain), '='*30)  
    if domain[:3] == 'dcl':
        batch_size = 6 
        if domain == 'dcl_cub':
            numcls = 200
        elif domain == 'dcl_car':
            numcls = 196    
        elif domain == 'dcl_air':
            numcls = 100
    elif domain == 'imagenet_incv3':
        batch_size = 16 
    elif domain == 'imagenet': 
        batch_size = 32 
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        model_vgg16 = nn.Sequential(Normalize(mean,std),torchvision.models.vgg16(pretrained=True)).cuda().eval()
        model_vgg19 = nn.Sequential(Normalize(mean,std),torchvision.models.vgg19(pretrained=True)).cuda().eval()
        model_res50 = nn.Sequential(Normalize(mean,std),torchvision.models.resnet50(pretrained=True)).cuda().eval()
        model_res152 = nn.Sequential(Normalize(mean,std),torchvision.models.resnet152(pretrained=True)).cuda().eval()
        model_dense121 = nn.Sequential(Normalize(mean,std),torchvision.models.densenet121(pretrained=True)).cuda().eval()
        model_dense169 = nn.Sequential(Normalize(mean,std),torchvision.models.densenet169(pretrained=True)).cuda().eval()
    else:
        batch_size = 128

    if domain == 'imagenet':
        ds_fetcher, is_imagenet = selector.select(domain)
    elif domain[:3] == 'dcl':
        model_res50, model_senet, model_seres101, ds_fetcher, is_imagenet = selector.select(domain)
        acc_res50, clean_res50, acc_senet, clean_senet, acc_seres101, clean_seres101 = 0,0,0,0,0,0
    else:
        model_raw, ds_fetcher, is_imagenet = selector.select(domain)

    if domain[-5:] == 'incv3':
        ds_val = ds_fetcher(batch_size=batch_size, input_size=299, train=False, val=True)
        data_length = len(ds_fetcher(batch_size=1, train=False, val=True))
    else:
        ds_val = ds_fetcher(batch_size=batch_size, train=False, val=True)
        data_length = len(ds_fetcher(batch_size=1, train=False, val=True))


    print('data length: ', data_length)
    clean_vgg16, clean_vgg19, clean_res50, clean_res152, clean_dense121, clean_dense169 = 0, 0, 0, 0, 0, 0
    acc_vgg16, acc_vgg19, acc_res50, acc_res152, acc_dense121, acc_dense169 = 0, 0, 0, 0, 0, 0
    clean, accuracy = 0, 0
    
    netG = load_gan(args, domain)
    netG = nn.DataParallel(netG).cuda().eval()


    for i, data_val in tqdm(enumerate(ds_val)):
        img, label = data_val

        img =  Variable(torch.FloatTensor(img)).cuda()
        label = Variable(torch.from_numpy(np.array(label)).long().cuda())
        adv = netG(img)
 
        adv = torch.min(torch.max(adv, img - eps), img + eps)
        adv = torch.clamp(adv, 0.0, 1.0)


        with torch.no_grad():
            if domain == 'imagenet':
                clean_vgg16 += torch.sum(torch.argmax(model_vgg16(img), dim = 1) == label.cuda())
                acc_vgg16 += torch.sum(torch.argmax(model_vgg16(adv), dim = 1) == label.cuda()) 

                clean_vgg19 += torch.sum(torch.argmax(model_vgg19(img), dim = 1) == label.cuda())
                acc_vgg19 += torch.sum(torch.argmax(model_vgg19(adv), dim = 1) == label.cuda())

                clean_res50 += torch.sum(torch.argmax(model_res50(img), dim = 1) == label.cuda())
                acc_res50 += torch.sum(torch.argmax(model_res50(adv), dim = 1) == label.cuda())
                
                clean_res152 += torch.sum(torch.argmax(model_res152(img), dim = 1) == label.cuda())
                acc_res152 += torch.sum(torch.argmax(model_res152(adv), dim = 1) == label.cuda())

                clean_dense121 += torch.sum(torch.argmax(model_dense121(img), dim = 1) == label.cuda())
                acc_dense121 += torch.sum(torch.argmax(model_dense121(adv), dim = 1) == label.cuda())

                clean_dense169 += torch.sum(torch.argmax(model_dense169(img), dim = 1) == label.cuda())
                acc_dense169 += torch.sum(torch.argmax(model_dense169(adv), dim = 1) == label.cuda())  

            elif domain[:3] != 'dcl':
                clean += torch.sum(torch.argmax(model_raw(img), dim = 1) == label.cuda())
                accuracy += torch.sum(torch.argmax(model_raw(adv), dim = 1) == label.cuda())
            else:
                outputs = model_res50(adv)
                outputs_clean = model_res50(img)
                outputs_pred = outputs[0] + outputs[1][:,0:numcls] + outputs[1][:,numcls:2*numcls]
                outputs_pred_clean = outputs_clean[0] + outputs_clean[1][:,0:numcls] + outputs_clean[1][:,numcls:2*numcls]
                acc_res50 += torch.sum(torch.argmax(outputs_pred, dim = 1) == label.cuda())
                clean_res50 += torch.sum(torch.argmax(outputs_pred_clean, dim = 1) == label.cuda())

                outputs2 = model_senet(adv)
                outputs_clean2 = model_senet(img)
                outputs_pred2 = outputs2[0] + outputs2[1][:,0:numcls] + outputs2[1][:,numcls:2*numcls]
                outputs_pred_clean2 = outputs_clean2[0] + outputs_clean2[1][:,0:numcls] + outputs_clean2[1][:,numcls:2*numcls]
                acc_senet += torch.sum(torch.argmax(outputs_pred2, dim = 1) == label.cuda())
                clean_senet += torch.sum(torch.argmax(outputs_pred_clean2, dim = 1) == label.cuda())

                outputs3 = model_seres101(adv)
                outputs_clean3 = model_seres101(img)
                outputs_pred3 = outputs3[0] + outputs3[1][:,0:numcls] + outputs3[1][:,numcls:2*numcls]
                outputs_pred_clean3 = outputs_clean3[0] + outputs_clean3[1][:,0:numcls] + outputs_clean3[1][:,numcls:2*numcls]
                acc_seres101 += torch.sum(torch.argmax(outputs_pred3, dim = 1) == label.cuda())
                clean_seres101 += torch.sum(torch.argmax(outputs_pred_clean3, dim = 1) == label.cuda())
    

    if domain == 'imagenet':
        print('----------------vgg16----------------')
        print(acc_vgg16 / data_length)
        print(clean_vgg16 / data_length)
        print('----------------vgg19----------------')
        print(acc_vgg19 / data_length)
        print(clean_vgg19 / data_length)
        print('----------------res50----------------')
        print(acc_res50 / data_length)
        print(clean_res50 / data_length)      
        print('----------------res152----------------')
        print(acc_res152 / data_length)
        print(clean_res152 / data_length)
        print('----------------dense121----------------')
        print(acc_dense121 / data_length)
        print(clean_dense121 / data_length)
        print('----------------dense169----------------')
        print(acc_dense169 / data_length)
        print(clean_dense169 / data_length)
    elif domain[:3] == 'dcl':
        print('----------------backbone:res50----------------')
        print(acc_res50 / data_length)
        print(clean_res50 / data_length)
        print('----------------backbone:se-net----------------')
        print(acc_senet / data_length)
        print(clean_senet / data_length)
        print('----------------backbone:se-res101----------------')
        print(acc_seres101 / data_length)
        print(clean_seres101 / data_length)

    else:
        print(accuracy / data_length)
        print(clean / data_length)


