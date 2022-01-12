import torch.nn as nn
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo
from utee import misc
import torch
import os
from . import config
from . import LoadModel
# import argparse

print = misc.logger.info

# def parse_args():
#     parser = argparse.ArgumentParser(description='dcl parameters')
#     parser.add_argument('--data', dest='dataset',
#     default=None, type=str)
#     parser.add_argument('--backbone', dest='backbone',
#     default='resnet50', type=str)
#     parser.add_argument('--b', dest='batch_size',
#     default=16, type=int)
#     parser.add_argument('--nw', dest='num_workers',
#     default=16, type=int)
#     parser.add_argument('--ver', dest='version',
#     default='test', type=str)
#     parser.add_argument('--save', dest='resume',
#     default=None, type=str)
#     parser.add_argument('--size', dest='resize_resolution',
#     default=512, type=int)
#     parser.add_argument('--crop', dest='crop_resolution',
#     default=448, type=int)
#     parser.add_argument('--ss', dest='save_suffix',
#     default=None, type=str)
#     parser.add_argument('--acc_report', dest='acc_report',
#     action='store_true')
#     parser.add_argument('--swap_num', default=[7, 7],
#     nargs=2, metavar=('swap1', 'swap2'),
#     type=int, help='specify a range')
#     args = parser.parse_args()
#     return args


def CUB():
    # args = parse_args()
    data = 'CUB'
    dataset = 'CUB'
    swap_num = [7,7]
    backbone = 'resnet50'
    Config = config.LoadConfig(data, dataset, swap_num, backbone, 'test')
    Config.cls_2xmul = True
    model = LoadModel.MainModel(Config)
    model_dict=model.state_dict()
    pretrained_dict= torch.load('model/CUB_Res_87.35.pth')
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    
    # args = parse_args()
    data = 'CUB'
    dataset = 'CUB'
    backbone = 'senet154'
    Config = config.LoadConfig(data, dataset, swap_num, backbone, 'test')
    # Config = config.LoadConfig(args, 'test')
    Config.cls_2xmul = True
    model2 = LoadModel.MainModel(Config)
    model2_dict=model2.state_dict()
    pretrained_dict2= torch.load('model/CUB_SENet_86.81.pth')
    pretrained_dict2 = {k[7:]: v for k, v in pretrained_dict2.items() if k[7:] in model2_dict}
    model2_dict.update(pretrained_dict2)
    model2.load_state_dict(model2_dict)


    # args = parse_args()
    # args.data = 'CUB'
    # args.dataset = 'CUB'
    backbone = 'se_resnet101'
    Config = config.LoadConfig(data, dataset, swap_num, backbone, 'test')
    # Config = config.LoadConfig(args, 'test')
    Config.cls_2xmul = True
    model3 = LoadModel.MainModel(Config)
    model3_dict=model3.state_dict()
    pretrained_dict3= torch.load('model/CUB_SE_86.56.pth')
    pretrained_dict3 = {k[7:]: v for k, v in pretrained_dict3.items() if k[7:] in model3_dict}
    model3_dict.update(pretrained_dict3)
    model3.load_state_dict(model3_dict)

    return model, model2, model3

def CAR():
    # args = parse_args()
    # args.data = 'STCAR'
    # args.dataset = 'STCAR'
    # Config = config.LoadConfig(args, 'test')
    data = 'STCAR'
    dataset = 'STCAR'
    swap_num = [7,7]
    backbone = 'resnet50'
    Config = config.LoadConfig(data, dataset, swap_num, backbone, 'test')

    Config.cls_2xmul = True
    model = LoadModel.MainModel(Config)
    model_dict=model.state_dict()
    pretrained_dict= torch.load('model/STCAR_Res_94.35.pth')
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


    # args = parse_args()
    # args.data = 'STCAR'
    # args.dataset = 'STCAR'
    # args.backbone = 'senet154'
    # Config = config.LoadConfig(args, 'test')
    backbone = 'senet154'
    Config = config.LoadConfig(data, dataset, swap_num, backbone, 'test')
    Config.cls_2xmul = True
    model2 = LoadModel.MainModel(Config)
    model2_dict=model2.state_dict()
    pretrained_dict2= torch.load('model/STCAR_SENet_93.36.pth')
    pretrained_dict2 = {k[7:]: v for k, v in pretrained_dict2.items() if k[7:] in model2_dict}
    model2_dict.update(pretrained_dict2)
    model2.load_state_dict(model2_dict)

    # args = parse_args()
    # args.data = 'STCAR'
    # args.dataset = 'STCAR'
    # args.backbone = 'se_resnet101'
    # Config = config.LoadConfig(args, 'test')
    backbone = 'se_resnet101'
    Config = config.LoadConfig(data, dataset, swap_num, backbone, 'test')
    Config.cls_2xmul = True
    model3 = LoadModel.MainModel(Config)
    model3_dict=model3.state_dict()
    pretrained_dict3= torch.load('model/STCAR_SE_92.97.pth')
    pretrained_dict3 = {k[7:]: v for k, v in pretrained_dict3.items() if k[7:] in model3_dict}
    model3_dict.update(pretrained_dict3)
    model3.load_state_dict(model3_dict)

    return model, model2, model3


def AIR():
    # args = parse_args()
    # args.data = 'AIR'
    # args.dataset = 'AIR'
    # args.swap_num = [2,2]
    # Config = config.LoadConfig(args, 'test')
    data = 'AIR'
    dataset = 'AIR'
    swap_num = [2,2]
    backbone = 'resnet50'
    Config = config.LoadConfig(data, dataset, swap_num, backbone, 'test')
    Config.cls_2xmul = True
    model = LoadModel.MainModel(Config)
    model_dict=model.state_dict()
    pretrained_dict= torch.load('model/AIR_Res_92.23.pth')
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # args = parse_args()
    # args.data = 'AIR'
    # args.dataset = 'AIR'
    # args.swap_num = [2,2]
    # args.backbone = 'senet154'
    # Config = config.LoadConfig(args, 'test')
    backbone = 'senet154'
    Config = config.LoadConfig(data, dataset, swap_num, backbone, 'test')
    Config.cls_2xmul = True
    model2 = LoadModel.MainModel(Config)
    model2_dict=model2.state_dict()
    pretrained_dict2= torch.load('model/AIR_SENet_92.08.pth')
    pretrained_dict2 = {k[7:]: v for k, v in pretrained_dict2.items() if k[7:] in model2_dict}
    model2_dict.update(pretrained_dict2)
    model2.load_state_dict(model2_dict)


    # args = parse_args()
    # args.data = 'AIR'
    # args.dataset = 'AIR'
    # args.swap_num = [2,2]
    # args.backbone = 'se_resnet101'
    # Config = config.LoadConfig(args, 'test')
    backbone = 'se_resnet101'
    Config = config.LoadConfig(data, dataset, swap_num, backbone, 'test')
    Config.cls_2xmul = True
    model3 = LoadModel.MainModel(Config)
    model3_dict=model3.state_dict()
    pretrained_dict3= torch.load('model/AIR_SE_91.90.pth')
    pretrained_dict3 = {k[7:]: v for k, v in pretrained_dict3.items() if k[7:] in model3_dict}
    model3_dict.update(pretrained_dict3)
    model3.load_state_dict(model3_dict)

    return model, model2, model3