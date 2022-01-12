import os
import pandas as pd
import torch

from transforms import transforms
# from utils.autoaugment import ImageNetPolicy

pretrained_model = {'None' : 'none',}

# transforms dict
def load_data_transformers(resize_reso=512, crop_reso=448, swap_num=[7, 7]):
    center_resize = 600
    Normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    data_transforms = {
       	'swap': transforms.Compose([
            transforms.Randomswap((swap_num[0], swap_num[1])),
        ]),
        'common_aug': transforms.Compose([
            transforms.Resize((resize_reso, resize_reso)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomCrop((crop_reso,crop_reso)),
            transforms.RandomHorizontalFlip(),
        ]),
        'train_totensor': transforms.Compose([
            transforms.Resize((crop_reso, crop_reso)),
            # ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val_totensor': transforms.Compose([
            transforms.Resize((crop_reso, crop_reso)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'test_totensor': transforms.Compose([
            transforms.Resize((resize_reso, resize_reso)),
            transforms.CenterCrop((crop_reso, crop_reso)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'None': None,
    }
    return data_transforms


class LoadConfig(object):
    def __init__(self, data, dataset, swap_num, backbone, version):
        if version == 'train':
            get_list = ['train', 'val']
        elif version == 'val':
            get_list = ['val']
        elif version == 'test':
            get_list = ['test']
        else:
            raise Exception("train/val/test ???\n")

        # please downloading the dataset first, and put them to the following path.
        if dataset == 'CUB':
            self.dataset = dataset
            self.rawdata_root = '/mnt/hdd1/zhangqilong/CUB_200_2011/images'
            self.anno_root = 'DCL_finegrained/datasets/CUB'
            self.numcls = 200
        elif dataset == 'STCAR':
            self.dataset = dataset
            self.rawdata_root = '/mnt/hdd1/zhangqilong/Cars/car_ims'
            self.anno_root = 'DCL_finegrained/datasets/STCAR'
            self.numcls = 196
        elif dataset == 'AIR':
            self.dataset = dataset
            self.rawdata_root = '/mnt/hdd1/zhangqilong/fgvc-aircraft-2013b/data/images'
            self.anno_root = 'DCL_finegrained/datasets/AIR'
            self.numcls = 100
        else:
            raise Exception('dataset not defined ???')

        # annotation file organized as :
        # path/image_name cls_num\n

        if 'train' in get_list:
             self.train_anno = pd.read_csv(os.path.join(self.anno_root, 'train.txt'),\
                                           sep=" ",\
                                           header=None,\
                                           names=['ImageName', 'label'])

        if 'val' in get_list:
            self.val_anno = pd.read_csv(os.path.join(self.anno_root, 'test.txt'),\
                                           sep=" ",\
                                           header=None,\
                                           names=['ImageName', 'label'])

        if 'test' in get_list:
            self.test_anno = pd.read_csv(os.path.join(self.anno_root, 'test.txt'),\
                                           sep=" ",\
                                           header=None,\
                                           names=['ImageName', 'label'])

        self.swap_num = swap_num

        self.save_dir = './net_model'
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.backbone = backbone

        self.use_dcl = True
        self.use_backbone = False if self.use_dcl else True
        self.use_Asoftmax = False
        self.use_focal_loss = False
        self.use_fpn = False
        self.use_hier = False

        self.weighted_sample = False
        self.cls_2 = True
        self.cls_2xmul = False

        self.log_folder = './logs'
        if not os.path.exists(self.log_folder):
            os.mkdir(self.log_folder)




