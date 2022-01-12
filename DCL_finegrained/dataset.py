from random import sample
from torch.utils.data import DataLoader
import torch
from torchvision import datasets, transforms
import os
import pandas as pd
from . import dataset_DCL
from . import config



def get_cub(batch_size, **kwargs):
    # args = parse_args()
    # args.data = 'CUB'
    # args.dataset = 'CUB'
    # Config = config.LoadConfig(args, 'test')
    data = 'CUB'
    dataset = 'CUB'
    swap_num = [7,7]
    backbone = 'resnet50'
    Config = config.LoadConfig(data, dataset, swap_num, backbone, 'test')
    anno = pd.read_csv(os.path.join(Config.anno_root, 'test.txt'),\
                                    sep=" ",\
                                    header=None,\
                                    names=['ImageName', 'label'])

    transformers = config.load_data_transformers(512, 448, [7,7])
    data_set = dataset_DCL.dataset(Config,
                       anno=anno,
                       swap=transformers["None"],
                       totensor=transformers['test_totensor'],
                       test=True)

    dataloader = torch.utils.data.DataLoader(data_set,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=8,
                                             collate_fn=dataset_DCL.collate_fn4test)

    return dataloader

def get_car(batch_size, **kwargs):
    # args = parse_args()
    # args.data = 'STCAR'
    # args.dataset = 'STCAR'
    # # args.data = data_name
    # Config = config.LoadConfig(args, 'test')
    data = 'STCAR'
    dataset = 'STCAR'
    swap_num = [7,7]
    backbone = 'resnet50'
    Config = config.LoadConfig(data, dataset, swap_num, backbone, 'test')
    anno = pd.read_csv(os.path.join(Config.anno_root, 'test.txt'),\
                                    sep=" ",\
                                    header=None,\
                                    names=['ImageName', 'label'])

    transformers = config.load_data_transformers(512, 448, [7,7])
    data_set = dataset_DCL.dataset(Config,
                       anno=anno,
                       swap=transformers["None"],
                       totensor=transformers['test_totensor'],
                       test=True)

    dataloader = torch.utils.data.DataLoader(data_set,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=8,
                                             collate_fn=dataset_DCL.collate_fn4test)

    return dataloader


def get_air(batch_size, **kwargs):
    # args = parse_args()
    # args.data = 'AIR'
    # args.dataset = 'AIR'
    # Config = config.LoadConfig(args, 'test')
    data = 'AIR'
    dataset = 'AIR'
    swap_num = [2,2]
    backbone = 'resnet50'
    Config = config.LoadConfig(data, dataset, swap_num, backbone, 'test')
    anno = pd.read_csv(os.path.join(Config.anno_root, 'test.txt'),\
                                    sep=" ",\
                                    header=None,\
                                    names=['ImageName', 'label'])

    transformers = config.load_data_transformers(512, 448, [2,2])
    data_set = dataset_DCL.dataset(Config,
                       anno=anno,
                       swap=transformers["None"],
                       totensor=transformers['test_totensor'],
                       test=True)

    dataloader = torch.utils.data.DataLoader(data_set,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=8,
                                             collate_fn=dataset_DCL.collate_fn4test)

    return dataloader


if __name__ == '__main__':
    data = get_cub(2)
    for img, label in data:
        print(img.shape)
        print(label.shape)