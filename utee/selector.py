import os
import torch.nn as nn
from .Normalize import Normalize
import torchvision.models as models

import pdb

known_models = [
    'cifar10', 'cifar100', # 32x32
    'stl10', # 96x96
    'svhn', # 32x32
    'dcl_cub_train',
    'dcl_cub_data',
    'dcl_cub',  # 448x448
    'dcl_air',  # 448x448
    'dcl_car',  # 448x448
    'imagenet', # 224x224
    'imagenet_incv3'  # 299x299
]
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def svhn(cuda=True, model_root=None):
    print("Building and initializing svhn parameters")
    from svhn import model, dataset
    m = nn.Sequential(Normalize([0.5, 0.5, 0.5], [0.5,0.5,0.5]), model.svhn(32, pretrained=os.path.join(model_root, 'svhn.pth')))
    if cuda:
        m = m.cuda()
    return m.eval(), dataset.get, False

def cifar10(cuda=True, model_root=None):
    print("Building and initializing cifar10 parameters")
    from cifar import model, dataset
    m = nn.Sequential(Normalize([0.5, 0.5, 0.5], [0.5,0.5,0.5]), model.cifar10(128, pretrained=os.path.join(model_root, 'cifar10.pth')))
    if cuda:
        m = m.cuda()
    return m.eval(), dataset.get10, False


def imagenet(cuda=True, model_root=None):
    print("Building and initializing imagenet parameters")
    from imagenet import dataset
    return dataset.get, True


def imagenet_incv3(cuda=True, model_root=None):
    print("Building and initializing imagenet parameters")
    from imagenet import dataset
    inc_v3 = nn.Sequential(Normalize(mean,std), models.inception_v3(pretrained=True, transform_input = False)).cuda().eval()
    return inc_v3, dataset.get, False



def cifar100(cuda=True, model_root=None):
    print("Building and initializing cifar100 parameters")
    from cifar import model, dataset
    m = nn.Sequential(Normalize([0.5, 0.5, 0.5], [0.5,0.5,0.5]), model.cifar100(128, pretrained=os.path.join(model_root, 'cifar100.pth')))
    if cuda:
        m = m.cuda()
    return m.eval(), dataset.get100, False


def stl10(cuda=True, model_root=None):
    print("Building and initializing stl10 parameters")
    from stl10 import model, dataset
    m = nn.Sequential(Normalize([0.5, 0.5, 0.5], [0.5,0.5,0.5]),model.stl10(32, pretrained=os.path.join(model_root, 'stl10.pth')))
    if cuda:
        m = m.cuda()
    return m.eval(), dataset.get, False

def dcl_cub(cuda=True, model_root=None):
    from DCL_finegrained import model, dataset
    # pdb.set_trace()
    cub_1, cub_2, cub_3 = model.CUB()
    m_1 = nn.Sequential(Normalize(mean,std), cub_1)
    m_2 = nn.Sequential(Normalize(mean,std), cub_2)
    m_3 = nn.Sequential(Normalize(mean,std), cub_3)
    if cuda:
        m_1 = m_1.cuda()
        m_2 = m_2.cuda()
        m_3 = m_3.cuda()
    # model = nn.DataParallel(model)
    # m = nn.parallel.DistributedDataParallel(m, device_ids=[0,1,2,3,4,5,6,7])
    # m = data_parallel.BalancedDataParallel(4, m, dim = 0)
    # m = nn.DataParallel(m)
    m_1.train(False)
    m_2.train(False)
    m_3.train(False)
    return m_1, m_2, m_3, dataset.get_cub, False

def dcl_cub_train(cuda=False, model_root=None):
    from DCL_finegrained import model, dataset
    # pdb.set_trace()
    cub_1, cub_2, cub_3 = model.CUB()
    # print('ssss')
    cub_1 = cub_1.cuda()
    cub_1.train(False)

    return cub_1

def dcl_cub_data(cuda=False, model_root=None):
    from DCL_finegrained import model, dataset
    # pdb.set_trace()
    # cub_1, cub_2, cub_3 = model.CUB()
    # cub_2 = cub_2.cuda()
    # cub_2.train(False)
    return dataset.get_cub

def dcl_car(cuda=True, model_root=None):
    from DCL_finegrained import model,dataset
    car_1, car_2, car_3 = model.CAR()
    m_1 = nn.Sequential(Normalize(mean,std), car_1)
    m_2 = nn.Sequential(Normalize(mean,std), car_2)
    m_3 = nn.Sequential(Normalize(mean,std), car_3)
    if cuda:
        m_1 = m_1.cuda()
        m_2 = m_2.cuda()
        m_3 = m_3.cuda()
    # model = nn.DataParallel(model)
    # m = nn.parallel.DistributedDataParallel(m, device_ids=[0,1,2,3,4,5,6,7])
    # m = data_parallel.BalancedDataParallel(4, m, dim = 0)
    # m = nn.DataParallel(m)
    m_1.train(False)
    m_2.train(False)
    m_3.train(False)
    return m_1, m_2, m_3, dataset.get_car, False

def dcl_air(cuda=True, model_root=None):
    from DCL_finegrained import model,dataset
    air_1, air_2, air_3 = model.AIR()
    m_1 = nn.Sequential(Normalize(mean,std), air_1)
    m_2 = nn.Sequential(Normalize(mean,std), air_2)
    m_3 = nn.Sequential(Normalize(mean,std), air_3)
    if cuda:
        m_1 = m_1.cuda()
        m_2 = m_2.cuda()
        m_3 = m_3.cuda()
    # model = nn.DataParallel(model)
    # m = nn.parallel.DistributedDataParallel(m, device_ids=[0,1,2,3,4,5,6,7])
    # m = data_parallel.BalancedDataParallel(4, m, dim = 0)
    # m = nn.DataParallel(m)
    m_1.train(False)
    m_2.train(False)
    m_3.train(False)
    return m_1, m_2, m_3, dataset.get_air, False


def select(model_name, **kwargs):
    assert model_name in known_models, model_name
    kwargs.setdefault('model_root', os.path.expanduser('~/.torch/models'))
    return eval('{}'.format(model_name))(**kwargs)



