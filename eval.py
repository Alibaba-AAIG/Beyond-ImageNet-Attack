import argparse
import os
from PIL.Image import ImagePointHandler
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
from model_layer import Vgg16_all_layer, Vgg19_all_layer,Res152_all_layer, Dense169_all_layer
import random
from generator import GeneratorResnet
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

parser = argparse.ArgumentParser(description='Transfer towards Black-box Domain')
parser.add_argument('--batch_size', type=int, default=16, help='Number of trainig samples/batch')
parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.0002, help='Initial learning rate for adam')
parser.add_argument('--eps', type=int, default=10, help='Perturbation Budget')
parser.add_argument('--model_type', type=str, default='vgg16',
                    help='Model against GAN is trained: vgg16, vgg19 res152, dense169')
parser.add_argument('--RN', type=lambda x: (str(x).lower() == 'true'), default=False, 
                    help='If true, activating the Random Normalization module in training phase')
parser.add_argument('--DA', type=lambda x: (str(x).lower() == 'true'), default=False, 
                    help='If true, activating the Domain-agnostic Attention module in training phase')
args = parser.parse_args()
print(args)
# Normalize (0-1)
eps = args.eps/255.0
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(0)
# GPU
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

####################
# Model
####################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.model_type == 'vgg16':
    model = Vgg16_all_layer.Vgg16()
    layer_idx = 16  # Maxpooling.3
elif args.model_type == 'vgg19':
    model = Vgg19_all_layer.Vgg19()
    layer_idx = 18  # Maxpooling.3
elif args.model_type == 'res152':
    model = Res152_all_layer.Resnet152()
    layer_idx = 5   # Conv3_8
elif args.model_type == 'dense169':
    model = Dense169_all_layer.Dense169()
    layer_idx = 6  # Denseblock.2
else:
    raise Exception('Please check the model_type')


if args.RN and args.DA:
    save_checkpoint_suffix = 'BIA+RN+DA'
elif args.RN:
    save_checkpoint_suffix = 'BIA+RN'
elif args.DA:
    save_checkpoint_suffix = 'BIA+DA'
else:
    save_checkpoint_suffix = 'BIA'  

model = model.to(device)
model.eval()

# Input dimensions
scale_size = 256
img_size = 224

# Generator
netG = GeneratorResnet().to(device)

# Optimizer
optimG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))

# Training Data
data_transform = transforms.Compose([
    transforms.Resize(scale_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
])

def default_normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - 0.485) / 0.229
    t[:, 1, :, :] = (t[:, 1, :, :] - 0.456) / 0.224
    t[:, 2, :, :] = (t[:, 2, :, :] - 0.406) / 0.225

    return t

def normalize(t, mean, std):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean) / std
    t[:, 1, :, :] = (t[:, 1, :, :] - mean) / std
    t[:, 2, :, :] = (t[:, 2, :, :] - mean) / std

    return t

train_dir = args.train_dir
train_set = datasets.ImageFolder(train_dir, data_transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
train_size = len(train_set)
print('Training data size:', train_size)

# Loss
criterion = nn.CrossEntropyLoss()

# Training
for epoch in range(args.epochs):
    running_loss = 0
    for i, (img, _) in enumerate(train_loader):
        img = img.to(device)
        netG.train()
        optimG.zero_grad()
        adv = netG(img)

        # Projection
        adv = torch.min(torch.max(adv, img - eps), img + eps)
        adv = torch.clamp(adv, 0.0, 1.0)

        # Saving adversarial examples
        flag = False
        if i <= 1000:
            if i % 100 == 0:
                flag = True
        else:
            if i % 2000 == 0:
                flag = True
        if flag:
            plt.subplot(121)
            plt.imshow(img[0,...].permute(1,2,0).detach().cpu())
            plt.axis('off')
            plt.subplot(122)
            plt.imshow(adv[0,...].permute(1,2,0).detach().cpu())
            plt.axis('off')
            save_path = 'output/{}'.format(args.model_type)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, '{}.png'.format(i)), bbox_inches='tight')

        if args.RN:
            mean = np.random.normal(0.50, 0.08)
            std = np.random.normal(0.75, 0.08)
            adv_out_slice = model(normalize(adv.clone(), mean, std))[layer_idx]
            img_out_slice = model(normalize(img.clone(), mean, std))[layer_idx]
        else:
            adv_out_slice = model(default_normalize(adv.clone()))[layer_idx]
            img_out_slice = model(default_normalize(img.clone()))[layer_idx]

        if args.DA:
            attention = abs(torch.mean(img_out_slice, dim=1, keepdim=True)).detach()
        else:
            attention = torch.ones(adv_out_slice.shape).cuda()

        loss = torch.cosine_similarity((adv_out_slice*attention).reshape(adv_out_slice.shape[0], -1), 
                            (img_out_slice*attention).reshape(img_out_slice.shape[0], -1)).mean()
        loss.backward()
        optimG.step()

        if i % 100 == 0:
            print('Epoch: {0} \t Batch: {1} \t loss: {2:.5f}'.format(epoch, i, running_loss / 100))
            running_loss = 0
        running_loss += abs(loss.item())

        # One epoch 
        if i % 80000 == 0 and i > 0:
            save_checkpoint_dir = 'saved_models/{}'.format(args.model_type)
            if not os.path.exists(save_checkpoint_dir):
                os.makedirs(save_checkpoint_dir)
            save_path = os.path.join(save_checkpoint_dir, 'netG_{}_{}.pth'.format(save_checkpoint_suffix, epoch))
            torch.save(netG.state_dict(), save_path)

