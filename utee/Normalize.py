import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Normalize(nn.Module):

    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, input):
        size = input.size()
        x = input.clone()
        for i in range(size[1]):
            x[:, i] = (x[:, i] - self.mean[i]) / self.std[i]
        return x

class Normalize_one(nn.Module):
    def __init__(self, mean, std):
        super(Normalize_one, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, input):
        x = input.clone()
        x = (x - self.mean) / self.std
        return x

class Normalize_TF(nn.Module):

    def __init__(self):
        super(Normalize_TF, self).__init__()
        self.mean = [0.5,0.5,0.5]
        self.std = [0.5,0.5,0.5]

    def forward(self, input):
        size = input.size()
        x = input.clone()
        for i in range(size[1]):
            x[:, i] = (x[:, i] - self.mean[i]) / self.std[i]
        return x


class Permute(nn.Module):
    def __init__(self, permutation=[2, 1, 0]):
        super().__init__()
        self.permutation = permutation

    def forward(self, input):
        return input[:, self.permutation]




class Blur(nn.Module):
    def __init__(self, kernel):
        super(Blur, self).__init__()
        self.register_buffer("uniform_kernel", kernel)

    def forward(self, input):
        x = F.conv2d(input, self.uniform_kernel, bias=None, stride=1, padding=(1, 1), groups=3)
        return x



class anti_Blur(nn.Module):
    def __init__(self, kernel):
        super(anti_Blur, self).__init__()
        self.register_buffer("rui_kernel", kernel)


    def forward(self, input):
        x = F.conv2d(input, self.rui_kernel, bias=None, stride=1, padding=(1, 1), groups=3)
        return x