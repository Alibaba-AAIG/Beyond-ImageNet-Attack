""" Adapted VGG pytorch model that used as surrogate. """
# from re import T
import torchvision.models as models
import torch

class Vgg19(torch.nn.Module):
    def __init__(self):
        super(Vgg19, self).__init__()
        model = models.vgg19(pretrained=True).eval()
        self.features = list(model.features)
        self.model = torch.nn.ModuleList(self.features)

    def forward(self, x):
        layers = []
        for ii, model_ in enumerate(self.model):
            x = model_(x)
            layers.append(x.clone())

        return layers

if __name__ == "__main__":
    # pass
    layer = Vgg19()(torch.zeros(2,3,224,224))
    print(len(layer))

