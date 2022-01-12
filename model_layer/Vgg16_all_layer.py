""" Adapted VGG pytorch model that used as surrogate. """
# from re import T
import torchvision.models as models
import torch


class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.vgg = models.vgg16(pretrained=True).eval()
        self.features = list(self.vgg.features)
        self.model = torch.nn.ModuleList(self.features)

    def forward(self, x):
        layers = []
        for ii, model_ in enumerate(self.model):
            x = model_(x)
            layers.append(x.clone())

        x = self.vgg.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.vgg.classifier(x)
        layers.append(x.clone())
        return layers

if __name__ == "__main__":
    # pass
    layer = Vgg16()(torch.zeros(2,3,224,224))
    print(len(layer))
