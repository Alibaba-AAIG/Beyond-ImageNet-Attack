""" Adapted VGG pytorch model that used as surrogate. """
# from re import T
import torchvision.models as models
import torch


class Dense169(torch.nn.Module):
    def __init__(self):
        super(Dense169, self).__init__()
        # self.model = models.resnet50(pretrained=True).eval()
        self.model = models.densenet169(pretrained=True).eval()
        features = list(self.model.features)
        self.features = torch.nn.ModuleList(features)

    def forward(self, x):
        layers = []
        for ii, model_ in enumerate(self.features):
            x = model_(x)
            layers.append(x.clone())
        return layers




if __name__ == "__main__":
    # pass
    layer = Dense169()(torch.zeros(2,3,224,224))
    print(len(layer))

