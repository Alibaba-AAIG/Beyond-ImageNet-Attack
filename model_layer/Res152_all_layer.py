""" Adapted VGG pytorch model that used as surrogate. """
# from re import T
import torchvision.models as models
import torch



class Resnet152(torch.nn.Module):
    def __init__(self):
        super(Resnet152, self).__init__()
        self.model = models.resnet152(pretrained=True).eval()
        features = list(self.model.children())
        self.features = torch.nn.ModuleList(features)
        self.internal = [ii for ii in range(9)]

        
    def forward(self, x):
        # pred = self.model(x)
        layers = []
        for ii, model_ in enumerate(self.features):
            if ii in self.internal:
                x = model_(x)
                layers.append(x.clone())
        return layers

if __name__ == "__main__":
    # pass
    layer = Resnet152()(torch.zeros(2,3,224,224))
    print(len(layer))



