import torch.nn as nn
import torch.nn.init as init
import math

class VGG(nn.Module):
    def __init__(self,features):
        super(VGG,self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Liner(512,512),
            # nn.Liner(-1,512)
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512,512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512,10),
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # 2d 행렬로 변환
        x = self.classifier(x)
        return x

def make_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg :
        if v =='M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding = 1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'A' : [64,'M',128, 'M',256,256,'M',512,512,'M',512,512,'M']
}

def vgg11():
    return VGG(make_layers(cfg['A']))
