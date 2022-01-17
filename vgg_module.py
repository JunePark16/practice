import math
import torch.nn as nn
import torch.nn.functional as F

cfg = {
    'A' : [64,'M',128, 'M',256,256,'M',512,512,'M',512,512,'M'],
    'D' : [64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M']
}

class VGG(nn.Module):
    def __init__(self, pipeline):
        super(VGG,self).__init__()
        self.pipeline = pipeline
        self.classifier = nn.Sequential(
            nn.Linear( 7*7*512,512), # 7 = 224 * (1/2)^(5)
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512,10), 
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n=m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            pass
        
    def forward(self, x):
        x = self.pipeline(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.softmax(x)
        return x
    
def vgg_models(cfg):
    layers = []
    in_channel = 3
    for v in cfg:
        if v=='M':
            layers += [nn.MaxPool2d(kernel_size = 2, stride = 2)]
        else:
            layers += [nn.Conv2d(in_channel, v, kernel_size = 3, padding = 1), nn.ReLU()]
            in_channel = v
    return nn.Sequential(*layers)

def vgg11(VGG):
    return VGG(vgg_models(cfg['A']))

def vgg16(VGG):
    return VGG(vgg_models(cfg['D']))