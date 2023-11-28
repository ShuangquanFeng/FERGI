# Adapted from https://github.com/deepinsight/insightface

import torch
import torch.nn as nn
from iresnet import IResNet, IBasicBlock

class IResNet_LastLayerModified(IResNet):
    def __init__(self, n_features, *args, **kwargs):
        super(IResNet_LastLayerModified, self).__init__(*args, **kwargs)
        self.fc = nn.Linear(512 * 7 * 7, n_features)
        del self.features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def _iresnet_basic(n_features, basic_block, n_blocks, pretrained=False, weights_path=None, **kwargs):
    model = IResNet_LastLayerModified(n_features, IBasicBlock, n_blocks, **kwargs)
    if pretrained == True:
        weights = torch.load(weights_path)
        weights = {k: v for k, v in weights.items() if k in model.state_dict() and weights[k].size() == model.state_dict()[k].size()}
        model.load_state_dict(weights, strict=False)
    return model
    
def iresnet_basic(n_layers, n_features, pretrained=False, weights_path=None, **kwargs):
    if n_layers == 18:
        n_blocks = [2, 2, 2, 2]
    elif n_layers == 34:
        n_blocks = [3, 4, 6, 3]
    elif n_layers == 50:
        n_blocks = [3, 4, 14, 3]
    elif n_layers == 100:
        n_blocks = [3, 13, 30, 3]
    elif n_layers == 200:
        n_blocks = [6, 26, 60, 6]
    else:
        raise ValueError('Invalid layer number')
    return _iresnet_basic(n_features, IBasicBlock, n_blocks, pretrained=pretrained, weights_path=weights_path, **kwargs)