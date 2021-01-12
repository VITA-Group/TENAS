import torch.nn as nn
from ..cell_operations import ResNetBasicblock
from .cells import InferCell
from pdb import set_trace as bp


# The macro structure for architectures in NAS-Bench-201
class TinyNetwork(nn.Module):

    def __init__(self, C, N, genotype, num_classes, C_in=3, depth=-1):
        super(TinyNetwork, self).__init__()
        self._C               = C
        self._layerN          = N
        # C_in: number of input channel
        # depth: number of cells to forward

        self.stem = nn.Sequential(nn.Conv2d(C_in, C, kernel_size=3, padding=1, bias=False),
                                  nn.BatchNorm2d(C))

        layer_channels   = [C    ] * N + [C*2 ] + [C*2  ] * N + [C*4 ] + [C*4  ] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        if depth == -1:
            self.depth = len(layer_channels)
        else:
            self.depth = min(depth, len(layer_channels))

        C_prev = C
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
            if index >= self.depth:
                break
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2, True)
            else:
                cell = InferCell(genotype, C_prev, C_curr, 1)
            self.cells.append( cell )
            C_prev = cell.out_dim
        self._Layer= len(self.cells)

        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def get_message(self):
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
        return string

    def extra_repr(self):
        return ('{name}(C={_C}, N={_layerN}, L={_Layer})'.format(name=self.__class__.__name__, **self.__dict__))

    def forward(self, inputs, return_features=False):
        features = []
        feature = self.stem(inputs)
        features.append(feature)
        for i, cell in enumerate(self.cells):
            feature = cell(feature)
            features.append(feature)

        out = self.lastact(feature)
        features.append(out)
        out = self.global_pooling(out)
        features.append(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        if return_features:
            return out, logits, features
        else:
            return out, logits
