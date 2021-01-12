import random
import torch
import torch.nn as nn
from copy import deepcopy
from ..cell_operations import ResNetBasicblock
from .search_cells     import NAS201SearchCell as SearchCell
from .genotypes        import Structure
import torch.nn.functional as F
from pdb import set_trace as bp


def cal_entropy(logit: torch.Tensor, dim=-1) -> torch.Tensor:
    """
    :param logit: An unnormalized vector.
    :param dim: ~
    :return: entropy
    """
    prob = F.softmax(logit, dim=dim)
    log_prob = F.log_softmax(logit, dim=dim)

    entropy = -(log_prob * prob).sum(-1, keepdim=False)

    return entropy


class TinyNetworkDarts(nn.Module):

    def __init__(self, C, N, max_nodes, num_classes, search_space, affine, track_running_stats, depth=-1, use_stem=True):
        super(TinyNetworkDarts, self).__init__()
        self._C        = C
        self._layerN   = N  # number of stacked cell at each stage
        self.max_nodes = max_nodes
        self.use_stem = use_stem
        self.stem = nn.Sequential(nn.Conv2d(min(3, C), C, kernel_size=3, padding=1, bias=False),
                                  nn.BatchNorm2d(C))

        layer_channels   = [C    ] * N + [C*2 ] + [C*2  ] * N + [C*4 ] + [C*4  ] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        C_prev, num_edge, edge2index = C, None, None
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
            if depth > 0 and index >= depth: break
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2)
            else:
                cell = SearchCell(C_prev, C_curr, 1, max_nodes, search_space, affine, track_running_stats)
                if num_edge is None: num_edge, edge2index = cell.num_edges, cell.edge2index
                else: assert num_edge == cell.num_edges and edge2index == cell.edge2index, 'invalid {:} vs. {:}.'.format(num_edge, cell.num_edges)
            self.cells.append( cell )
            C_prev = cell.out_dim
        self.op_names   = deepcopy( search_space )
        self._Layer     = len(self.cells)
        self.edge2index = edge2index
        self.lastact    = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self.arch_parameters = nn.Parameter( 1e-3*torch.randn(num_edge, len(search_space)) )

    def entropy(self, mean=True):
        if mean:
            return cal_entropy(self.arch_parameters, -1).mean().view(-1)
        else:
            return cal_entropy(self.arch_parameters, -1)

    def get_weights(self):
        xlist = list( self.stem.parameters() ) + list( self.cells.parameters() )
        xlist += list( self.lastact.parameters() ) + list( self.global_pooling.parameters() )
        xlist += list( self.classifier.parameters() )
        return xlist

    def get_alphas(self):
        return [self.arch_parameters]

    def set_alphas(self, arch_parameters):
        self.arch_parameters.data.copy_(arch_parameters[0].data)

    def show_alphas(self):
        with torch.no_grad():
            return 'arch-parameters :\n{:}'.format( nn.functional.softmax(self.arch_parameters, dim=-1).cpu() )

    def get_message(self):
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
        return string

    def extra_repr(self):
        return ('{name}(C={_C}, Max-Nodes={max_nodes}, N={_layerN}, L={_Layer})'.format(name=self.__class__.__name__, **self.__dict__))

    def genotype(self, get_random=False, hardwts=None):
        genotypes = []
        for i in range(1, self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                with torch.no_grad():
                    if hardwts is not None:
                        weights = hardwts[ self.edge2index[node_str] ]
                        op_name = self.op_names[ weights.argmax().item() ]
                    elif get_random:
                        op_name = random.choice(self.op_names)
                    else:
                        weights = self.arch_parameters[ self.edge2index[node_str] ]
                        op_name = self.op_names[ weights.argmax().item() ]
                xlist.append((op_name, j))
            genotypes.append( tuple(xlist) )
        return Structure( genotypes )

    def forward(self, inputs, return_features=False):
        alphas = nn.functional.softmax(self.arch_parameters, dim=-1)
        features_all = []
        if self.use_stem:
            feature = self.stem(inputs)
        else:
            feature = inputs
        features_all.append(feature.detach())
        for i, cell in enumerate(self.cells):
            if isinstance(cell, SearchCell):
                feature = cell(feature, alphas)
            else:
                feature = cell(feature)
            features_all.append(feature.detach())

        out = self.lastact(feature)
        out = self.global_pooling( out )
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        if return_features:
            return out, logits, features_all
        else:
            return out, logits
