import torch
import torch.nn as nn
from copy import deepcopy
from pdb import set_trace as bp
from ..cell_operations import OPS


INF = 1000


# This module is used for NAS-Bench-201, represents a small search space with a complete DAG
class NAS201SearchCell(nn.Module):

    def __init__(self, C_in, C_out, stride, max_nodes, op_names, affine=False, track_running_stats=True):
        super(NAS201SearchCell, self).__init__()

        self.op_names = deepcopy(op_names)
        self.edges = nn.ModuleDict()
        self.max_nodes = max_nodes
        self.in_dim = C_in
        self.out_dim = C_out
        for i in range(1, max_nodes):
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                if j == 0:
                    xlists = [OPS[op_name](C_in, C_out, stride, affine, track_running_stats) for op_name in op_names]
                else:
                    xlists = [OPS[op_name](C_in, C_out, 1, affine, track_running_stats) for op_name in op_names]
                self.edges[node_str] = nn.ModuleList(xlists)
        self.edge_keys = sorted(list(self.edges.keys()))
        self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}
        self.num_edges = len(self.edges)

    def extra_repr(self):
        string = 'info :: {max_nodes} nodes, inC={in_dim}, outC={out_dim}'.format(**self.__dict__)
        return string

    def forward(self, inputs, weightss):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            inter_nodes = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                weights = weightss[self.edge2index[node_str]]
                inter_nodes.append(sum(layer(nodes[j]) * w if w > 0.01 else 0 for layer, w in zip(self.edges[node_str], weights)))  # for pruning purpose
            nodes.append(sum(inter_nodes))
        return nodes[-1]


class MixedOp(nn.Module):

    def __init__(self, space, C, stride, affine, track_running_stats):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in space:
            op = OPS[primitive](C, C, stride, affine, track_running_stats)
            self._ops.append(op)

    def forward_darts(self, x, weights):
        return sum(w * op(x) if w > 0.01 else 0 for w, op in zip(weights, self._ops))  # for pruning purpose


# Learning Transferable Architectures for Scalable Image Recognition, CVPR 2018
class NASNetSearchCell(nn.Module):

    def __init__(self, space, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, affine, track_running_stats):
        super(NASNetSearchCell, self).__init__()
        self.reduction = reduction
        self.op_names = deepcopy(space)
        if reduction_prev: self.preprocess0 = OPS['skip_connect'](C_prev_prev, C, 2, affine, track_running_stats)
        else: self.preprocess0 = OPS['nor_conv_1x1'](C_prev_prev, C, 1, affine, track_running_stats)
        self.preprocess1 = OPS['nor_conv_1x1'](C_prev, C, 1, affine, track_running_stats)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self.edges = nn.ModuleDict()
        for i in range(self._steps):
            for j in range(2+i):
                node_str = '{:}<-{:}'.format(i, j)  # indicate the edge from node-(j) to node-(i+2)
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(space, C, stride, affine, track_running_stats)
                self.edges[ node_str ] = op
        self.edge_keys  = sorted(list(self.edges.keys()))
        self.edge2index = {key:i for i, key in enumerate(self.edge_keys)}
        self.num_edges  = len(self.edges)

    def forward_darts(self, s0, s1, weightss, alphass):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            clist = []
            for j, h in enumerate(states):
                node_str = '{:}<-{:}'.format(i, j)
                op = self.edges[ node_str ]
                weights = weightss[ self.edge2index[node_str] ]
                alphas = alphass[ self.edge2index[node_str] ]
                if sum(alphas) <= (-INF) * len(alphas):
                    # all ops on this edge are masked out
                    clist.append( 0 )
                else:
                    clist.append( op.forward_darts(h, weights) )
            states.append( sum(clist) )

        return torch.cat(states[-self._multiplier:], dim=1)
