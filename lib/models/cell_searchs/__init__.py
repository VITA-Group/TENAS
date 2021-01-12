# The macro structure is defined in NAS-Bench-201
from .search_model_darts    import TinyNetworkDarts
from .genotypes             import Structure as CellStructure, architectures as CellArchitectures
# NASNet-based macro structure
from .search_model_darts_nasnet import NASNetworkDARTS


nas201_super_nets = {'DARTS-V1': TinyNetworkDarts,
                     "DARTS-V2": TinyNetworkDarts}

nasnet_super_nets = {"DARTS-V1": NASNetworkDARTS,
                     "DARTS-V2": NASNetworkDARTS}
