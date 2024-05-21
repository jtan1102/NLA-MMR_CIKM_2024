from ogb.utils import smiles2graph
from torch_geometric.data import Data
import numpy as np
import torch


def graph_batch_from_smile(smiles_list):
    edge_idxes, edge_feats, node_feats, lstnode, batch = [], [], [], 0, []
    graphs = [smiles2graph(x) for x in smiles_list]
    lstedge = 0
    for idx, graph in enumerate(graphs):
        edge_idxes.append(graph['edge_index'] + lstnode)
        edge_feats.append(graph['edge_feat'])
        node_feats.append(graph['node_feat'])
        lstnode += graph['num_nodes']
        lstedge +=graph['edge_feat'].shape[0]
        assert graph['edge_feat'].shape[0] == graph['edge_index'].shape[1]
        batch.append(np.ones(graph['num_nodes'], dtype=np.int64) * idx)

    result = {
        'edge_index': np.concatenate(edge_idxes, axis=-1),
        'edge_attr': np.concatenate(edge_feats, axis=0),
        'batch': np.concatenate(batch, axis=0),
        'x': np.concatenate(node_feats, axis=0)
    }
    
    print(lstnode)
    print(lstedge)
    result = {k: torch.from_numpy(v) for k, v in result.items()}
    result['num_nodes'] = lstnode
    return Data(**result)

# 6767
# 14422 
# 2.13

# 8138
# 17442
# 2.1432784467928236