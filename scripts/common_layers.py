
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as I
import numpy as np
import math
import os

import pprint
from tqdm import tqdm
pp = pprint.PrettyPrinter(indent=1)
import numpy as np

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

import random

from rdkit import Chem

from collections.abc import Sequence




def buildPrjSmiles(molecule, med_voc, device="cpu:0"):


    average_index, smiles_all = [], []


    print(len(med_voc.items()))  # 131
    for index, ndc in med_voc.items():
        smilesList = []
        smilesList = molecule[ndc]

        """Create each data with the above defined functions."""
        counter = 0  # counter how many drugs are under that ATC-3
        for smiles in smilesList:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                smiles_all.append(smiles)
                counter += 1
            else:
                print('[SMILES]', smiles)
                print('[Error] Invalid smiles')
        average_index.append(counter)

        """Transform the above each data of numpy
        to pytorch tensor on a device (i.e., CPU or GPU).
        """
    # transform into projection matrix
    n_col = sum(average_index)
    n_row = len(average_index)

    average_projection = np.zeros((n_row, n_col))
    col_counter = 0
    for i, item in enumerate(average_index):
        average_projection[i, col_counter: col_counter + item] = 1 / item
        col_counter += item

    print("Smiles Num:{}".format(len(smiles_all)))
    print("n_col:{}".format(n_col))
    print("n_row:{}".format(n_row))

    return torch.FloatTensor(average_projection), smiles_all



class AdjAttenAgger(torch.nn.Module):
    def __init__(self, Qdim, Kdim, mid_dim, device, *args, **kwargs):
        super(AdjAttenAgger, self).__init__(*args, **kwargs)
        self.model_dim = mid_dim
        self.Qdense = torch.nn.Linear(Qdim, mid_dim)
        self.Kdense = torch.nn.Linear(Kdim, mid_dim)
        # self.use_ln = use_ln
        self.device = device
        self.drug_projection = torch.nn.Linear(Kdim, mid_dim)

    def forward(self, main_feat, other_feat, fix_feat=None, mask=None):
        # import pdb;pdb.set_trace()

        # main_feat shape :  torch.Size([bs, 768])
        # other_feat shape :  torch.Size([bs, 3, 768])
        # fix_feat shape :  torch.Size([bs, 3])

        Q = self.Qdense(main_feat)
        K = self.Kdense(other_feat)
        Q = Q.unsqueeze(-2)
        Attn = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.model_dim)

        # Q.shape : torch.Size([bs, 1, 256])
        # K.shape : torch.Size([bs, 3, 256])
        # K.transpose(1, 2).shape : torch.Size([bs, 256, 3])
        # Attn.shape : torch.Size([bs, 1, 3])

        Attn = Attn.squeeze(1)

        if mask is not None:
            Attn = Attn + mask
            # mask = mask.to(self.device)
            # Attn = torch.masked_fill(Attn, mask, -(1 << 32))

        Attn = torch.softmax(Attn, dim=-1)  # Attn.shape :  torch.Size([bs, 131, 3])
        # print(Attn[0])
        # print(mask[0])

        Attn = Attn.unsqueeze(-2)
        # import pdb;pdb.set_trace()
    
        # batch_size = fix_feat.shape[0]
        # feature_size = fix_feat.shape[1]
        # # fix_feat = torch.diag(fix_feat)
        # fix_feat_expand = fix_feat.unsqueeze(1).repeat(1, feature_size, 1)
        # mask = torch.eye(feature_size).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device)
        
        # fix_feat = fix_feat_expand * mask   # fix_feat.shape :  torch.Size([bs, 3, 3])
        
        # other_feat = torch.matmul(fix_feat, other_feat)   # other_feat.shape :  torch.Size([bs, 3, 64])
        # import pdb;pdb.set_trace()
        O = self.drug_projection(torch.bmm(Attn, other_feat).squeeze(-2))  # O.shape :  torch.Size([bs, 131, 64])
        return O, Attn

class AdjAttenAgger_no_projection(torch.nn.Module):
    def __init__(self, Qdim, Kdim, mid_dim, device, *args, **kwargs):
        super(AdjAttenAgger_no_projection, self).__init__(*args, **kwargs)
        self.model_dim = mid_dim
        self.Qdense = torch.nn.Linear(Qdim, mid_dim)
        self.Kdense = torch.nn.Linear(Kdim, mid_dim)
        # self.use_ln = use_ln
        self.device = device
        self.drug_projection = torch.nn.Linear(Kdim, mid_dim)

    def forward(self, main_feat, other_feat, fix_feat=None, mask=None):
        # import pdb;pdb.set_trace()

        # main_feat shape :  torch.Size([bs, 768])
        # other_feat shape :  torch.Size([bs, 3, 768])
        # fix_feat shape :  torch.Size([bs, 3])

        Q = self.Qdense(main_feat)
        K = self.Kdense(other_feat)
        Q = Q.unsqueeze(-2)
        Attn = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.model_dim)

        # Q.shape : torch.Size([bs, 1, 256])
        # K.shape : torch.Size([bs, 3, 256])
        # K.transpose(1, 2).shape : torch.Size([bs, 256, 3])
        # Attn.shape : torch.Size([bs, 1, 3])

        Attn = Attn.squeeze(1)

        if mask is not None:
            Attn = Attn + mask
            # mask = mask.to(self.device)
            # Attn = torch.masked_fill(Attn, mask, -(1 << 32))

        Attn = torch.softmax(Attn, dim=-1)  # Attn.shape :  torch.Size([bs, 131, 3])
        # print(Attn[0])
        # print(mask[0])

        Attn = Attn.unsqueeze(-2)

        # batch_size = fix_feat.shape[0]
        # feature_size = fix_feat.shape[1]
        # # fix_feat = torch.diag(fix_feat)
        # fix_feat_expand = fix_feat.unsqueeze(1).repeat(1, feature_size, 1)
        # mask = torch.eye(feature_size).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device)
        
        # fix_feat = fix_feat_expand * mask   # fix_feat.shape :  torch.Size([bs, 3, 3])
        
        # other_feat = torch.matmul(fix_feat, other_feat)   # other_feat.shape :  torch.Size([bs, 3, 64])
        
        O = torch.bmm(Attn, other_feat).squeeze(-2)  # O.shape :  torch.Size([bs, 131, 64])
        return O
    

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, batch_norm=False, activation="relu", dropout=0):
        super(MLP, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.dims = [input_dim] + hidden_dims

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
        if batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(len(self.dims) - 2):
                self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))
        else:
            self.batch_norms = None

    def forward(self, input):
        layer_input = input

        for i, layer in enumerate(self.layers):
            hidden = layer(layer_input)
            if i < len(self.layers) - 1:
                if self.batch_norms:
                    x = hidden.flatten(0, -2)
                    hidden = self.batch_norms[i](x).view_as(hidden)
                hidden = self.activation(hidden)
                if self.dropout:
                    hidden = self.dropout(hidden)
            if hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            layer_input = hidden

        return hidden

