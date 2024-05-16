import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl.function as fn
from utils import set_seed
from torch.distributions import Beta
from torch.distributions import VonMises
from torch.distributions.multivariate_normal import MultivariateNormal
import dgl
import sys

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    
from dgl.nn.pytorch.conv import EGATConv
from dgl.nn.pytorch.conv import GraphConv 
from dgl.nn.pytorch.conv import CFConv 
from egnn import EGNN
from schnet import SchNetGNN
from mpnn import MPNNGNN
from mgcn import MGCNGNN
# from dimenet_pp import DimeNetPP
# from megnet import MEGNet

class SymGNNLayer(nn.Module):
    def __init__(self, in_node_dim, in_edge_dim, out_node_dim, out_edge_dim, head, norm_type=None, skip_connect=True, last_layer=False):
        super(SymGNNLayer, self).__init__()
        self.egat_layer = EGATConv(in_node_dim, in_edge_dim, out_node_dim, out_edge_dim, head)
        self.node_norm = None
        self.edge_norm = None
        self.projection = None
        self.last_layer = last_layer
        self.norm_type = norm_type
        self.head = head
        if norm_type == 'batch':
            self.node_norm = nn.BatchNorm1d(out_node_dim * head)
            self.edge_norm = nn.BatchNorm1d(out_edge_dim * head)
        elif norm_type == 'layer':
            self.node_norm = nn.LayerNorm(out_node_dim * head)
            self.edge_norm = nn.LayerNorm(out_edge_dim * head)

        self.skip_connect = skip_connect
        if skip_connect:
            self.projection_node = nn.Linear(out_node_dim * head, out_node_dim)
            self.projection_edge = nn.Linear(out_edge_dim * head, out_edge_dim)
        
    def forward(self, g, nfeats, efeats, eweight=None):
        n_copy, e_copy = nfeats.clone(), efeats.clone()
        nh, eh = self.egat_layer(g, nfeats, efeats, edge_weight=eweight)
        if self.last_layer:
            nh = nh.mean(1)
            eh = eh.mean(1)
        else:
            nh = nh.flatten(1)
            eh = eh.flatten(1)
        
        if self.norm_type != None:
            nh = self.node_norm(nh)
            eh = self.edge_norm(eh)
        
        if self.skip_connect:
            nh = self.projection_node(nh)
            eh = self.projection_edge(eh)
            nh, eh = nh + n_copy, eh + e_copy
        
        return nh, eh
    
class SymGNN(nn.Module):
    def __init__(
        self,
        num_layers,
        num_ntypes,
        emb_dim,
        hidden_dim,
        out_dim,
        efeat_dim,
        dropout_ratio,
        activation,
        heads,
        norm_type="none",
    ):
        super().__init__()
        self.drop_out = nn.Dropout(dropout_ratio)
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            if i != self.num_layers - 1:
                self.layers.append(
                    SymGNNLayer(
                        emb_dim, emb_dim, emb_dim, emb_dim, heads[i], 'batch'
                    )
                )
            else:
                self.layers.append(
                    SymGNNLayer(
                        emb_dim, emb_dim, emb_dim, emb_dim, heads[i], 'batch', last_layer=True
                    )
                )
        
        self.node_encoder = nn.Sequential(nn.Embedding(num_ntypes, emb_dim),
                                          nn.BatchNorm1d(emb_dim),
                                          nn.ReLU(),
                                          nn.Linear(emb_dim, emb_dim), 
                                          nn.BatchNorm1d(emb_dim),
                                          nn.ReLU())

        self.node_decoder = nn.Sequential(nn.Linear(emb_dim, emb_dim), 
                                          nn.BatchNorm1d(emb_dim),
                                          nn.PReLU(),
                                          nn.Linear(emb_dim, emb_dim),
                                          nn.BatchNorm1d(emb_dim),
                                          nn.PReLU(),
                                          nn.Linear(emb_dim, emb_dim),
                                          nn.BatchNorm1d(emb_dim),
                                          nn.PReLU(),
                                          nn.Linear(emb_dim, emb_dim),
                                          nn.BatchNorm1d(emb_dim), 
                                          nn.PReLU(),
                                          nn.Linear(emb_dim, out_dim))


        self.edge_encoder = nn.Sequential(nn.Linear(efeat_dim, emb_dim),
                                          nn.BatchNorm1d(emb_dim),
                                          nn.ReLU(),
                                          nn.Linear(emb_dim, emb_dim), 
                                          nn.BatchNorm1d(emb_dim),
                                          nn.ReLU())
        
        # self.rot_loc1 = torch.tensor(0)
        # self.rot_con1 = torch.tensor(1)
        # self.rot_loc2 = torch.tensor(0.5)
        # self.rot_con2 = torch.tensor(1)
        # self.rot_loc3 = torch.tensor(-0.5)
        # self.rot_con3 = torch.tensor(1)
        
        # self.ref_loc1 = torch.tensor(0)
        # self.ref_con1 = torch.tensor(1)
        # self.ref_loc2 = torch.tensor(0.5)
        # self.ref_con2 = torch.tensor(1)
        # self.ref_loc3 = torch.tensor(-0.5)
        # self.ref_con3 = torch.tensor(1)
        
        self.rot_loc1 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.rot_con1 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.rot_loc2 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.rot_con2 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.rot_loc3 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.rot_con3 = nn.Parameter(torch.randn(1, requires_grad=True))
        
        self.ref_loc1 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.ref_con1 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.ref_loc2 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.ref_con2 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.ref_loc3 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.ref_con3 = nn.Parameter(torch.randn(1, requires_grad=True))
        
    def forward(self, g, nfeats, efeats, eweight=None, embed=False):
        nfeats = nfeats.squeeze(1)
        nh = self.node_encoder(nfeats)
        eh = torch.zeros(efeats.shape[0], self.emb_dim).to(efeats.device)
        
        # vonmise_rot_dist1 = VonMises(self.rot_loc1, torch.exp(self.rot_con1))
        # vonmise_rot_dist2 = VonMises(self.rot_loc2, torch.exp(self.rot_con2))
        # vonmise_rot_dist3 = VonMises(self.rot_loc3, torch.exp(self.rot_con3))
        # vonmise_ref_dist1 = VonMises(self.ref_loc1, torch.exp(self.ref_con1))
        # vonmise_ref_dist2 = VonMises(self.ref_loc2, torch.exp(self.ref_con2))
        # vonmise_ref_dist3 = VonMises(self.ref_loc3, torch.exp(self.ref_con3))
        
        # For rotations
        for i in range(2):
            # theta = vonmise_rot_dist1.sample()
            # phi = vonmise_rot_dist2.sample()
            # gamma = vonmise_rot_dist3.sample()
            
            theta, phi, gamma = torch.tensor(i), torch.tensor(i), torch.tensor(i)
            
            R_theta = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0], [torch.sin(theta), torch.cos(theta), 0], [0, 0, 1]]).to(efeats.device)
            R_phi = torch.tensor([[torch.cos(phi), 0, -torch.sin(phi)], [0, 1, 0], [torch.sin(phi), 0, torch.cos(phi)]]).to(efeats.device)
            R_gamma = torch.tensor([[1, 0, 0], [0, torch.cos(gamma), -torch.sin(gamma)], [0, torch.sin(gamma), torch.cos(gamma)]]).to(efeats.device)
            
            ortho = R_theta @ R_phi @ R_gamma
            feat = efeats @ ortho
            eh = eh + self.edge_encoder(feat)
        
        # For reflections
        # for _ in range(3):
        #     theta = vonmise_ref_dist1.sample()
        #     phi = vonmise_ref_dist2.sample()
        #     gamma = vonmise_ref_dist3.sample()
            
        #     R_theta = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0], [torch.sin(theta), torch.cos(theta), 0], [0, 0, 1]]).to(efeats.device)
        #     R_phi = torch.tensor([[torch.cos(phi), 0, -torch.sin(phi)], [0, 1, 0], [torch.sin(phi), 0, torch.cos(phi)]]).to(efeats.device)
        #     R_gamma = torch.tensor([[1, 0, 0], [0, torch.cos(gamma), -torch.sin(gamma)], [0, torch.sin(gamma), torch.cos(gamma)]]).to(efeats.device)
            
        #     ortho = -R_theta @ R_phi @ R_gamma
        #     feat = efeats @ ortho
        #     eh = eh + self.edge_encoder(feat) 
        
        h_list = []
        for i, layer in enumerate(self.layers):
            nh, eh = layer(g, nh, eh, eweight)
            h_list.append(nh)
            nh = self.drop_out(nh)
            eh = self.drop_out(eh)
        
        if embed:
            return [], nh
        
        h = self.node_decoder(nh)
        # h = torch.mean(nh, dim=1, keepdim=True)
        return h_list, h
    
class QM9(nn.Module):
    def __init__(
        self,
        num_layers,
        num_ntypes,
        emb_dim,
        hidden_dim,
        out_dim,
        efeat_dim,
        dropout_ratio,
        activation,
        heads,
        norm_type="none",
    ):
        super().__init__()
        self.drop_out = nn.Dropout(dropout_ratio)
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            if i != self.num_layers - 1:
                self.layers.append(
                    SymGNNLayer(
                        emb_dim, emb_dim, emb_dim, emb_dim, heads[i], 'batch'
                    )
                )
            else:
                self.layers.append(
                    SymGNNLayer(
                        emb_dim, emb_dim, emb_dim, emb_dim, heads[i], 'batch', last_layer=True
                    )
                )
        
        self.node_encoder = nn.Sequential(nn.Embedding(num_ntypes, emb_dim),
                                          nn.BatchNorm1d(emb_dim),
                                          nn.ReLU(),
                                          nn.Linear(emb_dim, emb_dim), 
                                          nn.BatchNorm1d(emb_dim),
                                          nn.ReLU())

        self.node_decoder = nn.Sequential(nn.Linear(emb_dim, emb_dim), 
                                          nn.BatchNorm1d(emb_dim),
                                          nn.PReLU(),
                                          nn.Linear(emb_dim, emb_dim),
                                          nn.BatchNorm1d(emb_dim),
                                          nn.PReLU(),
                                          nn.Linear(emb_dim, emb_dim),
                                          nn.BatchNorm1d(emb_dim),
                                          nn.PReLU(),
                                          nn.Linear(emb_dim, emb_dim),
                                          nn.BatchNorm1d(emb_dim), 
                                          nn.PReLU(),
                                          nn.Linear(emb_dim, out_dim))


        self.edge_encoder = nn.Sequential(nn.Linear(efeat_dim, emb_dim),
                                          nn.BatchNorm1d(emb_dim),
                                          nn.ReLU(),
                                          nn.Linear(emb_dim, emb_dim), 
                                          nn.BatchNorm1d(emb_dim),
                                          nn.ReLU())
        
        self.rot_loc1 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.rot_con1 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.rot_loc2 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.rot_con2 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.rot_loc3 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.rot_con3 = nn.Parameter(torch.randn(1, requires_grad=True))
        
        self.ref_loc1 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.ref_con1 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.ref_loc2 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.ref_con2 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.ref_loc3 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.ref_con3 = nn.Parameter(torch.randn(1, requires_grad=True))
        
    def forward(self, g, nfeats, efeats, eweight=None, embed=False):
        nfeats = nfeats.squeeze(1)
        nh = self.node_encoder(nfeats)
        eh = torch.zeros(efeats.shape[0], self.emb_dim).to(efeats.device)
        
        vonmise_rot_dist1 = VonMises(self.rot_loc1, torch.exp(self.rot_con1))
        vonmise_rot_dist2 = VonMises(self.rot_loc2, torch.exp(self.rot_con2))
        vonmise_rot_dist3 = VonMises(self.rot_loc3, torch.exp(self.rot_con3))
        vonmise_ref_dist1 = VonMises(self.ref_loc1, torch.exp(self.ref_con1))
        vonmise_ref_dist2 = VonMises(self.ref_loc2, torch.exp(self.ref_con2))
        vonmise_ref_dist3 = VonMises(self.ref_loc3, torch.exp(self.ref_con3))
        
        # For rotations
        for _ in range(1):
            theta = vonmise_rot_dist1.sample()
            phi = vonmise_rot_dist2.sample()
            gamma = vonmise_rot_dist3.sample()
            
            R_theta = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0], [torch.sin(theta), torch.cos(theta), 0], [0, 0, 1]]).to(efeats.device)
            R_phi = torch.tensor([[torch.cos(phi), 0, -torch.sin(phi)], [0, 1, 0], [torch.sin(phi), 0, torch.cos(phi)]]).to(efeats.device)
            R_gamma = torch.tensor([[1, 0, 0], [0, torch.cos(gamma), -torch.sin(gamma)], [0, torch.sin(gamma), torch.cos(gamma)]]).to(efeats.device)
            
            ortho = R_theta @ R_phi @ R_gamma
            feat = efeats @ ortho
            eh = eh + self.edge_encoder(feat)
        
        # For reflections
        for _ in range(1):
            theta = vonmise_ref_dist1.sample()
            phi = vonmise_ref_dist2.sample()
            gamma = vonmise_ref_dist3.sample()
            
            R_theta = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0], [torch.sin(theta), torch.cos(theta), 0], [0, 0, 1]]).to(efeats.device)
            R_phi = torch.tensor([[torch.cos(phi), 0, -torch.sin(phi)], [0, 1, 0], [torch.sin(phi), 0, torch.cos(phi)]]).to(efeats.device)
            R_gamma = torch.tensor([[1, 0, 0], [0, torch.cos(gamma), -torch.sin(gamma)], [0, torch.sin(gamma), torch.cos(gamma)]]).to(efeats.device)
            
            ortho = -R_theta @ R_phi @ R_gamma
            feat = efeats @ ortho
            eh = eh + self.edge_encoder(feat) 
        
        h_list = []
        for i, layer in enumerate(self.layers):
            nh, eh = layer(g, nh, eh, eweight)
            h_list.append(nh)
            nh = self.drop_out(nh)
            eh = self.drop_out(eh)
        
        if embed:
            return [], nh
        
        h = self.node_decoder(nh)
        g.ndata['h'] = h
        # h = torch.mean(nh, dim=1, keepdim=True)
        return h_list, dgl.sum_nodes(g, 'h')
    
class Ablation_SymGNN(nn.Module):
    def __init__(
        self,
        num_layers,
        num_ntypes,
        emb_dim,
        hidden_dim,
        out_dim,
        efeat_dim,
        dropout_ratio,
        activation,
        heads,
        norm_type="none",
    ):
        super().__init__()
        self.drop_out = nn.Dropout(dropout_ratio)
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            if i != self.num_layers - 1:
                self.layers.append(
                    SymGNNLayer(
                        emb_dim, emb_dim, emb_dim, emb_dim, heads[i], None
                    )
                )
            else:
                self.layers.append(
                    SymGNNLayer(
                        emb_dim, emb_dim, emb_dim, emb_dim, heads[i], None, last_layer=True
                    )
                )
        
        self.node_encoder = nn.Sequential(nn.Embedding(num_ntypes, emb_dim),
                                          nn.ReLU(),
                                          nn.Linear(emb_dim, emb_dim), 
                                          nn.ReLU())

        self.node_decoder = nn.Sequential(nn.Linear(emb_dim, emb_dim), 
                                          nn.PReLU(),
                                          nn.Linear(emb_dim, emb_dim),
                                          nn.PReLU(),
                                          nn.Linear(emb_dim, emb_dim),
                                          nn.PReLU(),
                                          nn.Linear(emb_dim, emb_dim),
                                          nn.PReLU(),
                                          nn.Linear(emb_dim, out_dim))


        self.edge_encoder = nn.Sequential(nn.Linear(efeat_dim, emb_dim),
                                          nn.ReLU(),
                                          nn.Linear(emb_dim, emb_dim), 
                                          nn.ReLU())
        
    def forward(self, g, nfeats, efeats, eweight=None):
        nfeats = nfeats.squeeze(1)
        nh = self.node_encoder(nfeats)
        eh = self.edge_encoder(efeats)
        
        h_list = []
        for i, layer in enumerate(self.layers):
            nh, eh = layer(g, nh, eh, eweight)
            h_list.append(nh)
            nh = self.drop_out(nh)
            eh = self.drop_out(eh)
        
        h = self.node_decoder(nh)
        return h_list, h
    
class MLP(nn.Module):
    def __init__(
        self,
        num_layers,
        num_ntypes,
        emb_dim,
        hidden_dim,
        out_dim,
        efeat_dim,
        dropout_ratio,
        activation,
    ):
        super().__init__()
        self.drop_out = nn.Dropout(dropout_ratio)
        self.in_dim = num_ntypes
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(
                nn.Sequential(2 * emb_dim, 2 * emb_dim),
                nn.ReLU(inplace=True)
            )
        
        
        self.node_encoder = nn.Sequential(nn.Embedding(num_ntypes, emb_dim),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(emb_dim, emb_dim), 
                                          nn.ReLU(inplace=True))

        self.npos_encoder = nn.Sequential(nn.Linear(efeat_dim, emb_dim), 
                                          nn.ReLU(inplace=True),
                                          nn.Linear(emb_dim, emb_dim), 
                                          nn.ReLU(inplace=True))
        
        self.decoder = nn.Sequential(nn.Linear(2 * emb_dim, emb_dim),
                                     nn.PReLU(),
                                     nn.Linear(emb_dim, out_dim))
        
    def forward(self, g, nfeats, npos, eweight=None):
        nfeats = nfeats.squeeze(1)
        nh = self.node_encoder(nfeats)
        nposh = self.npos_encoder(npos)
        feat = torch.cat((nh, nposh), dim=-1)
        
        for i, layer in enumerate(self.layers):
            nh = layer(feat)

        return [], nh
    
class GCNLayer_EdgeCat(nn.Module):
    def __init__(self, hidden_dim, norm_type='none'):
        super(GCNLayer_EdgeCat, self).__init__()
        if norm_type=='batch':
            self.nW = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), 
                                    nn.BatchNorm1d(hidden_dim),
                                    nn.ReLU(inplace=True))

            self.eW = nn.Sequential(nn.Linear(hidden_dim*3, hidden_dim), 
                                    nn.BatchNorm1d(hidden_dim),
                                    nn.ReLU(inplace=True))   

        else:
            self.nW = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), 
                                    nn.ReLU(inplace=True))

            self.eW = nn.Sequential(nn.Linear(hidden_dim*3, hidden_dim), 
                                    nn.ReLU(inplace=True))

    def forward(self, g, nfeats, efeats, eweight=None):
        if eweight is not None:
            g.edata['_eweights'] = eweight

        ## For skip connection
        node_identity = nfeats
        edge_identity = efeats
        g.ndata['nh'] = nfeats
        g.edata['eh'] = efeats
        
        def msg_fn(edges):
            eh = torch.cat([edges.src['nh'].squeeze(), edges.data['eh'], edges.dst['nh'].squeeze()], dim = -1)
            
            if eweight is not None:
                eh = eh * edges.data['_eweights'].unsqueeze(-1)

            m = self.eW(eh)
            g.edata['m'] = m
            return {'m' : m}

        reduce_fn = fn.mean('m', 'h')
        
        def apply_fn(nodes):
            nh = torch.cat([nodes.data['nh'].squeeze(), nodes.data['h']], dim = -1)
            h = self.nW(nh)
            return {'h': h}

        g.update_all(msg_fn, reduce_fn, apply_fn)
        
        return g.ndata['h'] + node_identity, g.edata['m'] + edge_identity
    
class GCN_Edge(nn.Module):
    def __init__(
        self,
        num_layers,
        num_ntypes,
        emb_dim,
        hidden_dim,
        out_dim,
        efeat_dim,
        dropout_ratio,
        activation,
        norm_type="none"
    ):
        
        '''
        emb_dim for encoder/decoder.
        hidden_dim for GNN layers.
        '''
        super().__init__()
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.dropout = nn.Dropout(dropout_ratio)

        EdgeCat = True
        print('Note: EdgeCat = ', EdgeCat)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(GCNLayer_EdgeCat(emb_dim, None))

        self.node_encoder = nn.Sequential(nn.Embedding(num_ntypes, emb_dim),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(emb_dim, emb_dim), 
                                          nn.ReLU(inplace=True))

        self.node_decoder = nn.Sequential(nn.Linear(emb_dim, emb_dim), 
                                          nn.ReLU(inplace=True),
                                          nn.Linear(emb_dim, emb_dim),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(emb_dim, out_dim))


        self.edge_encoder = nn.Sequential(nn.Linear(efeat_dim, emb_dim), 
                                          nn.ReLU(inplace=True),
                                          nn.Linear(emb_dim, emb_dim), 
                                          nn.ReLU(inplace=True))
        

    def forward(self, g, nfeats, efeats, eweight=None):
        nfeats = nfeats.squeeze(1)
        nh = self.node_encoder(nfeats)
        eh = self.edge_encoder(efeats)
            
        h_list = []
        for l, layer in enumerate(self.layers):
            nh, eh = layer(g, nh, eh, eweight)
            h_list.append(nh)
            
            nh = self.dropout(nh)
            eh = self.dropout(eh)
                            
        h = self.node_decoder(nh)
        return h_list, h          
                            
class GCN(nn.Module):
    def __init__(
        self,
        num_layers,
        ncate_dim,
        npos_dim,
        emb_dim,
        hidden_dim,
        out_dim,
        dropout_ratio,
        norm_type="none",
    ):
        super().__init__()
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(GraphConv(emb_dim, emb_dim))
        self.ncate_emb = nn.Embedding(ncate_dim, emb_dim)
        self.npos_emb = nn.Linear(npos_dim, emb_dim)
        
        self.node_encoder = nn.Sequential(nn.Linear(emb_dim, emb_dim),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(emb_dim, emb_dim), 
                                          nn.ReLU(inplace=True))
        
        self.node_decoder = nn.Sequential(nn.Linear(emb_dim, emb_dim), 
                                          nn.ReLU(inplace=True),
                                          nn.Linear(emb_dim, emb_dim),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(emb_dim, out_dim))   
        
    
    def forward(self, graph, feat, npos, eweight=None): 
        feat = feat.squeeze(1)
        h1 = self.ncate_emb(feat)
        h = self.node_encoder(h1)
        h_list = []
        
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            h_list.append(h)
            
        h = self.node_decoder(h)
        return h_list, h
    
class SchNet(nn.Module):
    def __init__(
        self,
        num_layers,
        num_ntypes,
        emb_dim,
        hidden_dim,
        out_dim,
        efeat_dim,
        dropout_ratio,
        activation,
        norm_type="none",
    ):
        super().__init__()
        self.num_layers = num_layers
        hidden_feats = [emb_dim for _ in range(self.num_layers)]
        self.model = SchNetGNN(node_feats=emb_dim, hidden_feats=hidden_feats, num_node_types=2, cutoff=51.0, gap=0.1)

        self.node_decoder = nn.Sequential(nn.Linear(emb_dim, emb_dim), 
                                          nn.PReLU(),
                                          nn.Linear(emb_dim, emb_dim),
                                          nn.PReLU(),
                                          nn.Linear(emb_dim, emb_dim),
                                          nn.PReLU(),
                                          nn.Linear(emb_dim, emb_dim),
                                          nn.PReLU(),
                                          nn.Linear(emb_dim, out_dim))

        self.dropout = nn.Dropout(dropout_ratio)
        
    def forward(self, g, nfeats, efeats, eweight=None, embed=None):
        nfeats = nfeats.squeeze(1)
        efeats = torch.norm(efeats, p=2, dim=1, keepdim=True)
        nh = self.model(g, nfeats, efeats)
        h = self.node_decoder(nh)
        return [], h
    
class MGCN(nn.Module):
    def __init__(
        self,
        num_layers,
        num_ntypes,
        emb_dim,
        hidden_dim,
        out_dim,
        efeat_dim,
        dropout_ratio,
        activation,
        norm_type="none",
    ):
        super().__init__()
        self.num_layers = num_layers
        self.model = MGCNGNN(feats=emb_dim, n_layers=self.num_layers, num_node_types=2, num_edge_types=3, cutoff=51.0, gap=0.1)

        self.node_decoder = nn.Sequential(nn.Linear(emb_dim * (self.num_layers + 1), emb_dim), 
                                          nn.PReLU(),
                                          nn.Linear(emb_dim, emb_dim),
                                          nn.PReLU(),
                                          nn.Linear(emb_dim, emb_dim),
                                          nn.PReLU(),
                                          nn.Linear(emb_dim, emb_dim),
                                          nn.PReLU(),
                                          nn.Linear(emb_dim, out_dim))

        self.dropout = nn.Dropout(dropout_ratio)
        
    def forward(self, g, nfeats, efeats, eweight=None):
        nfeats = nfeats.squeeze(1)
        efeats = torch.norm(efeats, p=2, dim=1, keepdim=True)
        nh = self.model(g, nfeats, efeats)
        h = self.node_decoder(nh)
        return [], h
    
class MPNN(nn.Module):
    def __init__(
        self,
        num_layers,
        num_ntypes,
        emb_dim,
        hidden_dim,
        out_dim,
        efeat_dim,
        dropout_ratio,
        activation,
        norm_type="none",
    ):
        super().__init__()
        self.num_layers = num_layers
        self.model = MPNNGNN(node_in_feats=emb_dim, edge_in_feats=32, node_out_feats=emb_dim, edge_hidden_feats=32, num_step_message_passing=3)
        self.node_encoder = nn.Sequential(nn.Embedding(num_ntypes, emb_dim),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(emb_dim, emb_dim), 
                                          nn.ReLU(inplace=True))
        self.edge_encoder = nn.Linear(efeat_dim, 32)
        
        self.node_decoder = nn.Sequential(nn.Linear(emb_dim, emb_dim), 
                                          nn.PReLU(),
                                          nn.Linear(emb_dim, emb_dim),
                                          nn.PReLU(),
                                          nn.Linear(emb_dim, emb_dim),
                                          nn.PReLU(),
                                          nn.Linear(emb_dim, out_dim))

        self.dropout = nn.Dropout(dropout_ratio)
        
    def forward(self, g, nfeats, efeats, eweight=None):
        nfeats = nfeats.squeeze(1)
        nh = self.node_encoder(nfeats)
        eh = self.edge_encoder(efeats)
        nh = self.model(g, nh, eh)
        h = self.node_decoder(nh)
        return [], h


class Model(nn.Module):
    """
    Wrapper of different models
    """

    def __init__(self, conf):
        super(Model, self).__init__()
        self.model_name = conf["model_name"]

        if 'mlp' in conf["model_name"]:
            self.encoder = MLP(
                num_layers=conf["num_layers"],
                num_ntypes=conf["num_ntypes"],
                emb_dim=conf["emb_dim"],
                hidden_dim=conf["hidden_dim"],
                out_dim=conf["label_dim"],
                efeat_dim=conf["efeat_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
            ).to(conf["device"])

        elif 'ab_symgnn' in conf["model_name"]:
            self.encoder = Ablation_SymGNN(
                num_layers=conf["num_layers"],
                num_ntypes=conf["num_ntypes"],
                emb_dim=conf["emb_dim"],
                hidden_dim=conf["hidden_dim"],
                out_dim=conf["label_dim"],
                efeat_dim=conf["efeat_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                heads=conf["heads"],
                norm_type=conf["norm_type"],
            ).to(conf["device"])
            
        elif 'symgnn' in conf["model_name"]:
            self.encoder = SymGNN(
                num_layers=conf["num_layers"],
                num_ntypes=conf["num_ntypes"],
                emb_dim=conf["emb_dim"],
                hidden_dim=conf["hidden_dim"],
                out_dim=conf["label_dim"],
                efeat_dim=conf["efeat_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                heads=conf["heads"],
                norm_type=conf["norm_type"],
            ).to(conf["device"])
        
        elif 'gcn_edge' in conf["model_name"]:
            self.encoder = GCN_Edge(
                num_layers=conf["num_layers"],
                num_ntypes=conf["num_ntypes"],
                emb_dim=conf["emb_dim"],
                hidden_dim=conf["hidden_dim"],
                out_dim=conf["label_dim"],
                efeat_dim=conf["efeat_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                norm_type=conf["norm_type"],
            ).to(conf["device"])
            
        elif "egnn" in conf["model_name"]:
            self.encoder = EGNN(
                n_layers=conf["num_layers"],
                in_node_nf=conf["num_ntypes"],
                hidden_nf=conf["hidden_dim"],
                out_node_nf=conf["label_dim"],
            ).to(conf["device"])
            
        elif "gcn" in conf["model_name"]:
            self.encoder = GCN(
                num_layers=conf["num_layers"],
                ncate_dim=conf["num_ntypes"],
                npos_dim=3,
                emb_dim=conf["emb_dim"],
                hidden_dim=conf["hidden_dim"],
                out_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                norm_type=conf["norm_type"],
            ).to(conf["device"])
            
        elif "schnet" in conf["model_name"]:
            self.encoder = SchNet(
                num_layers=conf["num_layers"],
                num_ntypes=conf["num_ntypes"],
                emb_dim=conf["emb_dim"],
                hidden_dim=conf["hidden_dim"],
                out_dim=conf["label_dim"],
                efeat_dim=conf["efeat_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                norm_type=conf["norm_type"],
            ).to(conf["device"])
            
        elif "MGCN" in conf["model_name"]:
            self.encoder = MGCN(
                num_layers=conf["num_layers"],
                num_ntypes=conf["num_ntypes"],
                emb_dim=conf["emb_dim"],
                hidden_dim=conf["hidden_dim"],
                out_dim=conf["label_dim"],
                efeat_dim=conf["efeat_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                norm_type=conf["norm_type"],
            ).to(conf["device"])
            
        elif "MPNN" in conf["model_name"]:
            self.encoder = MPNN(
                num_layers=conf["num_layers"],
                num_ntypes=conf["num_ntypes"],
                emb_dim=conf["emb_dim"],
                hidden_dim=conf["hidden_dim"],
                out_dim=conf["label_dim"],
                efeat_dim=conf["efeat_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                norm_type=conf["norm_type"],
            ).to(conf["device"])
            
        elif "QM9" in conf["model_name"]:
            self.encoder = QM9(
                num_layers=conf["num_layers"],
                num_ntypes=conf["num_ntypes"],
                emb_dim=conf["emb_dim"],
                hidden_dim=conf["hidden_dim"],
                out_dim=conf["label_dim"],
                efeat_dim=conf["efeat_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                heads=conf["heads"],
                norm_type=conf["norm_type"],
            ).to(conf["device"])
            
        # elif "dimenet" in conf["model_name"]:
        #     self.encoder = DimeNet_pp(
        #         num_layers=conf["num_layers"],
        #         num_ntypes=conf["num_ntypes"],
        #         emb_dim=conf["emb_dim"],
        #         hidden_dim=conf["hidden_dim"],
        #         out_dim=conf["label_dim"],
        #         efeat_dim=conf["efeat_dim"],
        #         dropout_ratio=conf["dropout_ratio"],
        #         activation=F.relu,
        #         norm_type=conf["norm_type"],
        #     ).to(conf["device"])

    def forward(self, data, ncate, npos, embed=False, eweight=None):
        '''
        ncate: categorical node feature
        npos: node position, usually 3D coordinates
        '''
        return self.encoder(data, ncate, npos, eweight, embed)[1]
    
    def inference(self, data, ncate, npos):
        return self.forward(data, ncate, npos)
