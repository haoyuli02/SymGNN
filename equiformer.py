import torch
# import dgl
import sys
from tqdm import tqdm
from dp_attention_transformer_oc20 import DotProductAttentionTransformerOC20
from ocpmodels.models.painn.painn import PaiNN
import pickle
import logging
import numpy as np
import copy
import os
from utils import set_seed, get_logger
import torch.optim as optim

from torch_geometric.loader import DataLoader

# model = DotProductAttentionTransformerOC20(
#             num_atoms=None,
#             bond_feat_dim=None,
#             num_targets=None,
#             irreps_node_embedding='256x0e+128x1e', num_layers=3,
#             irreps_node_attr='1x0e', use_node_attr=False,
#             irreps_sh='1x0e+1x1e',
#             max_radius=5.0,
#             number_of_basis=128, fc_neurons=[64, 64], 
#             use_atom_edge_attr=False, irreps_atom_edge_attr='8x0e',
#             irreps_feature='512x0e',
#             irreps_head='32x0e+16x1e', num_heads=8, irreps_pre_attn=None,
#             rescale_degree=False, nonlinear_message=False,
#             irreps_mlp_mid='768x0e+384x1e',
#             norm_layer='layer',
#             alpha_drop=0.2, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
#             use_auxiliary_task=False,
#             otf_graph=False, use_pbc=False, max_neighbors=50)

def correlation(preds, labels):
    # Criterion for evaluate
    return torch.corrcoef(torch.cat([preds.view(1, -1), labels.view(1, -1)]))[0, 1].item()

def evaluate(model, data, labels, criterion, evaluator, idx_eval=None):
    """
    Returns:
    loss & score (float): evaluated loss & score, if idx_eval is not None, only loss & score on those idx.
    """
    # print(model.device(), data.device(), labels.device())
    model.eval()
    with torch.no_grad():
        out = model(data).squeeze(1)
        if idx_eval is None:
            loss = criterion(out, labels)
            score = evaluator(out, labels)
        else:
            loss = criterion(out[idx_eval], labels[idx_eval])
            score = evaluator(out[idx_eval], labels[idx_eval])
            
    return out, loss.item(), score

def train(model, data, labels, criterion, optimizer):
    """
    GNN full-batch training.
    """
    model.train()
    
    out = model(data).squeeze(1)
    # print(out.size(), labels.size())
    loss = criterion(out, labels)
    
    optimizer.zero_grad()
    loss.backward()
    
    # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    # print(optimizer.state_dict())
    return loss.item()

def run(
    conf,
    model,
    train_loader,
    test_loader,
    val_loader,
    criterion,
    evaluator,
    optimizer,
    logger,
    apply_random_rotation=False
):
    """
    Train and eval under the inductive setting.
    """
    # set_seed(conf["seed"]) ### Conf for configure
    device = torch.device(f"cuda:{2}")
    model = model.to(device)
    state = None
    train_score, test_score, val_score = [], [], []
    best_epoch, best_val_score, best_test_score, count = 0, 0, 0, 0
    cur_epoch = 0
    for epoch in range(1, conf['max_epoch'] + 1):
        loss = 0
        for data in train_loader:
            # print(data)
            data = data.to(device)
            labels = data.y
            labels = labels.to(device)
            loss += train(model, data, labels, criterion, optimizer)
             
        if epoch % conf["eval_interval"] == 0:
            
            # Train data eval
            loss_train_all = []
            score_train_all = []
            for data in train_loader:
                data = data.to(device)
                labels = data.y
                labels = labels.to(device)

                out, loss_train, score_train = evaluate(
                    model, data, labels, criterion, evaluator
                )
                loss_train_all += [loss_train]
                score_train_all += [score_train]
             
            loss_train = np.mean(loss_train_all)
            score_train = np.mean(score_train_all)
            train_score += [score_train]
            
            # Validation data eval
            loss_val_all = []
            score_val_all = []
            for data in val_loader:
                data = data.to(device)
                labels = data.y
                labels = labels.to(device)

                out, loss_val, score_val = evaluate(
                    model, data, labels, criterion, evaluator
                )
                loss_val_all += [loss_val]
                score_val_all += [score_val]
             
            loss_val = np.mean(loss_val_all)
            score_val = np.mean(score_val_all)
            val_score += [score_val]
            
            # Test data eval
            loss_test_all = []
            score_test_all = []
            for data in test_loader:
                data = data.to(device)
                labels = data.y
                labels = labels.to(device)
                
                # Evaluate the inductive part with the full graph
                out, loss_test, score_test = evaluate(
                    model, data, labels, criterion, evaluator
                )
                loss_test_all += [loss_test]
                score_test_all += [score_test]
                
            loss_test = np.mean(loss_test_all)
            score_test = np.mean(score_test_all)
            test_score += [score_test]

            logger.debug(
                f"Ep {epoch:3d} | l_tr: {loss_train:.4f} | s_tr: {score_train:.4f} | l_tt: {loss_test:.4f} | s_tt: {score_test:.4f} | l_vl: {loss_val:.4f} | s_vl: {score_val:.4f}"
            )

            if score_val >= best_val_score:
                best_epoch = epoch
                best_val_score = score_val
                best_test_score = score_test
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        cur_epoch = epoch
        if count == conf["patience"]:
            break
    
    best_model = f"Best model at epoch: {best_epoch :3d}, best test score: {best_test_score:.4f}, best val score: {best_val_score}"
    logger.info(best_model)
    
    # os.makedirs('saved_models', exist_ok=True)
    # torch.save(state, './saved_models/best')
    # logger.info('Best model saved')
            
    return score_test

if __name__ == '__main__':
    output_dir = './outputs/equiformer_again'
    console_log = True
    log_level = 10
    logger = get_logger(output_dir, "a.log", True, log_level)
    
    gpu = 0
    train_graphs = torch.load('./datasets/torch_graphs/3/train_graphs')
    test_graphs = torch.load('./datasets/torch_graphs/3/test_graphs')
    val_graphs = torch.load('./datasets/torch_graphs/3/val_graphs')

    train_cells = []
    train_labels = []
    for i in range(6):
        with open(f'datasets/raw_data/train{i}.pickle', 'rb') as file:
            data = pickle.load(file)
            box = data['box']
            labels = torch.from_numpy(data['trajectory_start_velocities']).view(-1, 1)
            labels = (labels - 1.6346) / 0.3104
            train_labels += labels
            train_cells.append([[box[0], 0, 0], [0, box[0], 0], [0, 0, box[0]]])

    test_cells = []
    test_labels = []
    for i in range(2):
        with open(f'datasets/raw_data/test{i}.pickle', 'rb') as file:
            data = pickle.load(file)
            box = data['box']
            labels = torch.from_numpy(data['trajectory_start_velocities']).view(-1, 1)
            labels = (labels - 1.6346) / 0.3104
            test_labels += labels
            test_cells.append([[box[0], 0, 0], [0, box[0], 0], [0, 0, box[0]]])
            
    val_cells = []
    val_labels = []
    for i in range(6,7):
        with open(f'datasets/raw_data/train{i}.pickle', 'rb') as file:
            data = pickle.load(file)
            box = data['box']
            labels = torch.from_numpy(data['trajectory_start_velocities']).view(-1, 1)
            labels = (labels - 1.6346) / 0.3104
            val_labels += labels
            val_cells.append([[box[0], 0, 0], [0, box[0], 0], [0, 0, box[0]]])

    for i, data in tqdm(enumerate(train_graphs[:200]), total=len(train_graphs[:200])):
        data.pos = data.positions.clone().float()
        data.R = data.positions.clone().float()
        data.Z = data.type.clone()
        data.d = torch.norm(data.cross_position, p=2, dim=1, keepdim=True).float()
        data.o = data.cross_position.clone().float()
        data.cell = torch.tensor(train_cells[i // 8000]).float().unsqueeze(0)
        data.num_nodes = data.positions.size(0)
        data.natoms = data.positions.size(0)
        data.atomic_numbers = data.type.clone() 
        data.y = train_labels[i]
    
    for j, data in tqdm(enumerate(test_graphs[:200]), total=len(test_graphs[:200])):
        data.pos = data.positions.clone().float()
        data.R = data.positions.clone().float()
        data.Z = data.type.clone()
        data.d = torch.norm(data.cross_position, p=2, dim=1, keepdim=True).float()
        data.o = data.cross_position.clone().float()
        data.cell = torch.tensor(test_cells[j // 3000]).float().unsqueeze(0)
        data.num_nodes = data.positions.size(0)
        data.natoms = data.positions.size(0)
        data.atomic_numbers = data.type.clone() 
        data.y = test_labels[j]
        
    for k, data in tqdm(enumerate(val_graphs[:200]), total=len(val_graphs[:200])):
        data.pos = data.positions.clone().float()
        data.R = data.positions.clone().float()
        data.Z = data.type.clone()
        data.d = torch.norm(data.cross_position, p=2, dim=1, keepdim=True).float()
        data.o = data.cross_position.clone().float()
        data.cell = torch.tensor(val_cells[k // 8000]).float().unsqueeze(0)
        data.num_nodes = data.positions.size(0)
        data.natoms = data.positions.size(0)
        data.atomic_numbers = data.type.clone() 
        data.y = val_labels[k]
        
    print(train_graphs[0])

    train_loader = DataLoader(train_graphs[:200], batch_size=2, shuffle=True)
    test_loader = DataLoader(train_graphs[:200], batch_size=2, shuffle=True)
    val_loader = DataLoader(val_graphs[:200], batch_size=2, shuffle=True)
    
    print(train_loader.dataset[0])
    
    set_seed(0)
    model = DotProductAttentionTransformerOC20(
            num_atoms=None,
            bond_feat_dim=None,
            num_targets=None,
            irreps_node_embedding='64x0e+32x1e', num_layers=3,
            irreps_node_attr='1x0e', use_node_attr=False,
            irreps_sh='1x0e+1x1e',
            max_radius=5.0,
            number_of_basis=32, fc_neurons=[64, 64], 
            use_atom_edge_attr=False, irreps_atom_edge_attr='8x0e',
            irreps_feature='1x0e',
            irreps_head='32x0e+16x1e', num_heads=4, irreps_pre_attn=None,
            rescale_degree=False, nonlinear_message=False,
            irreps_mlp_mid='256x0e+128x1e',
            norm_layer='layer',
            alpha_drop=0.2, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
            use_auxiliary_task=False,
            otf_graph=True, use_pbc=True, max_neighbors=100)
    
    # model = PaiNN(
    #         num_atoms = None,
    #         bond_feat_dim = None,
    #         num_targets = None,
    #         hidden_channels = 64,
    #         num_layers = 3,
    #         num_rbf = 32,
    #         cutoff = 5.0,
    #         max_neighbors = 80,
    #         rbf = {"name": "gaussian"},
    #         envelope = {
    #             "name": "polynomial",
    #             "exponent": 5,
    #         },
    #         regress_forces = False,
    #         direct_forces = False,
    #         use_pbc = True,
    #         otf_graph = True,
    #         num_elements = 2,
    #         scale_file = None,
    #     )
    
    optimizer = optimizer = optim.Adam(
        model.parameters(), lr=0.0001, weight_decay=0
    )
    
    criterion = torch.nn.MSELoss()
    evaluator = correlation
    # logger.info(f'model: {model}')
    
    score_test = run(
        {'max_epoch': 400, 'eval_interval': 1, 'patience': 20},
        model,
        train_loader,
        test_loader,
        val_loader,
        criterion,
        evaluator,
        optimizer,
        logger,
    )

# for data in train_loader:
#     data = data.to(device)
#     outputs = model(data)
#     print(outputs)
#     exit(1)
