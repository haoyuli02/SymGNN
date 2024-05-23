import os
import argparse
import torch
import copy
import pickle
import logging
import pytz
import random
import dgl
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import torch.optim as optim
from pathlib import Path
from models import Model
from utils import set_seed, get_logger
from gnnexplainer import GNNExplainer
from pgexplainer import PGExplainer
from models import Model
# import wandb
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

import yaml
import pickle

def get_args():
    parser = argparse.ArgumentParser(description="PyTorch DGL implementation")
    parser.add_argument("--device", type=int, default=-1, help="CUDA device, -1 means CPU")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--log_level",
        type=int,
        default=20,
        help="Logger levels for run {10: DEBUG, 20: INFO, 30: WARNING}",
    )
    parser.add_argument(
        "--console_log",
        action="store_true",
        help="Set to True to display log info in console",
    )
    parser.add_argument(
        "--output_path", type=str, default="outputs", help="Path to save outputs"
    )
    parser.add_argument(
        "--output_dir", type=str, help="Path to save score curve and training results"
    )
    parser.add_argument(
        "--num_exp", type=int, default=1, help="Repeat how many experiments"
    )
    parser.add_argument(
        "--eval_interval", type=int, default=1, help="Evaluate once per how many epochs"
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Set to True to save the loss curves and trained model",
    )

    """Dataset"""
    parser.add_argument("--dataset_dir", type=str, default="../datasets", help="Path to dataset")
    parser.add_argument(
        "--split_rate",
        type=float,
        default=0.9,
        help="Rate for graph split, see comment of graph_split for more details",
    )
    
    """Model"""
    parser.add_argument(
        "--model_config_path",
        type=str,
        default="./train.conf.yaml",
        help="Path to model configeration",
    )
    parser.add_argument("--model_name", type=str, default="SAGE", help="Model")
    parser.add_argument(
        "--num_layers", type=int, default=4, help="Model number of layers"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="Model hidden layer dimensions"
    )
    parser.add_argument("--dropout_ratio", type=float, default=0)
    parser.add_argument(
        "--norm_type", type=str, default="none", help="One of [none, batch, layer]"
    )

    """SAGE Specific"""
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument(
        "--fan_out",
        type=str,
        default="5,5",
        help="Number of samples for each layer in SAGE. Length = num_layers",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of workers for sampler"
    )

    """Optimization"""
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.0000)
    parser.add_argument(
        "--max_epoch", type=int, default=500, help="Evaluate once per how many epochs"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stop is the score on validation set does not improve for how many epochs",
    )

    args = parser.parse_args() # Actual python file use

    # args = parser.parse_args('') # Use in notebook

    return args

def get_training_config(config_path, model_name):
    with open(config_path, "r") as conf:
        full_config = yaml.load(conf, Loader=yaml.FullLoader)
    specific_config = full_config["global"]
    model_specific_config = full_config[model_name]
    if model_specific_config is not None:
        specific_config = dict(specific_config, **model_specific_config)

    specific_config["model_name"] = model_name
    return specific_config

def plot_3d_local_graph(g, central_node, saved_dir=None, layout=None, layout_seed=0, node_size=0.5, edge_kwargs={}, selected_edges=None, selected_edges_kwargs={}, label='nid', legend=False):
    nx_graph = dgl.to_networkx(g.cpu(), node_attrs=['type', 'position', 'ID'], edge_attrs=['cross_position', 'distances'])
    print(nx_graph)
    new_central = -1
    for node in range(nx_graph.number_of_nodes()):
        if nx_graph.nodes[node]['ID'] == central_node:
            new_central = node
            break
    
    pos = {}
    for node in nx_graph.nodes():
        pos[node] = np.array(nx_graph.nodes[node]['position'])
        
    nodes_connected = nx_graph.edges(new_central)
    distances = {}
    for node in nodes_connected:
        distances[node] = nx_graph.get_edge_data(node[0], node[1])[0]['distances'].item()
    
    closest = sorted(distances, key=distances.get)[:10]
    close_nodes = []
    for u, v in closest:
        if u == new_central:
            close_nodes.append(v)
        else:
            close_nodes.append(u)
        
    def _format_axes(ax):
        ax.grid(False)
        ax.set_xlabel("x_1", fontsize=16)
        ax.set_ylabel("x_2", fontsize=16)
        ax.set_zlabel("x_3", fontsize=16)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    
    central_node_xyz = np.array([pos[central_node]])
    close_node_xyz = np.array([pos[close_node] for close_node in close_nodes])
    edge_between_close = []
    close_nodes_and_central = [new_central] + close_nodes
    for u, v, a in nx_graph.edges(data=True):
        if (u in close_nodes_and_central) and (v in close_nodes_and_central):
            edge_between_close.append((u, v))
    print(f'Total Edges: {len(edge_between_close)}')
    
    top_important_edges = []
    top_important_nodes = []
    for edge_id in selected_edges:
        for u, v, a in nx_graph.edges(data=True):
            if a['id'] == edge_id:
                if u == v:
                    top_important_nodes.append(u)
                else:
                    top_important_edges.append((u, v))
    
    important_and_close_nodes = []
    important_and_close_edges = []
    for u, v in top_important_edges:
        if u != v and (((u, v) in edge_between_close) or ((v, u) in edge_between_close)):
            important_and_close_edges.append((u, v))
    for u, v in top_important_edges:
        if u == v and ((u, u) in edge_between_close):
            important_and_close_nodes.append(u)

    top_important_xyz = np.array([pos[u] for u in important_and_close_nodes])
    all_edge_xyz = np.array([(pos[u], pos[v]) for u, v in edge_between_close])
    important_edge_xyz = np.array([(pos[u], pos[v]) for u, v in important_and_close_edges])
    
    if top_important_xyz.shape[0] > 0:
        ax.scatter(*top_important_xyz.T, s=60, color='green', label='Self Loop')
    ax.scatter(*central_node_xyz.T, s=100, color='red', label='Central Node being Explained')
    ax.scatter(*close_node_xyz.T, s=40, color='orange', label='Closest 10 Nodes')
    
    for i in range(len(all_edge_xyz)):
        if i != len(all_edge_xyz) - 1:
            ax.plot(*(all_edge_xyz[i]).T, linestyle='dotted', color='tab:grey')
        else:
            ax.plot(*(all_edge_xyz[i]).T, linestyle='dotted', color='tab:grey', label='Existing Edges')
            
    for j in range(len(important_edge_xyz)):
        if j != len(important_edge_xyz) - 1:
            ax.plot(*(important_edge_xyz[j]).T, linewidth=5.0, color='tab:blue')
        else:
            ax.plot(*(important_edge_xyz[j]).T, linewidth=5.0, color='tab:blue', label='Selected Edges')
    
    _format_axes(ax)
    plt.title(f'Node {central_node} Explanation')
    plt.legend(fontsize=16)
    plt.show()
    
    if saved_dir != None:
        plt.savefig(os.path.join(saved_dir, 'explain.png'), dpi=200)
        
    plt.close()
        
    return nx_graph

def get_hop(g, central_node, selected_edges):
    nx_graph = dgl.to_networkx(g.cpu(), node_attrs=['type', 'position', 'ID'], edge_attrs=['cross_position', 'distances'])
    new_central = -1
    for node in range(nx_graph.number_of_nodes()):
        if nx_graph.nodes[node]['ID'] == central_node:
            new_central = node
            break
        
    bfs_tree = nx.bfs_tree(nx_graph, source=new_central)
    hop_distances = {node: nx.shortest_path_length(bfs_tree, source=new_central, target=node) for node in nx_graph.nodes()}
    
    edge_hop_distances = []
    for edge_id in selected_edges:
        for u, v, a in nx_graph.edges(data=True):
            if a['id'] == edge_id:
                source_dist = hop_distances.get(u, float('inf'))
                target_dist = hop_distances.get(v, float('inf'))
                relative_hop = max(source_dist, target_dist)
                edge_hop_distances.append(relative_hop)
                
    return edge_hop_distances

def plot_3d_graph(g, central_node, saved_dir=None, layout=None, layout_seed=0, node_size=0.5, edge_kwargs={}, selected_edges=None, selected_edges_kwargs={}, label='nid', legend=False):
    nx_graph = dgl.to_networkx(g.cpu(), node_attrs=['type', 'position', 'ID'], edge_attrs=['cross_position', 'distances'])
    new_central = -1
    for node in range(nx_graph.number_of_nodes()):
        if nx_graph.nodes[node]['ID'] == central_node:
            new_central = node
            break
    
    pos = {}
    for node in nx_graph.nodes():
        pos[node] = np.array(nx_graph.nodes[node]['position'])
    
    nodes_connected = nx_graph.edges(new_central)
    distances = {}
    for node in nodes_connected:
        distances[node] = nx_graph.get_edge_data(node[0], node[1])[0]['distances'].item()
    
    closest = sorted(distances, key=distances.get)[:10]
    close_nodes = []
    for u, v in closest:
        if u == new_central:
            close_nodes.append(v)
        else:
            close_nodes.append(u)
        
    def _format_axes(ax):
        ax.grid(False)
        ax.set_xlabel("x_1", fontsize=16)
        ax.set_ylabel("x_2", fontsize=16)
        ax.set_zlabel("x_3", fontsize=16)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    
    central_node_xyz = np.array([pos[central_node]])
    close_node_xyz = np.array([pos[close_node] for close_node in close_nodes])
    
    top_important_edges = []
    top_important_nodes = []
    for edge_id in selected_edges:
        for u, v, a in nx_graph.edges(data=True):
            if a['id'] == edge_id:
                if u == v:
                    top_important_nodes.append(u)
                else:
                    top_important_edges.append((u, v))
    
    top_important_xyz = np.array([pos[v] for v in top_important_nodes])
    nodes_needs_plot = list(set(sorted(nx_graph)) - (set(top_important_nodes) | set(close_nodes)))
    node_xyz = np.array([pos[v] for v in nodes_needs_plot])
    edge_xyz = np.array([(pos[u], pos[v]) for u, v in top_important_edges])
    
    ax.scatter(*node_xyz.T, s=20, ec="w")
    if top_important_xyz.shape[0] > 0:
        ax.scatter(*top_important_xyz.T, s=60, color='green', label='Self Loop')
    ax.scatter(*central_node_xyz.T, s=100, color='red', label='Central Node being Explained')
    ax.scatter(*close_node_xyz.T, s=40, color='orange', label='Closest 10 Nodes')
    
    for i in range(len(edge_xyz)):
        if i != len(edge_xyz) - 1:
            ax.plot(*(edge_xyz[i]).T, color='tab:grey')
        else:
            ax.plot(*(edge_xyz[i]).T, color='tab:grey', label='Selected Edges')
    
    _format_axes(ax)
    plt.title(f'Node {central_node} Explanation')
    plt.legend(fontsize=16)
    plt.show()
    
    if saved_dir != None:
        plt.savefig(os.path.join(saved_dir, 'explain.png'), dpi=200)
        
    plt.close()
        
    return nx_graph

if __name__ == "__main__":
    set_seed(0)
    args = get_args()
    args.num_ntypes = 2
    args.efeat_dim = 3
    args.label_dim = 1
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device("cuda:" + str(args.device))
    else:
        device = "cpu"
        
    # logger = get_logger(args.output_dir, "a.log", args.console_log, args.log_level)
    
    print(args.model_name)
    conf = {}
    if args.model_config_path is not None:
        conf = get_training_config(args.model_config_path, args.model_name)
    conf = dict(args.__dict__, **conf)
    conf["device"] = device
    conf["heads"] = [8, 4, 2, 1]
    
    args.dataset_dir = './datasets/dgl_graphs/dimenet'
    loading_dgl = True
    num_train_graphs = 7
    num_test_graphs = 2


    test_ids = range(2)
    test_path = str(Path.cwd().joinpath(args.dataset_dir, f'test_graphs'))

    test_graphs, test_labels_dict = dgl.load_graphs(test_path)
    test_labels = [test_labels_dict[str(i)] for i in test_ids]
    test_graphs = test_graphs[:num_test_graphs]

    graphs_test = test_graphs
    # logger.info(f"graphs: {graphs}")

    test_graphs = dgl.batch(test_graphs)
    test_labels = torch.cat(test_labels)

    test_data  = [(test_graphs, test_labels)]
        
    model = Model(conf)
    model.load_state_dict(torch.load('./saved_models/schnet_best'))
    model.to(device)
    model.eval()
    
    g = graphs_test[0].to(device)
    close_to_zero_dist = []
    close_to_zero_node = []
    min_dist = 1000000000
    min_node = -1
    for node in g.nodes():
        pos = g.ndata['positions'][node]
        dist = torch.norm(pos, 2)
        close_to_zero_dist.append(dist.item())
        close_to_zero_node.append(node.item())
        if dist < min_dist:
            min_dist = dist
            min_node = node
    
    close_to_zero_dist = np.array(close_to_zero_dist)
    close_to_zero_node = np.array(close_to_zero_node)
    close = np.argsort(close_to_zero_dist)
    node_to_use = close_to_zero_node[close][:50]
    
    edge_mask_full = {}
    info_full = {}
    distances = []
    # for i in tqdm(range(len(node_to_use))):
    for i in range(len(node_to_use)):
        ind = torch.tensor(node_to_use[i]).to(device)
        explainer = GNNExplainer(model, num_hops=4, num_epochs=500)
        explainer.to(device)
        test_graph = graphs_test[0].to(device)
        features = graphs_test[0].ndata['type'].to(device)
        edge_features = graphs_test[0].edata['cross_position'].to(device)
        new_node_id, sg, feat_mask, edge_mask = explainer.explain_node(ind, test_graph, features.unsqueeze(1), **{'npos': edge_features})
        edge_mask_full[node_to_use[i]] = {'sg': sg, 'new_node_id': new_node_id.item(), 'edge_mask': edge_mask.tolist()}
        sg.ndata['ID'] = torch.arange(sg.number_of_nodes()).to(device)
        num_edges = len(edge_mask)
        selected_edges = np.flip(torch.argsort(edge_mask).cpu().numpy())[(num_edges - int(0.001 * num_edges)):]
        info_full[ind] = {'node_id': new_node_id, 'sg': sg, 'feat_mask': feat_mask, 'edge_mask': edge_mask}
    
    torch.save(info_full, './explain/new_schnet_explain')
         
## python3 explain.py --model_name='schnet' --device=0
                
