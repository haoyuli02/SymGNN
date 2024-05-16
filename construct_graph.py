from pathlib import Path
import pickle
import torch
import numpy as np
import dgl

'''
Data functions
'''
def load_data(dataset_path):
    """
    Return a graph and its target
    """
    ### rb means read binary
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)

    graph = construct_dgl_graph(data, edge_threshold=5)
    ### Function view is to change layout format of a tensor
    labels = torch.from_numpy(data['trajectory_start_velocities']).view(-1, 1)
    labels = (labels - 1.6346) / 0.3104
    return graph, labels

def construct_dgl_graph(data, edge_threshold=5):
    positions = data['positions']
    node_types = data['types']
    num_nodes = node_types.shape[0]
    box = data['box']
    
    # Calculate pairwise relative distances between particles: shape [n, n, 3].
    cross_positions = positions[None, :, :] - positions[:, None, :]
    # Enforces periodic boundary conditions.
    box_ = box[None, None, :]
    cross_positions += (cross_positions < -box_ / 2.).astype(np.float32) * box_
    cross_positions -= (cross_positions > box_ / 2.).astype(np.float32) * box_

    # Calculates distances and cut off using edge_threshold.
    distances = np.linalg.norm(cross_positions, axis=-1)
    edges = np.where(distances < edge_threshold)

    ### Add edge if distances smaller than some threshold.
    g = dgl.graph(edges, num_nodes=num_nodes)
    # g.ndata['type'] = torch.from_numpy(node_types).view(-1, 1)
    # g.ndata['type'] = torch.from_numpy(node_types).view(-1, 1).int() - 1
    g.ndata['positions'] = torch.from_numpy(positions)
    g.ndata['type'] = torch.from_numpy(node_types).int() - 1
    g.edata['distances'] = torch.from_numpy(distances[edges]).view(-1, 1).float()
    g.edata['cross_position'] = torch.from_numpy(cross_positions[edges]).view(-1, 3).float()

    return g


'''
Graphs in DGL format
'''

train_ids = range(7)
train_graphs = []
train_labels = []

dataset_dir = './datasets/raw_data'
train_dataset_names = ['train' + str(i) for i in train_ids]
for train_dataset_name in train_dataset_names:
    train_dataset_path = Path.cwd().joinpath(dataset_dir, f'{train_dataset_name}.pickle')
    g, labels = load_data(train_dataset_path)
    train_graphs += [g]
    train_labels += [labels]
    
test_ids = range(2)
test_dataset_names = ['test' + str(i) for i in test_ids]
test_graphs = []
test_labels = []

for test_dataset_name in test_dataset_names:
    test_dataset_path = Path.cwd().joinpath(dataset_dir, f'{test_dataset_name}.pickle')
    g, labels = load_data(test_dataset_path)
    test_graphs += [g]
    test_labels += [labels]
    
print(test_graphs)

saving_dgl = True
# saving_dgl = True
loading_dgl = False
num_train_graphs = 7
num_test_graphs = 2
dataset_dir = './datasets/dgl_graphs/dimenet'
train_ids = range(7)[:num_train_graphs]
# train_graphs = []
# train_labels = []
train_path = str(Path.cwd().joinpath(dataset_dir, f'train_graphs'))

test_ids = range(2)[:num_test_graphs]
# test_graphs = []
# test_labels = []
test_path = str(Path.cwd().joinpath(dataset_dir, f'test_graphs'))
    
if saving_dgl:
    dgl.save_graphs(train_path, 
                    train_graphs, 
                    labels={str(i): train_labels[i] for i in train_ids})
    
    dgl.save_graphs(test_path, 
                    test_graphs, 
                    labels={str(i): test_labels[i] for i in test_ids})

if loading_dgl:
    train_graphs, train_labels_dict = dgl.load_graphs(train_path)
    train_labels = [train_labels_dict[str(i)] for i in train_ids]

    test_graphs, test_labels_dict = dgl.load_graphs(test_path)
    test_labels = [test_labels_dict[str(i)] for i in test_ids]

graphs_train = train_graphs
graphs_test = test_graphs

train_graphs = dgl.batch(train_graphs)
train_labels = torch.cat(train_labels)
test_graphs = dgl.batch(test_graphs)
test_labels = torch.cat(test_labels)

train_data = [(train_graphs, train_labels)]
test_data  = [(test_graphs, test_labels)]
