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

import torch.optim as optim
from pathlib import Path
from models import Model
from utils import set_seed, get_logger
from matplotlib import pyplot as plt
import time

import yaml

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

    args = parser.parse_args() 

    return args

'''
Train and Eval functions
'''
def get_training_config(config_path, model_name):
    with open(config_path, "r") as conf:
        full_config = yaml.load(conf, Loader=yaml.FullLoader)
    specific_config = full_config["global"]
    model_specific_config = full_config[model_name]
    if model_specific_config is not None:
        specific_config = dict(specific_config, **model_specific_config)

    specific_config["model_name"] = model_name
    return specific_config

def train(model, graph, nfeats, efeats, labels, criterion, optimizer, clip=0.1):
    """
    GNN full-batch training.
    """
    model.train()
    
    out = model(graph, nfeats, efeats)
    # out = out.squeeze()
    # print(out.size())
    # print(out.size(), labels.size())
    loss = criterion(out, labels)
    optimizer.zero_grad()
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    # print(optimizer.state_dict())
    return loss.item()

def evaluate(model, graph, nfeats, efeats, labels, criterion, evaluator, idx_eval=None):
    """
    Returns:
    loss & score (float): evaluated loss & score, if idx_eval is not None, only loss & score on those idx.
    """
    model.eval()
    with torch.no_grad():
        out = model.inference(graph, nfeats, efeats)
        # out = out.squeeze()
        if idx_eval is None:
            loss = criterion(out, labels)
            score = evaluator(out, labels)
        else:
            loss = criterion(out[idx_eval], labels[idx_eval])
            score = evaluator(out[idx_eval], labels[idx_eval])
            
    return out, loss.item(), score

def correlation(preds, labels):
    # Criterion for evaluate
    # print("Corr Between:")
    # print(preds, labels)
    return torch.corrcoef(torch.cat([preds.view(1, -1), labels.view(1, -1)]))[0, 1].item()

'''
GCN run
'''
def run(
    conf,
    model,
    train_data,
    test_data,
    val_data,
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
    device = conf["device"]

    state = None
    train_score, test_score, val_score = [], [], []
    best_epoch, best_val_score, best_test_score, count = 0, 0, 0, 0
    cur_epoch = 0
    for epoch in range(1, conf["max_epoch"] + 1):
        loss = 0
        for graph, labels in train_data:
            graph  = graph.to(device)
            if conf['model_name'] == 'egnn':
                nfeats = graph.ndata["type"].to(device)
            else:
                nfeats = graph.ndata["type"].to(device).unsqueeze(1)
                # print(nfeats.shape)
                 
            if conf['model_name'] in ['mlp', 'egnn', 'gcn']:
                efeats = graph.ndata["positions"].float().to(device)
            else:
                if conf['simple_invariance']:
                    efeats = torch.norm(graph.edata["cross_position"].to(device), dim=1, keepdim=True)
                else:
                    efeats = graph.edata["cross_position"].to(device)
                
            labels = labels.to(device)
            
            loss += train(model, graph, nfeats, efeats, labels, criterion, optimizer)
             
        if epoch % conf["eval_interval"] == 0:
            
            # Train data eval
            loss_train_all = []
            score_train_all = []
            for graph, labels in train_data:
                graph  = graph.to(device)
                if conf['model_name'] == 'egnn':
                    nfeats = graph.ndata["type"].to(device)
                else:
                    nfeats = graph.ndata["type"].to(device).unsqueeze(1)
                    
                if conf['model_name'] in ['mlp', 'egnn', 'gcn']:
                    efeats = graph.ndata["positions"].float().to(device)
                else:
                    if conf['simple_invariance']:
                        efeats = torch.norm(graph.edata["cross_position"].to(device), dim=1, keepdim=True)
                    else:
                        efeats = graph.edata["cross_position"].to(device)
                    
                labels = labels.to(device)

                out, loss_train, score_train = evaluate(
                    model, graph, nfeats, efeats, labels, criterion, evaluator
                )
                loss_train_all += [loss_train]
                score_train_all += [score_train]
             
            loss_train = np.mean(loss_train_all)
            score_train = np.mean(score_train_all)
            train_score += [score_train]
            
            # Validation data eval
            loss_val_all = []
            score_val_all = []
            for graph, labels in val_data:
                graph  = graph.to(device)
                if conf['model_name'] == 'egnn':
                    nfeats = graph.ndata["type"].to(device)
                else:
                    nfeats = graph.ndata["type"].to(device).unsqueeze(1)
                if conf['model_name'] in ['mlp', 'egnn', 'gcn']:
                    efeats = graph.ndata["positions"].float().to(device)
                else:
                    if conf['simple_invariance']:
                        efeats = torch.norm(graph.edata["cross_position"].to(device), dim=1, keepdim=True)
                    else:
                        efeats = graph.edata["cross_position"].to(device)
                labels = labels.to(device)

                out, loss_val, score_val = evaluate(
                    model, graph, nfeats, efeats, labels, criterion, evaluator
                )
                loss_val_all += [loss_val]
                score_val_all += [score_val]
             
            loss_val = np.mean(loss_val_all)
            score_val = np.mean(score_val_all)
            val_score += [score_val]
            
            # Test data eval
            loss_test_all = []
            score_test_all = []
            for graph, labels in test_data:
                graph  = graph.to(device)
                
                if conf['model_name'] == 'egnn':
                    nfeats = graph.ndata["type"].to(device)
                else:
                    nfeats = graph.ndata["type"].to(device).unsqueeze(1)
                    
                if conf['model_name'] in ['mlp', 'egnn', 'gcn']:
                    efeats = graph.ndata["positions"].float().to(device)
                else:
                    if conf['model_name'] in ['schnet', 'MGCN']:
                        efeats = torch.norm(graph.edata["cross_position"].to(device), dim=1, keepdim=True)
                    else:
                        efeats = graph.edata["cross_position"].to(device)
                        
                labels = labels.to(device)
                
                # Evaluate the inductive part with the full graph
                out, loss_test, score_test = evaluate(
                    model, graph, nfeats, efeats, labels, criterion, evaluator
                )
                loss_test_all += [loss_test]
                score_test_all += [score_test]
                
            loss_test = np.mean(loss_test_all)
            score_test = np.mean(score_test_all)
            test_score += [score_test]

            logger.debug(
                f"Ep {epoch:3d} | l_tr: {loss_train:.4f} | s_tr: {score_train:.4f} | l_tt: {loss_test:.4f} | s_tt: {score_test:.4f} | l_vl: {loss_val:.4f} | s_vl: {score_val:.4f}"
            )

            if score_test >= best_test_score:
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
    
    os.makedirs('saved_models', exist_ok=True)
    torch.save(state, './saved_models/schnet_best')
    logger.info('Best model saved')
            
    return score_test

if __name__ == "__main__":
    args = get_args()
    args.num_ntypes = 2
    args.efeat_dim = 3
    args.label_dim = 1
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device("cuda:" + str(args.device))
    else:
        device = "cpu"
        
    # device = 'cpu'

    logger = get_logger(args.output_dir, "a.log", args.console_log, args.log_level)
    
    conf = {}
    if args.model_config_path is not None:
        conf = get_training_config(args.model_config_path, args.model_name)
    conf = dict(args.__dict__, **conf)
    
    conf['model_name'] = args.model_name
    conf["device"] = device
    conf["norm_type"] = 'batch'
    # conf["norm_type"] = None
    conf["simple_invariance"] = False
    if 'symgnn' in args.model_name:
        conf["heads"] = [8, 4, 2, 1]
        conf["simple_invariance"] = False
    logger.info(f"conf: {conf}")
    
    if args.model_name in ['mlp', 'gcn']:
        dataset_dir = 'datasets/dgl_graphs/egnn'
    elif args.model_name in ['dimenet', 'egnn']:
        dataset_dir = 'datasets/dgl_graphs/dimenet'
    else:
        dataset_dir = 'datasets/dgl_graphs/5'

    num_train_graphs = 6
    num_test_graphs = 2
    num_val_graphs = 1

    train_ids = range(7)
    train_path = str(Path.cwd().joinpath(dataset_dir, f'train_graphs'))

    test_ids = range(2)
    test_path = str(Path.cwd().joinpath(dataset_dir, f'test_graphs'))

    train_graphs_full, train_labels_dict = dgl.load_graphs(train_path)
    train_labels_full = [train_labels_dict[str(i)] for i in train_ids]
    
    val_graphs = [train_graphs_full[-1]]
    val_labels = [train_labels_full[-1]]
    
    train_labels = train_labels_full[:-1]
    train_graphs = train_graphs_full[:-1]
    
    # test_graphs = [train_graphs_full[0]]
    # test_labels = [train_labels_full[0]]

    test_graphs, test_labels_dict = dgl.load_graphs(test_path)
    test_labels = [test_labels_dict[str(i)] for i in test_ids]
    test_graphs = test_graphs[:num_test_graphs]

    logger.info(f"train_graphs: {train_graphs}")
    logger.info(f"test_graphs: {test_graphs}")
    logger.info(f"val_graphs: {val_graphs}")

    train_graphs = dgl.batch(train_graphs)
    train_labels = torch.cat(train_labels)
    test_graphs = dgl.batch(test_graphs)
    test_labels = torch.cat(test_labels)
    train_labels = train_labels.float()
    test_labels = test_labels.float()
    val_graphs = dgl.batch(val_graphs)
    val_labels = torch.cat(val_labels)
    
    
    train_data = [(train_graphs, train_labels)]
    test_data = [(test_graphs, test_labels)]
    val_data = [(val_graphs, val_labels)]

    
    """ Model init """
    set_seed(0)
    # print(conf)
    model = Model(conf)
    model = model.to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=conf["learning_rate"], weight_decay=conf["weight_decay"], amsgrad=True
    )
    # logger.info(f"optimizer: {optimizer}")
    criterion = torch.nn.MSELoss()
    evaluator = correlation
    logger.info(f"model: {model}")

    # set_seed(0)
    score_test = run(
        conf,
        model,
        train_data,
        test_data,
        val_data,
        # [],
        criterion,
        evaluator,
        optimizer,
        logger,
    )
    
## python3 train.py --model_name='schnet' --output_dir='./outputs/save_schnet' --device=1 --console_log --log_level=10 --learning_rate=0.0001 --patience=100 --max_epoch=20000 --eval_interval=10
