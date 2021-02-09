#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: Anti-perturbation of Online Social Networks via Bayesian Label Transition
@topic: Simulate non-malicious perturbation (random connections)
@authors: Anonymous Authors
"""

import sys
import numpy as np
import dgl
import torch
import torch.nn.functional as F
import scipy.sparse as ss
from models_GCN import GCN
from load_data import LoadDataset
from utils import dump_pickle, split_masks, MinMaxScaler, concate_adj, concate_feat
from pretrain import evaluation, prediction


def select_target_nodes(label, test_mask, sample_rate=0.1, atk_class=-1):
    """
    @topic: Select target nodes for targeted/non-targeted perturbations.
    @input:
        label (int tensor): ground-truth label;
        test_mask (bool tensor): the mask for testing set;
        sample_rate (float): the ratio of sampling in the testing set;
        atk_class (int): the attacked target class.
    @return:
        target_nodes_list (array): the list of target nodes;
        target_mask (bool tensor): the mask for target nodes.
    """
    target_mask = torch.zeros([len(label)], dtype=torch.bool)
    test_id_list = [i for i in range(len(test_mask)) if test_mask[i] == True]
    target_size = int(len(label[test_mask])*sample_rate) # Decide the size of target nodes
    if int(atk_class) in torch.unique(label): # Select "atk_class" nodes from test graph
        target_idx = [l for l in range(len(label)) if label[l] == atk_class and l in test_id_list]
        target_nodes_list = [i for i in target_idx[:target_size]]
    else: # Random select "target_size" nodes if "atk_class" doesn't belong to any existing classes
        np.random.seed(abs(atk_class)) # Fix the random seed for reproduction.
        target_nodes_list = np.random.choice(test_id_list, target_size, replace=False)
    target_mask[target_nodes_list] = True # Generate the target mask
    return target_nodes_list, target_mask


def perturb_adj(rows, cols, target_nodes_list, num_connect, seed=0):
    """Generate perturbator adjacency matrix"""
    np.random.seed(seed)
    col_idx = np.random.choice(target_nodes_list, \
                                   (rows, num_connect), replace=False)
    pert_adj = np.zeros((rows, cols), dtype=np.int8)
    for r in range(rows):
        pert_adj[r, col_idx[r]] = 1
    return ss.csr_matrix(pert_adj) # returns as a scipy sparse matrix


def perturb_feat(rows, cols, feat_val=1.):
    """Generate perturbator feature matrix"""
    pert_feat = np.zeros((rows, cols), dtype=float)
    pert_feat[:] = feat_val # Assign feat values
    return pert_feat


def format_check(pert_adj, len_graph, num_connect):
    """Check the format of perturbator adj"""
    # input: attack_adj: attacker adj matrix (scipy sparse matrix);
    # for each new node k_i, the number of its link should be smaller or equal to the threshold.
    if pert_adj.getnnz(axis=1).max() <= num_connect: # Should be True
        check1 = True
    else:
        check1 = False
    # the edges pattern between new nodes must be symmetric
    if (pert_adj[:, len_graph:].T != pert_adj[:, len_graph:]).sum() == 0:
        check2 = True
    else:
        check2 = False
    print("The result of 2 checks: ", check1, check2)
    if check1 and check2:
        return "Check Passed!"
    else:
        return "Check Failed!"


if __name__ == "__main__":    
    # ---Initialize the arugments---
    try:
        data_name = str(sys.argv[1])
        model_name = str(sys.argv[2])
        NOISY_RATIO = float(sys.argv[3])
        TARGET_CLASS = int(sys.argv[4])
        SAMPLE_RATE = float(sys.argv[5])
        gpu = int(sys.argv[6])
    except:
        data_name = "cora" # data_name: kdd20_s1, kdd20_s2, cora, citeseer, amazoncobuy, coauthor, reddit
        model_name = "GCN"  # model_name: GCN, SGC, GraphSAGE, TAGCN, GIN
        NOISY_RATIO = 0.1 # the same as that in pre-training
        TARGET_CLASS = -1 # random seed = abs(TARGET_CLASS)
        SAMPLE_RATE = 1.0 # The sampling rate of the target nodes
        gpu = -1
    # The paramaters for perturbations
    PERT_RATE = 0.01 # num_perturbators cannot exceed this ratio    
    #OUT_DIR = '../data/attacker_data/'
    CUT_RATE = 0.3
    # The paramaters for trained model
    lr = 0.001
    n_layers = 2
    n_hidden = 200
    dropout = 0
    weight_decay = 0
    if model_name == "GraphSAGE":
        aggregator_type = "mean"
    elif model_name == "GIN":
        aggregator_type = "sum"
    else:
        aggregator_type = None

    # ---Preprocessing---
    # Load dataset
    data = LoadDataset(data_name)
    graph, feat, label = data.load_data()
    # Randomly split the train, validation, test mask by given cut rate
    _, _, test_mask = split_masks(label, cut_rate=CUT_RATE)

    # --- Non-malicious Perturbations --- (Generate perturbator adj & feat matrix)
    num_victims = int(len(label[test_mask])*SAMPLE_RATE) # The number of victim nodes
    num_perturbators = int(num_victims*PERT_RATE) # The number of perturbator nodes
    rows = num_perturbators # The number of rows in perturbator adj
    cols = len(graph)+rows # The number of columns in perturbator adj
    # Select target nodes for perturbations
    target_nodes_list, target_mask = \
        select_target_nodes(label, test_mask, SAMPLE_RATE, atk_class=TARGET_CLASS)
    # Compute the average degree of target nodes
    adj_sp = graph.adjacency_matrix(scipy_fmt="csr")
    node_degrees = adj_sp.sum(axis=0).A1
    avg_degree_target = np.mean(node_degrees[target_mask])
    print("The average degree of target nodes: ", np.mean(node_degrees[target_mask]))
    print("The average degree of test nodes: ", np.mean(node_degrees[test_mask]))
    # Implement Non-malicious Perturbations
    pert_adj = perturb_adj(rows, cols, target_nodes_list, num_connect=int(1/PERT_RATE), seed=0)
    pert_feat = perturb_feat(rows, feat.shape[1], feat_val=avg_degree_target)
    # Check the format of perturbed adj
    print(format_check(pert_adj, len(graph), num_connect=int(1/PERT_RATE)))
    # Reform the perturbator adj & feat matrix
    #pert_feat = MinMaxScaler(pert_feat, feat.numpy().min(), feat.numpy().max())
    adj_new = concate_adj(graph.adjacency_matrix(scipy_fmt="csr"), pert_adj)
    graph_pert = dgl.from_scipy(adj_new) # Update adj to dgl graph
    feat_pert = concate_feat(feat.numpy(), pert_feat)
    feat_pert = torch.FloatTensor(feat_pert) # convert to torch.tensor
    graph_pert.ndata['feat'] = feat_pert # Update the graph

    # ---Initialize the node classifier---
    print("Initialize the node classifier...")
    # Setup the gpu if necessary
    if gpu < 0:
        print("Using CPU!")
        cuda = False
    else:
        print("Using GPU!")
        cuda = True
        torch.cuda.set_device(gpu)
        graph_pert = graph_pert.to('cuda')
        feat_pert = feat_pert.cuda()
        label = label.cuda()
        target_mask = target_mask.cuda()
    # Create the trained model
    model = GCN(g=graph_pert,
                in_feats=feat_pert.shape[1],
                n_hidden=n_hidden,
                n_classes=len(torch.unique(label)),
                n_layers=n_layers,
                activation=F.relu,
                dropout=dropout,
                model_name=model_name,
                aggregator_type = aggregator_type)
    if cuda: # if gpu is available
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Path for saving the parameters
    path = 'runs/{0}_{1}_nr{2}/'.format(data_name, model_name, NOISY_RATIO) + 'model_best.pth.tar'

    # ---Evaluation on the attacked testing nodes---
    print("Evaluation on the target nodes.")
    # Padding the label and test mask with zero for perturbator node
    label_padding = torch.zeros((rows, ), dtype=torch.long)
    target_mask_padding = torch.zeros((rows, ), dtype=bool)
    if cuda: # Setup GPU if necessary
        label_padding = label_padding.cuda()
        target_mask_padding = target_mask_padding.cuda()
    label_new = torch.cat([label, label_padding], dim=0)
    graph_pert.ndata['label'] = label_new
    target_mask_new = torch.cat([target_mask, target_mask_padding], dim=0)
    if cuda: # Setup GPU if necessary
        label_new = label_new.cuda()
        target_mask_new = target_mask_new.cuda()
    evaluation(model, optimizer, path, graph_pert, feat_pert, label_new, target_mask_new)

    # ---Generate predicted label after interference---
    print("Generate the predicted label after perturbation.")
    Y_pred, Y_pred_sm = prediction(model, optimizer, path, graph_pert, feat_pert)
    if cuda:
        Y_pred, Y_pred_sm = Y_pred.cpu(), Y_pred_sm.cpu()
        target_mask = target_mask.cpu()
    Y_pred, Y_pred_sm = Y_pred[:-num_perturbators], Y_pred_sm[:-num_perturbators]
    dump_pickle('../data/noisy_label/Y_preds_attack.pkl', [Y_pred, Y_pred_sm, target_mask])
    print("Y_pred/Y_pred_sm/target_mask.shape: ", Y_pred.shape, Y_pred_sm.shape, target_mask.shape)
