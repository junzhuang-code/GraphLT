#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: Deperturbation of Online Social Networks via Bayesian Label Transition
@topic: Utils modules
@authors: Anonymous Authors
"""

import os
import pickle
import torch
import dgl
import numpy as np
import scipy.sparse as ss


def read_pickle(file_name):
    """Load the dataset"""
    with open (file_name,'rb') as file:
        return pickle.load(file)

def dump_pickle(file_name, data):
    """Export the dataset"""
    with open (file_name,'wb') as file:
        pickle.dump(data, file)

def preprocess_dgl_adj(graph):
    """
    @topic: Normalize adjacency matrix (dgl graph) for GCNs.
    @input: graph (dgl graph); @return: graph_normalized (dgl graph).
    """
    adj_csr = graph.adjacency_matrix(scipy_fmt="csr") # convert dgl graph to csr matrix
    adj_csr = ss.csr_matrix(np.eye(adj_csr.shape[0]) + adj_csr.toarray()) # add self-connection for each node
    rowsum = adj_csr.sum(1).A1 # sum up along the columns
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0. # Eliminate the inf number
    d_mat_inv_sqrt = ss.diags(d_inv_sqrt) # compute the inverse squared degree matrix
    adj_normalized = adj_csr.dot(d_mat_inv_sqrt).T.dot(d_mat_inv_sqrt) # note that both matrix should be sparse matrix.
    graph_normalized = dgl.from_scipy(adj_normalized)
    return graph_normalized

def split_masks(Y, cut_rate=0.2):
    """Split the train/val/test masks"""
    # input: Y: real label; cut_rate: the cur ratio of test mask.

    def create_mask(shape):
        # Create a zero tensor for mask
        return torch.zeros([shape], dtype=torch.bool)

    # Create masks
    tensor_shape = Y.shape[0]
    train_mask, val_mask, test_mask = create_mask(tensor_shape), create_mask(tensor_shape), create_mask(tensor_shape)
    # Generate a random idx
    torch.manual_seed(12)
    idx = list(torch.utils.data.RandomSampler(range(0, Y.shape[0])))
    # Split the mask
    train_mask[idx[ : int(len(idx)*(1-cut_rate*2))]] = True # First (1-cut_rate*2) for train
    val_mask[idx[int(len(idx)*(1-cut_rate*2)) : int(len(idx)*(1-cut_rate))]] = True # Second part for val
    test_mask[idx[int(len(idx)*(1-cut_rate)) : int(len(idx)*1)]] = True # Rest for test
    return train_mask, val_mask, test_mask

def generate_random_noise_label(label, noisy_ratio=0.3, seed=0):
    """
    @topic: Randomly generate noise label with given noisy_ratio.
    @input: lable(1D-array), noise_ratio(float), seed(int).
    @return: noisy label (1D-array).
    """
    np.random.seed(seed)
    label_ = np.random.randint(min(label), high=max(label), size=len(label))
    mask_idx = np.random.choice(len(label), int(noisy_ratio*len(label)), replace=False)
    label = np.array(label)
    label[mask_idx] = label_[mask_idx]
    return label

def gen_init_trans_matrix(Y_pred_sm, Y_noisy, NUM_CLASSES):
    """
    @topic: Generate initial transition matrix
    @input:
        Y_pred_sm (2D-array: NUM_SAMPLES x NUM_CLASSES);
        Y_noisy (1D-array: NUM_SAMPLES x 1);
        NUM_CLASSES (int).
    @return: TM_init (2D-array: NUM_CLASSES x NUM_CLASSES).
    """
    unnorm_TM = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for i in range(len(Y_noisy)):
        label = Y_noisy[i]
        unnorm_TM[:,label] += Y_pred_sm[i]
    unnorm_TM_sum = np.sum(unnorm_TM, axis=1)
    TM_init = unnorm_TM / unnorm_TM_sum[:,None]
    return TM_init

def MinMaxScaler(data, low, high):
    """Rescale 2D matrix into given range"""
    # input: data (2d matrix), low/high (Scalar).
    # output: scaled data (2d matrix).
    data_max, data_min = data.max(axis=0), data.min(axis=0)
    data_std = (data - data_min) / (data_max - data_min + 0.00001)
    data_scaled = data_std * (high - low) + low
    return data_scaled

def compute_accuracy(logits, labels, mask):
    """Compute the accuracy"""
    logits = logits[mask]
    labels = labels[mask]
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

def save_checkpoint(state, is_best, directory, filename='checkpoint.pth.tar'):
    """Saves checkpoint"""
    import shutil
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, directory+'model_best.pth.tar')

def load_checkpoint(checkpoint_fpath, model, optimizer):
    """Loads checkpoint"""
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_acc = checkpoint['best_acc']
    return model, optimizer, checkpoint['epoch'], best_acc

def count_parameters(model):
    """Count the number of trianable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def tensor2array(tensor, gpu):
    """Convert tensor to numpy array (Input must be Tensor)!"""
    return tensor.numpy() if gpu < 0 else tensor.cpu().numpy()

def concate_adj(adj_old, adj_new):
    """Concate the perturbator adj matrix into original adj matrix"""
    # input: adj_old/adj_new are scipy sparse matrix.
    # outtput: scipy sparse matrix.
    if adj_old.shape[1] != adj_new[:,0:adj_old.shape[1]].shape[1]:
        return "Columns are not matched!"
    adj_vstack = ss.vstack([adj_old, adj_new[:,0: adj_old.shape[1]]])
    if adj_vstack.shape[0] != adj_new.T.shape[0]:
        return "Rows are not matched!"
    att_hstack = ss.hstack([adj_vstack, adj_new.T])
    return att_hstack

def concate_feat(feat_old, feat_new):
    """Concate the feature matrix"""
    # input: feat_old/feat_new are numpy array.
    # outtput: numpy array.
    if feat_old.shape[1] != feat_new.shape[1]:
        return "Columns are not matched!"
    return np.vstack((feat_old, feat_new))
