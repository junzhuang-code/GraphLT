#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: Anti-perturbation of Online Social Networks via Bayesian Label Transition
@topic: Model for Label Transition
@authors: Anonymous Authors
"""

import os
import numpy as np
import torch
from utils import dump_pickle, tensor2array
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score


# LabelTransition model --------------------
class LabelTransition():
    def __init__(self, ALPHA=1.0):
        self.ALPHA = ALPHA


    def get_labels_dict(self, Y_pred, Y_noisy):
        """
        @topic: Convert labels as dict
        @input: Y_pred/Y_noisy(1D-array).
        @return: infer_dict/noisy_dict(dict).
        """
        infer_dict = dict() # keys: idx; values: Y_pred.
        noisy_dict = dict() # keys: idx; values: Y_noisy. 
        idx = np.array([i for i in range(len(Y_noisy))])
        for i in range(len(idx)):
            infer_dict[idx[i]] = Y_pred[i]
            noisy_dict[idx[i]] = Y_noisy[i]
        return infer_dict, noisy_dict


    def generate_counting_matrix(self, Y_pred, Y_noisy, NUM_CLASSES):
        """
        @topic: Generate counting matrix and testing labels
        @input: Y_pred/Y_noisy (1D-array); NUM_CLASSES (int).
        @return: C (2D-array).
        """
        C_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
        for i in range(len(Y_noisy)):
            r = Y_pred[i]
            c = Y_noisy[i]
            C_matrix[r][c] += 1
        return C_matrix # (NUM_CLASSES, NUM_CLASSES)


    def approx_Gibbs_sampling(self, Y_pred_sm, Y_noisy, TM):
        """
        @topic: Approximate Gibbs Sampling
        @input:
            Y_pred_sm (2D Tensor: NUM_SAMPLES x NUM_CLASSES);
            Y_noisy (1D Tensor: NUM_SAMPLES x 1);
            TM (2D Tensor: NUM_CLASSES x NUM_CLASSES).
        @return: Y_infer (1D Tensor: NUM_SAMPLES x 1).
        """
        unnorm_probs = Y_pred_sm * torch.index_select(torch.transpose(TM,0,1), 0, Y_noisy)
        probs = unnorm_probs / torch.sum(unnorm_probs, axis=1, keepdims=True)
        Y_infer = torch.max(probs, dim=1)[1]
        return Y_infer


    def infer_label(self, Y_pred, Y_pred_sm, Y_noisy, Y_gt, TM_warmup, \
                    GPU, NUM_EPOCHS=100, WARMUP_STEP=20):
        """
        @topic: Infer the labels with noisy labels
        @input:
            Y_pred/Y_noisy/Y_gt: predicted/noisy/groundtruth labels (1D array: NUM_SAMPLES x 1);
            Y_pred_sm: categorical distribution (2D array: NUM_SAMPLES x NUM_CLASSES);
            TM_warmup: warming-up transition matrix (2D array: NUM_CLASSES x NUM_CLASSES);
            GPU: the ID of GPU device;
            NUM_EPOCHS: the number of training epochs (int);
            WARMUP_STEP: using TM_warmup if step < WARMUP_STEP (int);
            is_fakenodes: whether detect fake nodes (boolean).
        @return:
            Y_infer: new inferred label (1D array: NUM_SAMPLES x 1);
            C: counting matrix (2D array: NUM_CLASSES x NUM_CLASSES).
        """
        # Get index of label
        idx = np.array([i for i in range(len(Y_noisy))])
        # Get Y_pred/Y_noisy dict
        z_dict, y_dict = self.get_labels_dict(Y_pred, Y_noisy)
        # Generate counting matrix
        C = self.generate_counting_matrix(Y_pred, Y_noisy, int(Y_pred_sm.shape[1]))
        # Convert to pytorch tensor
        #Y_pred = torch.LongTensor(Y_pred)
        Y_pred_sm = torch.FloatTensor(Y_pred_sm)
        Y_noisy = torch.LongTensor(Y_noisy)
        C = torch.FloatTensor(C)
        #C_0 = C.clone().detach()
        TM_warmup = torch.FloatTensor(TM_warmup)
        # Setup the GPU
        if GPU >= 0:
            Y_pred_sm = Y_pred_sm.cuda()
            Y_noisy = Y_noisy.cuda()
            C = C.cuda()
            TM_warmup = TM_warmup.cuda()    
        # Setup the interval
        if WARMUP_STEP >= 1000:
            interval = int(WARMUP_STEP//100)
        else:
            interval = 10
        # Record the data if necessary
        writer = SummaryWriter(log_dir=os.path.join("runs", 'Logs_NS'))

        for step in range(NUM_EPOCHS):
            # Update transition matrix TM for every n steps
            #if step >= interval and step % interval == 0:
            if step % interval == 0:
                TM_i = (C + self.ALPHA) / torch.sum(C + self.ALPHA, axis=1, keepdims=True)
                TM_i = TM_i.cuda() if GPU >= 0 else TM_i
                print(".", end = ' ')
            # Infer Z by Gibbs sampling based on corresponding TM
            if step < WARMUP_STEP:
                Y_infer = self.approx_Gibbs_sampling(Y_pred_sm, Y_noisy, TM_warmup)
            else:
                Y_infer = self.approx_Gibbs_sampling(Y_pred_sm, Y_noisy, TM_i)
            Y_infer = Y_infer.cuda() if GPU >= 0 else Y_infer
            # Update the counting matrix C
            for num_i, idx_i in enumerate(idx):
                C[z_dict[idx_i]][y_dict[idx_i]] -= 1
                assert C[z_dict[idx_i]][y_dict[idx_i]] >= 0
                z_dict[idx_i] = int(Y_infer[num_i])
                C[z_dict[idx_i]][y_dict[idx_i]] += 1

            # Compute accuracy for every n steps
            # Tensorboard --logdir=./runs/Logs_NS --port 8999
            if step % interval == 0:
                Y_infer_i = np.array([v for v in z_dict.values()])
                acc_i = accuracy_score(Y_gt, Y_infer_i)
            writer.add_scalar('Accuracy_Y_infer', acc_i, step)

        # Get new infer label z
        Y_inferred = np.array([v for v in z_dict.values()])
        # Store the parameters
        C = tensor2array(C, GPU) # array
        dump_pickle('../data/noisy_label/Y_C.pkl', [Y_inferred, C])
        writer.close()
        return Y_inferred, C
