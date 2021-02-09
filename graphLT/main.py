#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: Anti-perturbation of Online Social Networks via Bayesian Label Transition
@topic: Employ GraphLT model to infer label
@authors: Anonymous Authors
"""

import sys
import time
import torch
from utils import read_pickle, gen_init_trans_matrix
from perturbation import select_target_nodes
from models_LT import LabelTransition
from sklearn.metrics import accuracy_score


# ---Initialize the arugments---
try:
    is_attacked = bool(sys.argv[1]) # true or ""
    SAMPLE_RATE = float(sys.argv[2])
    GPU = int(sys.argv[3]) # 0 or -1
    is_inferable = bool(sys.argv[4]) # true or ""
    NUM_INFER = int(sys.argv[5])
    WARMUP_STEP = int(sys.argv[6])
except:
    is_attacked = True  # True / False
    SAMPLE_RATE = 1.0 # The sampling rate of the target nodes
    GPU = -1
    is_inferable = True # True / False
    NUM_INFER = 100 # the number of total epochs for inference
    WARMUP_STEP = 20 # the number of epochs for warm-up stage

# ---Load groundtruth label, predicted label, noisy label and initial transition matrix---
label, Y_noisy, train_mask, val_mask, test_mask = \
                      read_pickle('../data/noisy_label/Y_gt_noisy_masks.pkl')
if is_attacked:
    print("Loading perturbed predicted labels... ")
    Y_pred, Y_pred_sm, target_mask = read_pickle('../data/noisy_label/Y_preds_attack.pkl')
else:
    print("Loading clean predicted labels... ")
    Y_pred, Y_pred_sm = read_pickle('../data/noisy_label/Y_preds.pkl')
    _, target_mask = select_target_nodes(label, test_mask, SAMPLE_RATE, atk_class=-1)
# Convert tensor to numpy array
Y_gt, Y_noisy, Y_pred, Y_pred_sm = \
    label.numpy(), Y_noisy.numpy(), Y_pred.numpy(), Y_pred_sm.detach().numpy() 

# ---Select target nodes for inference---
Y_gt_target = Y_gt[target_mask]
Y_noisy_target = Y_noisy[target_mask]
Y_pred_target = Y_pred[target_mask]
Y_pred_sm_target = Y_pred_sm[target_mask]

# ---Initialize the initial warm-up transition matrix---
print("Initialize the warm-up transition matrix...")
Y_pred_sm_train = Y_pred_sm[train_mask] # predicted probability table (num_samples, num_classes)
Y_noisy_train = Y_noisy[train_mask] # noisy label
NUM_CLASSES = len(set(label.numpy()))
print("NUM_CLASSES: ", NUM_CLASSES)
TM_warmup = gen_init_trans_matrix(Y_pred_sm_train, Y_noisy_train, NUM_CLASSES)
print("The shape of warm-up TM is: ", TM_warmup.shape)

# ---Bayesian Label Transition---
if is_inferable:
    if GPU < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(GPU)
    # Infer the label
    print("Infer the label...")
    timer_0 = time.time()
    lt = LabelTransition(ALPHA=1.0)
    Y_infer, C_new = lt.infer_label(Y_pred_target, Y_pred_sm_target, Y_noisy_target, Y_gt_target, \
                    TM_warmup, GPU, NUM_INFER, WARMUP_STEP)
    runtime = time.time() - timer_0
    print("\n Runtime: ", runtime)
    print("Y_inferred: \n {0} \n C_new: \n {1}".format(Y_infer, C_new))
# Evaluation after inference (accuracy)
print("Evaluation after inference:")
Y_infer, C_new = read_pickle('../data/noisy_label/Y_C.pkl') # array
acc_noisy = accuracy_score(Y_gt_target, Y_noisy_target)
acc_pred = accuracy_score(Y_gt_target, Y_pred_target)
acc_infer = accuracy_score(Y_gt_target, Y_infer)
print("Accuracy of Y_noisy/Y_pred/Y_infer: {0}, {1}, {2}.".format(acc_noisy, acc_pred, acc_infer))
