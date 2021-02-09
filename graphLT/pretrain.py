#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: Anti-perturbation of Online Social Networks via Bayesian Label Transition
@topic: GCNs' training, prediction, and evaluation
@authors: Anonymous Authors
"""

import os
import time
from datetime import datetime
import numpy as np
import sys
#import dgl
import torch
import torch.nn.functional as F
from models_GCN import GCN
from load_data import LoadDataset
from utils import  dump_pickle, split_masks, generate_random_noise_label, \
                    load_checkpoint, save_checkpoint, compute_accuracy, count_parameters
from torch.utils.tensorboard import SummaryWriter
#from collections import Counter


# Fitting the model --------------------
def train(model, optimizer, dirs, feat, label, train_mask, val_mask, n_epochs):
    """
    @topic: Fitting the GCNs
    @input: feature matrix, label, train/val masks, and #epochs.
    @return: train and save the model parameters.
    """
    loss_fn = torch.nn.CrossEntropyLoss()  # Define the loss function

    # Load checkpoint
    try:
        model, optimizer, start_epoch, best_acc = \
            load_checkpoint(dirs+'model_best.pth.tar', model, optimizer)
    except:
        print("Model parameter is not found.")
        start_epoch = 1
    if n_epochs <= start_epoch:
        n_epochs += start_epoch

    writer = SummaryWriter(log_dir=dirs, 
            comment="_time%s"%(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), purge_step=start_epoch)

    dur = []
    best_acc = 0
    for epoch in range(start_epoch, n_epochs):
        model.train()

        if epoch >= 3:
            t0 = time.time()

        # forward
        logits = model(feat)
        loss = loss_fn(logits[train_mask], label[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        with torch.no_grad():
            acc_train = compute_accuracy(logits, label, train_mask)
            loss_train = loss_fn(logits[train_mask], label[train_mask])
            acc_val = compute_accuracy(logits, label, val_mask)
            loss_val = loss_fn(logits[val_mask], label[val_mask])

        # Define the file name
        FileName = "Epoch{0}.pth.tar".format(epoch)
        # Delete previous existing parameter file
        if os.path.exists(dirs+"Epoch{0}.pth.tar".format(epoch-1)):
            os.remove(dirs+"Epoch{0}.pth.tar".format(epoch-1))
        if acc_val > best_acc:
            best_acc = acc_val
            is_best = True
            # Save checkpoint
            save_checkpoint(
                    state = {
                            'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'best_acc': best_acc,
                            'optimizer': optimizer.state_dict(),
                            }, \
                    is_best = is_best, \
                    directory = dirs, \
                    filename = FileName
                            )

        # Output the result
        #if epoch % n_epochs//10 == 0:
        print("Epoch {:05d} | Time(s) {:.4f} | Train Loss {:.4f} | Val Loss {:.4f} "
                      "| Train Accuracy {:.4f} | Val Accuracy {:.4f} "\
        .format(epoch, np.mean(dur), loss_train.item(), loss_val.item(), acc_train, acc_val))

        # Update SummaryWriter
        writer.add_scalar('Loss/train', loss_train.item(), epoch)
        writer.add_scalar('Loss/cross', loss_val.item(), epoch)
        writer.add_scalar('Accuracy/train', acc_train, epoch)
        writer.add_scalar('Accuracy/cross', acc_val, epoch)
        writer.flush()

    writer.close()


# Evaluation --------------------
def evaluation(model, optimizer, path, graph, feat, label, test_mask): # for dgl 0.5.x
    """
    @topic: Evaluation on the given model
    @input: graph, feature matrix, label and its mask.
    @return: print out the test acc.
    """
    try:
        if not len(graph) == len(feat) == len(label) == len(test_mask):
            return "The length of adj/feat/label/test_mask is not equal!"
        model.eval()
        model, optimizer, start_epoch, best_acc = load_checkpoint(path, model, optimizer)
        model.g = graph # update the graph
        logits = model(feat)
        acc = compute_accuracy(logits, label, test_mask)
        print("Best Testing Accuracy: {:.2%}".format(acc))
    except:
        return "Model parameter is not found."


# Prediction --------------------
def prediction(model, optimizer, path, graph, feat):
    """
    @topic: Generate predicted label with well-trained GCN model
    @input: graph, feature matrix.
    @return: predicted label (1D Tensor), probabilistic matrix (2D Tensor).
    """
    try:
        if len(graph) != len(feat):
            return "The length of adj/feat is not equal!"
        model.eval()
        model, optimizer, _, _ = load_checkpoint(path, model, optimizer)
        model.g = graph # update the graph (dgl 0.5.x)
        Y_pred_2d = model(feat) # predicted label (2d)
        Y_pred_2d_softmax = torch.nn.functional.softmax(Y_pred_2d, dim=1) # Normalize each row to sum=1
        Y_pred = torch.max(Y_pred_2d_softmax, dim=1)[1] # predicted label (1d)
        return Y_pred, Y_pred_2d_softmax
    except:
        return "Model parameter is not found."


if __name__ == "__main__":
    # Initialize the arugments
    try:
        data_name = str(sys.argv[1])
        model_name = str(sys.argv[2])
        NUM_EPOCHS = int(sys.argv[3])
        GPU = int(sys.argv[4])
        NOISY_RATIO = float(sys.argv[5])
        is_trainable = bool(sys.argv[6])
    except:
        data_name = "cora" # kdd20_s1, kdd20_s2, cora, citeseer, amazoncobuy, coauthor, reddit
        model_name = "GCN" # GCN, SGC, GraphSAGE, TAGCN, GIN
        NUM_EPOCHS = 200
        GPU = -1
        NOISY_RATIO = 0.1
        is_trainable = True
    CUT_RATE = 0.3
    LR = 0.001
    N_LAYERS = 2
    N_HIDDEN = 200
    DROPOUT = 0
    WEIGHT_DECAY = 0
    if model_name == "GraphSAGE":
        aggregator_type = "mean"
    elif model_name == "GIN":
        aggregator_type = "mean"
    else:
        aggregator_type = None

    # Load dataset
    data = LoadDataset(data_name)
    graph, feat, label = data.load_data()
    print("Class ID: ", set(label.numpy()))
    # Randomly split the train, validation, test mask by given cut rate
    train_mask, val_mask, test_mask = split_masks(label, cut_rate=CUT_RATE)
    # Generate noisy label
    Y_noisy = generate_random_noise_label(label, noisy_ratio=NOISY_RATIO, seed=0)
    Y_noisy = torch.LongTensor(Y_noisy)
    dump_pickle('../data/noisy_label/Y_gt_noisy_masks.pkl', \
                [label, Y_noisy, train_mask, val_mask, test_mask])
    # Display the variables
    print("""-------Data statistics-------'
          # Nodes: {0}
          # Edges: {1}
          # Features: {2}
          # Classes: {3}
          # Train samples: {4}
          # Val samples: {5}
          # Test samples: {6}
          """.format(graph.number_of_nodes(), graph.number_of_edges(),\
                     feat.shape[1], len(torch.unique(label)), \
                      train_mask.int().sum().item(), \
                      val_mask.int().sum().item(), \
                      test_mask.int().sum().item()))

    # Setup the gpu if necessary
    if GPU < 0:
        print("Using CPU!")
        cuda = False
    else:
        print("Using GPU!")
        cuda = True
        torch.cuda.set_device(GPU)
        graph = graph.to('cuda')
        feat = feat.cuda()
        label = label.cuda()
        Y_noisy = Y_noisy.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()

    # ---Initialize the node classifier---
    model = GCN(g=graph,
                in_feats=feat.shape[1],
                n_hidden=N_HIDDEN,
                n_classes=len(torch.unique(label)),
                n_layers=N_LAYERS, 
                activation=F.relu,
                dropout=DROPOUT,
                model_name=model_name,
                aggregator_type=aggregator_type)
    print(model)
    print(f'The model has {count_parameters(model):,} trainable parameters.')
    if cuda: # if gpu is available
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # Path for saving the parameters
    dirs = 'runs/{0}_{1}_nr{2}/'.format(data_name, model_name, NOISY_RATIO)
    path = dirs + 'model_best.pth.tar'

    # Training the model
    if is_trainable:
        train(model, optimizer, dirs, feat, Y_noisy, train_mask, val_mask, NUM_EPOCHS)

     # Evaluation
    print("Evaluation before attack!")
    evaluation(model, optimizer, path, graph, feat, label, test_mask)
    # Generate and save predicted labels
    Y_pred, Y_pred_sm = prediction(model, optimizer, path, graph, feat)
    if cuda:
        Y_pred, Y_pred_sm = Y_pred.cpu(), Y_pred_sm.cpu()
    dump_pickle('../data/noisy_label/Y_preds.pkl', [Y_pred, Y_pred_sm])
    print("Y_pred/Y_pred_sm.shape: ", Y_pred.shape, Y_pred_sm.shape)
