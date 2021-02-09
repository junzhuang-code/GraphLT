#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: Anti-perturbation of Online Social Networks via Bayesian Label Transition
@topic: GCN Models
@authors: Anonymous Authors
"""

from dgl.nn.pytorch import GraphConv, SGConv, SAGEConv, TAGConv, GINConv
import torch


# GCN model --------------------
class GCN(torch.nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 model_name,
                **kwargs):
        super(GCN, self).__init__()
        self.g = g # graph DGLGraph
        self.layers = torch.nn.ModuleList()
        self.aggregator_type = kwargs["aggregator_type"]
        # Select the model layer
        if model_name == "GCN":
            model_in = GraphConv(in_feats, n_hidden, activation=activation, allow_zero_in_degree=True)
            model_h = GraphConv(n_hidden, n_hidden, activation=activation, allow_zero_in_degree=True)
            model_out = GraphConv(n_hidden, n_classes, allow_zero_in_degree=True)
        elif model_name == "SGC": # k for the size of filter
            model_in = SGConv(in_feats, n_hidden, k=2, allow_zero_in_degree=True)
            model_h = SGConv(n_hidden, n_hidden, k=2, allow_zero_in_degree=True)
            model_out = SGConv(n_hidden, n_classes, k=2, allow_zero_in_degree=True)
        elif model_name == "GraphSAGE": # Aggregator type: mean, gcn, pool, lstm.
            model_in = SAGEConv(in_feats, n_hidden, self.aggregator_type, activation=activation)
            model_h = SAGEConv(n_hidden, n_hidden, self.aggregator_type, activation=activation)
            model_out = SAGEConv(n_hidden, n_classes, self.aggregator_type)
        elif model_name == "TAGCN": # k for the size of filter
            model_in = TAGConv(in_feats, n_hidden, k=2, activation=activation)
            model_h = TAGConv(n_hidden, n_hidden, k=2, activation=activation)
            model_out = TAGConv(n_hidden, n_classes, k=2)
        elif model_name == "GIN": # Aggregator type: sum, max or mean.
            model_in = GINConv(torch.nn.Linear(in_feats, n_hidden), self.aggregator_type, init_eps=0)
            model_h = GINConv(torch.nn.Linear(n_hidden, n_hidden), self.aggregator_type, init_eps=0)
            model_out = GINConv(torch.nn.Linear(n_hidden, n_classes), self.aggregator_type, init_eps=0)
        # if model_name == "xxx": # Add model layer here if necessary
        else:
            print("model_name is incorrect!")
            return 0
        # Build the model
        self.layers.append(model_in) # input layer
        for _ in range(n_layers - 1):
            self.layers.append(torch.nn.Dropout(p=dropout))
            self.layers.append(model_h) # hidden layers
        self.layers.append(torch.nn.Dropout(p=dropout))
        self.layers.append(model_out) # output layer


    def forward(self, feat):
        h = feat
        for i, layer in enumerate(self.layers):
            if type(layer) == torch.nn.modules.dropout.Dropout:
                h = layer(h)
            else:
                h = layer(self.g, h)
        return h
