#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: Anti-perturbation of Online Social Networks via Bayesian Label Transition
@topic: Load dataset
@authors: Anonymous Authors
"""

from utils import read_pickle, preprocess_dgl_adj
import numpy as np
import dgl
import dgl.data
import torch

class LoadDataset():
    def __init__(self, data_name):
        self.data_name = data_name
        print("Current dataset: kdd20_s1, kdd20_s2, cora, citeseer, amazoncobuy, coauthor, reddit.")
        print("Selecting {0} Dataset ...".format(self.data_name))


    def load_data(self):
        # Load dataset based on given data_name.
        if self.data_name == 'kdd20_s1':
            return self.load_kdd20(1)
        if self.data_name == 'kdd20_s2':
            return self.load_kdd20(2)
        if self.data_name == "cora":  # cora_v2
            dataset = dgl.data.CoraGraphDataset()
        if self.data_name == "citeseer":  # citeseer
            dataset = dgl.data.CiteseerGraphDataset()
        if self.data_name == "pubmed":  # pubmed
            dataset = dgl.data.PubmedGraphDataset()
        if self.data_name == "amazoncobuy":  # amazon_co_buy_photo
            dataset = dgl.data.AmazonCoBuyComputerDataset()  # AmazonCoBuyPhotoDataset
        if self.data_name == "coauthor":  # coauthor_cs
            dataset = dgl.data.CoauthorCSDataset()
        if self.data_name == "reddit":  # reddit
            dataset = dgl.data.RedditDataset()

        # Load graph, feature matrix, and label
        graph = dataset[0]
        feat = graph.ndata['feat']  # float32
        label = graph.ndata['label']  # int64

        # Preprocessing the adjacency matrix (dgl graph) and update the graph
        if self.data_name == "amazoncobuy" or self.data_name == "coauthor":
            graph = preprocess_dgl_adj(graph)
            graph.ndata['feat'] = feat
            graph.ndata['label'] = label

        print("Data is stored in: /Users/[user_name]/.dgl")
        print("{0} Dataset Loaded!".format(self.data_name))
        return graph, feat, label


    def load_kdd20(self, stage=1):
        # Load KDD Cup 2020 Dataset
        # Read the pickle file
        if stage == 1:
            A = read_pickle('../data/kdd20_s1/experimental_adj.pkl') # (scipy.sparse.csr_matrix)
            X = read_pickle('../data/kdd20_s1/experimental_features.pkl') # (array)
            Y = read_pickle('../data/kdd20_s1/experimental_train.pkl') # (array)
        elif stage == 2:
            A = read_pickle('../data/kdd20_s2/adj_matrix_formal_stage.pkl') # (scipy.sparse.csr_matrix)
            X = np.load('../data/kdd20_s2/feature_formal_stage.npy') # (array)
            Y = np.load('../data/kdd20_s2/train_labels_formal_stage.npy') # (array)
        else:
            return "Please select stage 1 or 2."
        if Y.max() + 1 != len(set(Y)):
            # Execute Label Transform
            from sklearn import preprocessing
            le = preprocessing.LabelEncoder()
            Y = le.fit_transform(Y)
        print("The shape of A, X, Y: ", A.shape, X.shape, Y.shape)
        # Remove the nodes which has no labels. # new
        A = A[:Y.shape[0], :Y.shape[0]]
        X = X[:Y.shape[0]]
        # Convert to the desirable format
        graph = dgl.from_scipy(A)  # dgl graph
        feat = torch.FloatTensor(X)  # feature matrix
        label = torch.LongTensor(Y)  # label
        graph.ndata['feat'] = feat  # update the graph
        graph.ndata['label'] = label
        # label_fake = torch.zeros(A.shape[0] - Y.shape[0], dtype = int)
        # label = torch.cat([label_real, label_fake])
        print("Data is stored in: ../data/kdd20_s{0}/".format(stage))
        print("KDD Cup 2020 Dataset Loaded!")
        return graph, feat, label
