# Deperturbation of Online Social Networks via Bayesian Label Transition

### Authors: Jun Zhuang, Mohammad Al Hasan

### Paper:
Accepted by SDM 2022'

### Abstract:
Online social networks (OSNs) classify users into different categories based on their online activities and interests, a task which is referred as a node classification task. Such a task can be solved effectively using Graph Convolutional Networks (GCNs). However, a small number of users, so-called perturbators, may perform random activities on an OSN, which significantly deteriorate the performance of a GCN-based node classification task. Existing works in this direction defend GCNs either by adversarial training or by identifying the attacker nodes followed by their removal. However, both of these approaches require that the attack patterns or attacker nodes be identified first, which is difficult in the scenario when the number of perturbator nodes is very small. In this work, we develop a GCN defense model, namely GraphLT, which uses the concept of label transition. GraphLT assumes that perturbators' random activities deteriorate GCN's performance. To overcome this issue, GraphLT subsequently uses a novel Bayesian label transition model, which takes GCN's predicted labels and applies label transitions by Gibbs-sampling-based inference and thus repairs GCN's prediction to achieve better node classification. Extensive experiments on seven benchmark datasets show that GraphLT considerably enhances the performance of the node classifier in an unperturbed environment; furthermore, it validates that GraphLT can successfully repair a GCN-based node classifier with superior performance than several competing methods.

### Dataset:
 kdd20_s1, kdd20_s2, cora, citeseer, amazoncobuy, coauthor, reddit \
 KDD20 Datasets can be downloaded from **https://www.biendata.xyz/competition/kddcup_2020/**

### Getting Started:
#### Prerequisites
 Linux or macOS \
 CPU or NVIDIA GPU + CUDA CuDNN \
 Python 3 \
 pytorch, dgl, numpy, scipy, sklearn

#### Clone this repo
**git clone https://github.com/[user_name]/GraphLT.git** \
**cd GraphLT/graphLT**

#### Install dependencies
For pip users, please type the command: **pip install -r requirements.txt** \
For Conda users, you may create a new Conda environment using: **conda env create -f environment.yml**

#### Directories
##### graphLT:
 1. *pretrain.py*: Train the GCN-based node classifier on given dataset
 2. *perturbation.py*: Simulate non-malicious perturbations
 3. *main.py*: Infer label by GraphLT
 4. *models_GCN.py*/*models_LT.py*: Model scripts
 5. *load_data.py*: Load data script
 6. *utils.py*: Utils modules
##### data:
 1. *kdd20_s1/kdd20_s2*: Two directories that store KDD20 datasets
 2. *noisy_label*: A directory that stores labels

#### Runs
 1. Train the GCN-based node classifier on given dataset \
  python pretrain.py -data_name -model_name -NUM_EPOCHS -GPU -NOISY_RATIO -is_trainable \
  e.g. python pretrain.py cora GCN 200 0 0.1 true

 2. Simulate non-malicious perturbations and generate predicted labels \
  python perturbation.py -data_name -model_name -NOISY_RATIO -TARGET_CLASS -SAMPLE_RATE -GPU \
  e.g. python perturbation.py cora GCN 0.1 -1 1.0 0

 3. Infer labels and evaluation \
  python main.py -is_attacked -SAMPLE_RATE -GPU -is_inferable -NUM_INFER -WARMUP_STEP \
  e.g. python main.py true 1.0 0 true 100 20

 4. Visualization with Tensorboard \
  Under the directory of "graphLT", run: Tensorboard --logdir=./runs/Logs_LT --port=8999
