# Anti-perturbation of Online Social Networks via Bayesian Label Transition

### Authors: Anonymous Authors

### Paper:
Under reviewed by SDM 2022'

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
 1. *pretrain.py*: Pre-train GCN model on given dataset
 2. *perturbation.py*: Simulate non-malicious perturbations
 3. *main.py*: Infer label by GraphLT
 4. *models_GCN.py*/*models_LT.py*: Model scripts
 5. *load_data.py*: Load data script
 6. *utils.py*: Utils modules
##### data:
 1. *kdd20_s1/kdd20_s2*: Two directories that store KDD20 datasets
 2. *noisy_label*: A directory that stores labels

#### Runs
 1. Pre-train GCN model on given dataset \
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
