# GeoBridge_train.py
import sys, os
current_path = os.getcwd()
sys.path.append(os.path.join(current_path, "train"))
"""
Main training script for the GeoBridge model.

This script accepts a command-line argument to specify which model to train.
Example usage:
    python train/GeoBridge_train.py --model MET_EMT_All
    python train/GeoBridge_train.py --model sc_beta
"""

import torch
import argparse
from config import TRAINING_CONFIGS, DEVICE
import utils
import model as model_builder
import data_loader
from trainer import INNTrainer
from pathlib import Path

config = TRAINING_CONFIGS['MET']
config['device'] = DEVICE

utils.set_seed(config['seed'])

# 3. 加载数据
train_data, train_t, n_dim = data_loader.load_anndata_to_tensors(
    path=config['data_path'],
    time_column=config['time_column'],
    is_sparse=config['is_sparse'],
    device=config['device']
)

# 4. 构建模型
inn_model = model_builder.build_inn_model(
    n_dim=n_dim,
    num_blocks=config['num_blocks']
).to(config['device'])



# 5. 初始化训练器并开始训练
trainer = INNTrainer(
    model=inn_model,
    config=config,
    train_data=train_data,
    train_t=train_t
)
trainer.train()
