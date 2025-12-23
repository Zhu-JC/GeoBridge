# GeoBridge_train.py
import sys, os
current_path = os.getcwd()
sys.path.append(os.path.join(current_path, "train"))
"""
Main training script for the GeoBridge model.

This script accepts a command-line argument to specify which model to train.
Example usage:
    python train/GeoBridge_train.py --model MET
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



def run_training(model_name: str):
    """
    Executes the training pipeline for a given model name.

    Args:
        model_name (str): The key for the model configuration in TRAINING_CONFIGS.
    """

    # 1. 从库中获取选定模型的配置
    if model_name not in TRAINING_CONFIGS:
        raise ValueError(f"Model '{model_name}' not found in TRAINING_CONFIGS.")
    config = TRAINING_CONFIGS[model_name]
    config['device'] = DEVICE

    print(f"--- Starting training for model: {model_name} ---")

    # 2. 设置随机种子
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
    )

    # 5. 初始化训练器并开始训练
    trainer = INNTrainer(
        model=inn_model,
        config=config,
        train_data=train_data,
        train_t=train_t
    )
    trainer.train()

    # 6. 保存最佳模型
    save_path = Path(config['save_path'])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.best_model_state.state_dict(), save_path)
    print(f"Best model for '{model_name}' saved to {save_path}")


if __name__ == '__main__':
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="Train a GeoBridge INN model.")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=list(TRAINING_CONFIGS.keys()),
        help='The name of the model configuration to train.'
    )

    args = parser.parse_args()

    # 将解析出的模型名称传递给训练函数
    run_training(model_name=args.model)
