# Heldout_train.py
import sys, os
current_path = os.getcwd()
sys.path.append(os.path.join(current_path, "train"))
"""
Main training script for the GeoBridge model of heldout test.

This script accepts a command-line argument to specify which model to train.
Example usage:
    python train/Heldout_train.py --model EMT --heldout 12
"""

import torch
import argparse
from Heldout_config import TRAINING_CONFIGS, DEVICE
from Heldout_plot import inter_plot
import utils
import model as model_builder
import data_loader
from trainer import INNTrainer
from pathlib import Path

def run_training(model_name: str, heldout: int):
    """
    Executes the training pipeline for a given model name.

    Args:
        model_name (str): The key for the model configuration in TRAINING_CONFIGS.
    """

    # 1. 从库中获取选定模型的配置
    if model_name not in TRAINING_CONFIGS:
        raise ValueError(f"Model '{model_name}' not found in TRAINING_CONFIGS.")
    config = TRAINING_CONFIGS[model_name]
    config['device'] = DEVICE  # 将全局设备设置添加到特定配置中

    print(f"--- Starting training for model: {model_name} ---")

    # 2. 设置随机种子
    utils.set_seed(config['seed'])

    # 3. 加载数据
    All_data, All_t, n_dim = data_loader.load_anndata_to_tensors(
        path=config['data_path'],
        time_column=config['time_column'],
        is_sparse=config['is_sparse'],
        device=config['device']
    )

    train_data, train_t, test_data, test_t, n_dim = data_loader.load_heldout_anndata_to_tensors(
        path=config['data_path'],
        time_column=config['time_column'],  # 对应 adata.obs.day
        heldout_value=heldout,
        is_sparse=config['is_sparse'],
        use_pca=config['use_pca'],
        device=config['device']
    )
    # 4. 构建模型
    inn_model = model_builder.build_inn_model(
        n_dim=n_dim,
        num_blocks=config['num_blocks'],
        hidden_dim=config['hidden_dim']
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
    save_path = Path(config['save_path'] / f"{model_name}_heldout_{config['time_column']}{heldout}_model.pth")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.best_model_state.state_dict(), save_path)
    print(f"Best model for '{model_name}' saved to {save_path}")

    # 7. 画heldout生成图
    output_dir = os.path.join(current_path, "results", f"{model_name}_heldout")
    os.makedirs(output_dir)
    inter_plot(All_data, All_t, trainer.best_model_state, heldout=heldout, plot='org', reg=3e-2, use_kde=config['use_kde'], use_pca=config['use_pca'], output_dir=output_dir)
    inter_plot(All_data, All_t, trainer.best_model_state, heldout=heldout, plot='eu', reg=3e-2, use_kde=config['use_kde'], use_pca=config['use_pca'], output_dir=output_dir)

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
    parser.add_argument(
        '--heldout',
        type=int,
        default=None,
        help='The heldout time label.'
    )
    args = parser.parse_args()

    # 将解析出的模型名称传递给训练函数
    run_training(
        model_name=args.model,
        heldout=args.heldout
    )
