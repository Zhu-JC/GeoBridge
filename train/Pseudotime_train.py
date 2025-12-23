# Pseudotime_train.py

"""
Main training script for the GeoBridge model of single-snapshot data.

This script accepts a command-line argument to specify which model to train.
Example usage:
    python train/Pseudotime_train.py --model EMT
"""
import sys, os
current_path = os.getcwd()
sys.path.append(os.path.join(current_path, "train"))
import torch
import argparse
from config import TRAINING_CONFIGS, DEVICE
import utils
import model as model_builder
import data_loader
import Pseudotime_methods
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from snapshot_trainer import SnapShotTrainer
import anndata


def run_training(model_name: str, interval: int, cluster: str, num_clusters: int):
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
    output_dir = os.path.join(current_path, "results", f"{model_name}_pseudotime_results")
    os.makedirs(output_dir, exist_ok=True)
    print(f"--- Starting training for model: {model_name} ---")

    # 2. 设置随机种子
    utils.set_seed(config['seed'])

    # 3. 加载数据
    train_data, true_t, n_dim = data_loader.load_anndata_to_tensors(
        path=config['data_path'],
        time_column=config['time_column'],
        is_sparse=config['is_sparse'],
        device=config['device']
    )
    if cluster:
        adata = anndata.read_h5ad(config['data_path'])
        cluster_labels = adata.obs[cluster]
    else:
        cluster_labels = None

    # 4. 构建模型
    inn_model = model_builder.build_inn_model(
        n_dim=n_dim,
        num_blocks=config['num_blocks']
    )

    # 5. 初始化训练器并开始训练
    trainer = SnapShotTrainer(
        model=inn_model,
        config=config,
        train_data=train_data,
        true_t=true_t,
        interval=interval,
        cluster_labels=cluster_labels,
        num_clusters=num_clusters,
        save_dir=output_dir
    )
    trainer.train()

    # 6. 保存最佳模型

    torch.save(trainer.best_model_state.state_dict(), f'{output_dir}/pseudotime_model.pth')
    np.save(f'{output_dir}/pseudo_t.npy', trainer.train_t_np)
    print(f"Best pseudotime model for '{model_name}' saved to {output_dir}")

    # 7.画与真实时间相关性benchmark
    x_rounds = range(len(trainer.cor_list))
    y_correlation_values = trainer.cor_list
    sns.set_theme(style="darkgrid", context="talk", palette="deep")
    fig, ax = plt.subplots(figsize=(8, 5))
    # 主曲线（GeoBridge）
    ax.plot(x_rounds, y_correlation_values, marker='o', linestyle='-',
            markersize=10, markerfacecolor='royalblue', color='royalblue',
            linewidth=2.2, label='GeoBridge')

    Slingshot_cor = Pseudotime_methods.get_slingshot_pseudotime(config['data_path'], true_t, cluster, output_dir)
    PAGA_cor = Pseudotime_methods.get_PAGA_pseudotime(config['data_path'], true_t, cluster, output_dir)
    Palantir_cor = Pseudotime_methods.get_palantir_pseudotime(config['data_path'], true_t, cluster, output_dir)
    # 基线对比方法
    ax.axhline(y=Slingshot_cor, linestyle='--', color='orange', linewidth=2, label='Slingshot')
    ax.axhline(y=PAGA_cor, linestyle='--', color='green', linewidth=2, label='DPT')
    ax.axhline(y=Palantir_cor, linestyle='--', color='red', linewidth=2, label='Palantir')

    # 设置标签字体与网格效果
    ax.set_xlabel("Training Round", fontsize=14, labelpad=10)
    ax.set_ylabel("Pseudo-time / Real Time Correlation", fontsize=14, labelpad=10)

    # 调整 y 轴范围
    min_corr = min(y_correlation_values) if y_correlation_values else 0
    max_corr = max(y_correlation_values) if y_correlation_values else 1
    bm_cor_list = [y_correlation_values, Slingshot_cor, PAGA_cor, Palantir_cor]
    ax.set_ylim((min(min_corr, np.nanmin(bm_cor_list)) - 0.05),
                (max(max_corr, np.nanmax(bm_cor_list)) + 0.05))
    # 添加图例
    ax.legend(frameon=True, facecolor='white', edgecolor='gray', fontsize=12)
    ax.grid(alpha=0.4)
    plt.tight_layout()
    # 保存图像
    plt.savefig(f"{output_dir}/cor_bm.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


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
        '--interval',
        type=int,
        required=True,
        help='Epoch interval for iterative optimization.'
    )
    parser.add_argument(
        '--cluster',
        type=str,
        default=None,
        help='Cluster name of the dataset. If not provided, cluster will get by kmeans.'
    )
    parser.add_argument(
        '--num_clusters',
        type=int,
        default=None,
        help='Number of kmeans clusters.'
    )
    args = parser.parse_args()

    # 将解析出的模型名称传递给训练函数
    run_training(
        model_name=args.model,
        interval=args.interval,
        cluster=args.cluster,
        num_clusters=args.num_clusters
    )
