# data_loader.py
import anndata
import numpy as np
import torch
from typing import Tuple
from scipy.sparse import spmatrix
from sklearn.decomposition import PCA

def load_anndata_to_tensors(
        path: str,
        time_column: str,
        is_sparse: bool,
        device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    从 h5ad 文件加载数据，并根据配置处理稀疏性。
    """
    print(f"Loading data from: {path}")
    adata = anndata.read_h5ad(path)

    # 根据 is_sparse 配置处理数据
    if is_sparse and isinstance(adata.X, spmatrix):
        print("Sparse data detected. Converting to dense array.")
        data_np = adata.X.toarray()
    else:
        data_np = adata.X

    time_labels_np = np.array(adata.obs[time_column])

    data = torch.from_numpy(data_np).to(device)
    time_labels = torch.from_numpy(time_labels_np).to(device)
    n_dim = data.shape[1]

    print(f"Data loaded. Shape: {data.shape}, Time labels from column '{time_column}'.")
    return data, time_labels, n_dim


def load_heldout_anndata_to_tensors(
        path: str,
        time_column: str,
        heldout_value: int,  # 例如 0, 'day0' 等，对应 raw code 中的 day
        is_sparse: bool,
        use_pca: bool,
        device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    加载 h5ad 数据，并根据指定的时间点 (holdout_value) 将数据划分为训练集和测试集。

    返回:
        train_data: 训练集数据 Tensor (All_t != holdout_value)
        train_time: 训练集时间标签 Tensor
        test_data:  测试集数据 Tensor (All_t == holdout_value)
        test_time:  测试集时间标签 Tensor
        n_dim:      特征维度
    """

    # 1. 读取数据
    adata = anndata.read_h5ad(path)

    # 2. 处理稀疏矩阵
    if is_sparse and isinstance(adata.X, spmatrix):
        print("Sparse data detected. Converting to dense array.")
        data_np = adata.X.toarray()
    elif use_pca:
        pca = PCA(n_components=50)
        data_np = pca.fit_transform(adata.X)
    else:
        data_np = adata.X

    # 3. 获取时间标签
    # 对应原始代码: All_t = np.array(adata.obs.day)
    all_time_np = np.array(adata.obs[time_column])

    # 4. 划分索引
    # 对应原始代码: train_indices = np.where(All_t != day)[0]
    train_indices = np.where(all_time_np != heldout_value)[0]
    test_indices = np.where(all_time_np == heldout_value)[0]

    if len(test_indices) == 0:
        print(f"Warning: No samples found for {time_column}={heldout_value}!")

    # 5. 切分数据并转换为 Tensor
    # 对应原始代码中切分 train_data/test_data 的逻辑
    train_data_np = data_np[train_indices]
    train_time_np = all_time_np[train_indices]

    test_data_np = data_np[test_indices]
    test_time_np = all_time_np[test_indices]

    # 转为 Tensor 并移动到设备
    train_data = torch.from_numpy(train_data_np).float().to(device)
    train_time = torch.from_numpy(train_time_np).float().to(device)

    test_data = torch.from_numpy(test_data_np).float().to(device)
    test_time = torch.from_numpy(test_time_np).float().to(device)

    n_dim = train_data.shape[1]

    return train_data, train_time, test_data, test_time, n_dim


def stratified_sampling(train_data, train_t, batch_size):
    """
    从每个时间点 stratified 采样数据，每个时间点取一样数量的数据，生成每个训练 epoch 的子数据集。

    参数:
    - train_data: np.ndarray, 训练数据，形状为 (num_samples, num_features)。
    - train_t: np.ndarray, 训练数据对应的时间点标签，形状为 (num_samples,)。
    - batch_size: int, 每个 batch 的总样本数量。

    返回:
    - sampled_data: np.ndarray, stratified 采样后的数据，形状为 (batch_size, num_features)。
    - sampled_t: np.ndarray, stratified 采样后的时间点标签，形状为 (batch_size,)。
    """
    unique_days = np.unique(train_t)
    num_days = len(unique_days)
    samples_per_day = batch_size // num_days

    if samples_per_day == 0:
        raise ValueError("Batch size is too small to sample from each time point.")

    sampled_data = []
    sampled_t = []
    for day in unique_days:
        day_indices = np.where(train_t == day)[0]
        sampled_indices = np.random.choice(day_indices, size=samples_per_day, replace=False)

        sampled_data.append(train_data[sampled_indices, :])
        sampled_t.append(train_t[sampled_indices])

    # 合并所有时间点的样本
    sampled_data = np.vstack(sampled_data)
    sampled_t = np.hstack(sampled_t)

    return torch.Tensor(sampled_data), torch.from_numpy(sampled_t)
