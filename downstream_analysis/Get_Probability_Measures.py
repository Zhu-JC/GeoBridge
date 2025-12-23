import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA as PCA2
import numpy as np
from scipy.stats import gaussian_kde
import torch
from torch_pca import PCA
# 基因表达数据保存在一个DataFrame中，每行代表一个细胞，每列代表一个基因
# 使用PCA降维
def Neighbor_Measures(data, n_components=10, threshold=10, epsilon=1e-10):
    # 初始化PCA，设置主成分数量
    pca = PCA2(n_components=n_components)

    # 拟合并转换数据
    reduced_data = pca.fit_transform(data)

    # Step 2: 计算两两之间的欧氏距离
    # pdist 计算所有两两点之间的距离，然后用 squareform 转换为矩阵形式
    distance_matrix = squareform(pdist(reduced_data, metric='euclidean'))

    # 转换为DataFrame，列名和行名都为细胞名或索引
    distance_df = pd.DataFrame(distance_matrix, index=data.index, columns=data.index)
    dis_threshold = np.percentile(distance_df.values.flatten(), threshold)
    # 生成布尔矩阵：距离小于阈值的地方为True，否则为False
    neighbors_matrix = (distance_df < dis_threshold).astype(int)

    # 自己不算自己的邻居，所以对角线置零
    np.fill_diagonal(neighbors_matrix.values, 0)

    # 每个细胞的邻居数（True的数量）
    neighbors_count = neighbors_matrix.sum(axis=1)

    # 加上一个很小的数以避免测度为零
    neighbors_count += epsilon

    # Step 2: 归一化为概率测度
    total_neighbors = neighbors_count.sum()
    neighbors_probability = neighbors_count / total_neighbors

    return neighbors_probability


def kde_gene_expression(gene_expression_matrix, n_components=2):
    """
    对高维基因表达数据进行PCA降维，并在降维后的空间中进行核密度估计。

    Parameters:
    - gene_expression_matrix: numpy array, 形状为 (n_cells, n_genes)
    - n_components: int, PCA降维的维度，默认为2
    - grid_size: int, 定义核密度估计的网格大小，默认为100

    Returns:
    - 绘制KDE图
    """

    # Step 1: PCA降维
    pca = PCA2(n_components=n_components)
    reduced_data = pca.fit_transform(gene_expression_matrix)

    # Step 2: 创建核密度估计模型
    kde = gaussian_kde(reduced_data.T)

    # Step 3: 计算每个细胞的概率密度
    densities = kde(reduced_data.T)

    # Step 4: 归一化密度值，使得总和为1
    normalized_densities = densities / densities.sum()

    return normalized_densities

##自定义的pca降维方法，不会打断计算图
def pca_reduction(samples, n_components=2):
    # 样本归一化，减去均值
    mean = torch.mean(samples, dim=0, keepdim=True)
    normalized_data = samples - mean

    # 进行 SVD 分解
    U, S, V = torch.svd(normalized_data)

    # 取前 n_components 右奇异向量作为主成分方向
    top_eigenvectors = V[:, :n_components]

    # 投影到前 n_components 主成分方向
    reduced_data = torch.mm(normalized_data, top_eigenvectors)

    return reduced_data



def kde_density_estimate(samples, method = 'Silverman', n_components=2):
    """
    使用高斯核对样本进行核密度估计。

    参数：
    - samples: [num_samples, num_features]，基因表达矩阵
    - method: 估计带宽的方法

    返回：
    - densities: [num_samples,]，每个样本的概率密度估计
    """

    # Step 1: PCA降维
    reduced_data = pca_reduction(samples, n_components)

    num_samples = reduced_data.size(0)
    num_features = reduced_data.size(1)

    std_dev = torch.std(reduced_data, dim=0, unbiased=True)  # [num_features]
    std_dev_avg = torch.mean(std_dev)  # 对所有特征求平均

    if method == 'Silverman':
        bandwidth = ((4 / ((num_features + 2) * num_samples)) ** (1 / (num_features + 4))) * std_dev_avg ##Silverman 法则计算带宽
    else:
        bandwidth = num_samples ** (-1 / (num_features + 4)) * std_dev_avg ##Scott 法则计算带宽

    # 计算样本之间的两两距离平方
    diff = reduced_data.unsqueeze(1) - reduced_data.unsqueeze(0)  # [num_samples, num_samples, num_features]
    distances_sq = torch.sum(diff ** 2, dim=2)  # [num_samples, num_samples]

    # 计算高斯核值
    kernel_vals = torch.exp(-distances_sq / (2 * bandwidth ** 2))  # [num_samples, num_samples]

    # 将自身的核值置零，避免自身对自身的影响
    diag_mask = torch.eye(kernel_vals.size(0), device=kernel_vals.device)
    kernel_vals = kernel_vals * (1 - diag_mask)  # 将对角线置零，而不是原地操作

    # 密度估计
    densities = kernel_vals.sum(dim=1)  # [num_samples,]

    # 归一化因子
    normalization_constant = (num_samples - 1) * ((2 * np.pi * bandwidth ** 2) ** (1 / 2))
    densities = densities / normalization_constant
    densities = densities / densities.sum()
    return densities