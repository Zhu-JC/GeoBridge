import torch
import ot
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

##适用于梯度下降的核密度估计
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
    # diag_mask = torch.eye(kernel_vals.size(0), device=kernel_vals.device)
    # kernel_vals = kernel_vals * (1 - diag_mask)  # 将对角线置零，而不是原地操作

    # 密度估计
    densities = kernel_vals.sum(dim=1)  # [num_samples,]

    # 归一化因子
    # normalization_constant = (num_samples - 1) * ((2 * torch.pi * bandwidth ** 2) ** (1 / 2))
    normalization_constant = (num_samples) * ((2 * torch.pi * bandwidth ** 2) ** (1 / 2))
    densities = densities / normalization_constant
    densities = densities / densities.sum()
    return densities

def get_Wasserstein(data_s, data_t, reg=1e-2, numItermax = 1000):
    """
    计算Wasserstein距离。

    参数：
    - data_s: 源域张量
    - data_t: 目标域张量

    返回：
    - WD: Wasserstein距离
    """
    #代价矩阵
    C2 = torch.cdist(data_s, data_t, p=2).double()  # p=2 是欧几里得距离
    C2 = 10000 * C2 / C2.sum()
    #C2 = C2 / (C2.max() - C2.min())  # 标准化距离矩阵
    #执行核密度估计
    mu = kde_density_estimate(data_s, method = 'Silverman')
    nu = kde_density_estimate(data_t, method = 'Silverman')

    #计算最优传输方案
    P = ot.sinkhorn(mu, nu, C2, reg=reg, numItermax=numItermax)
    WD = torch.sum(P * C2)
    return WD


def get_OT_plan(data_s, data_t, reg=1e-2, numItermax = 1000):
    """
    计算Wasserstein距离。

    参数：
    - data_s: 源域张量
    - data_t: 目标域张量

    返回：
    - WD: Wasserstein距离
    """
    #代价矩阵
    C2 = torch.cdist(data_s, data_t, p=2).double()  # p=2 是欧几里得距离
    C2 = 10000 * C2 / C2.sum()
    #C2 = C2 / (C2.max() - C2.min())  # 标准化距离矩阵
    #执行核密度估计
    mu = kde_density_estimate(data_s, method = 'Silverman')
    nu = kde_density_estimate(data_t, method = 'Silverman')
    #计算最优传输方案
    P = ot.sinkhorn(mu, nu, C2, reg=reg, numItermax=numItermax)
    return P