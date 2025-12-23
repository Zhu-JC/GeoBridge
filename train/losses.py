# losses.py
import torch
import random
from mmd import compute_linear_mmd
from Wasserstein_loss import get_OT_plan


def compute_pairwise_distance(tensor1, tensor2):
    return torch.cdist(tensor1, tensor2, p=2)  # 欧式距离

def iso_loss_per_time_point(input, z, t, time_point):
    # 按照时间点过滤样本
    mask = (t == time_point)
    input_subset = input[mask]
    z_subset = z[mask]

    # 计算 input 和 z 之间的两两距离
    input_distances = compute_pairwise_distance(input_subset, input_subset)
    z_distances = compute_pairwise_distance(z_subset, z_subset)

    # 设置对角线为一个非常大的值，以排除自己与自己的距离
    n = input_distances.shape[0]
    inf_mask = torch.eye(n, device=input.device).bool()  # 对角线位置
    # 创建一个对角线为无穷大的矩阵
    inf_matrix = torch.eye(input_distances.size(0), device=input_distances.device)
    inf_matrix[inf_mask] = float('inf')
    input_distances = input_distances + inf_matrix
    z_distances = z_distances + inf_matrix

    # 找出 input 中最小距离的两个点，排除自己与自己的距离
    min_input_distance, min_input_indices = torch.min(input_distances, dim=1)

    # 使用 min_input_indices 找到对应的 z 中的最小距离
    min_z_distance = z_distances[torch.arange(n, device=input.device), min_input_indices]

    # 返回 input 最小距离和 z 最小距离之间的差
    prop = min_input_distance / min_z_distance
    return prop.var(dim=0)/prop.mean()

def compute_iso_loss(input_data: torch.Tensor, z: torch.Tensor, t: torch.Tensor):
    """计算保持等距性的损失。"""
    # 获取所有独特的时间标签
    unique_times = torch.unique(t)

    # 计算每个时间点的损失
    total_loss = 0
    for time_point in unique_times:
        total_loss += iso_loss_per_time_point(input_data, z, t, time_point)

    # 计算所有时间点损失的平均值
    return total_loss / len(unique_times)


def compute_velocity_consistency_loss(t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """计算潜在空间中速度向量的一致性损失（验证损失）。"""
    # 1. 找到唯一的时间点及其对应的索引
    t_unique, inverse_indices = torch.unique(t, sorted=True, return_inverse=True)
    num_time_points = len(t_unique)

    # 2. 计算每个唯一时间点对应的 z 的均值
    z_unique = torch.zeros((num_time_points, z.size(1)), device=z.device)
    for j in range(num_time_points):
        indices_for_t_j = (inverse_indices == j)
        z_unique[j] = z[indices_for_t_j].mean(dim=0)

    mean_diff_values = []   # 存储均值变化率向量

    # 3. & 4. 计算不同唯一时间点对之间的 MMD 和均值差异
    for i in range(num_time_points):
        # 获取时间点 i 的样本均值
        zmean_i = z_unique[i]
        for j in range(i + 1, num_time_points): # 仅计算 j > i 的情况，避免重复和自比较

            # 获取时间点 j 的样本均值
            zmean_j = z_unique[j]

            # 计算时间差
            time_diff = t_unique[j] - t_unique[i]

            # 均值差异计算 (均值的变化率)
            mean_diff = (zmean_j - zmean_i) / (time_diff)
            mean_diff_values.append(mean_diff)

    # 5. 计算最终损失
    # 均值损失 (loss_mean) 计算
    mean_diff_values = torch.stack(mean_diff_values) # 形状: (num_pairs, D)，num_pairs 是有效时间点对的数量
    # 计算这些“均值变化率”向量之间的两两平方欧氏距离
    # cdist 计算的是欧氏距离 (p=2)，需要再平方
    mean_matrix = torch.cdist(mean_diff_values, mean_diff_values, p=2).pow(2) # 形状: (num_pairs, num_pairs)
    # 损失是所有这些成对距离的均值
    loss_mean = torch.mean(mean_matrix)
    return loss_mean


def compute_cvl_loss(
        t: torch.Tensor,
        z: torch.Tensor,
        input_data: torch.Tensor,
        inn_model: torch.nn.Module,
        z_min: torch.Tensor,
        z_max: torch.Tensor,
        ot_reg: float = 1e-2,
        num_gap: int = 2
) -> torch.Tensor:
    """计算基于最优传输插值的 MMD 损失。"""
    t_unique, inverse_indices = torch.unique(t, sorted=True, return_inverse=True)
    num_unique_times = len(t_unique)
    indices_range = list(range(num_unique_times))
    # 确保起始和结束时间点之间至少有两个其他时间点
    valid_pairs = [(i, j) for i in indices_range for j in indices_range if abs(j - i) > num_gap]

    start_time_idx, end_time_idx = random.choice(valid_pairs)

    data_s = z[inverse_indices == start_time_idx]
    data_t = z[inverse_indices == end_time_idx]

    P = get_OT_plan(data_s, data_t, reg=ot_reg, numItermax=1000)

    P_sum_rows = P.sum(axis=1, keepdims=True)
    P_normalized = P / (P_sum_rows)
    data_trans = torch.matmul(P_normalized.to(data_t.dtype), data_t)

    t_max = t_unique[end_time_idx].to(torch.float)
    t_min = t_unique[start_time_idx].to(torch.float)
    time_span = t_max - t_min

    mask = torch.ones(num_unique_times, dtype=torch.bool, device=t_unique.device)
    mask[start_time_idx] = False
    t_values = t_unique[mask] # 其他时间点的值

    org_mmd_values = []
    # 6. 遍历每个中间时间点进行计算
    for i, t_use in enumerate(t_values):
        # 获取对应的原始输入数据
        true_data_indices = (t == t_use)
        data_true = input_data[true_data_indices]

        # 计算插值比例
        inter = (t_use - t_min) / time_span # 确保 t_use 也在 float 类型

        # 线性插值 (使用 data_s 和传输后的 data_trans)
        data_inter_scaled = (1 - inter) * data_s + inter * data_trans

        # 反归一化/反缩放
        scale = z_max - z_min
        data_inter = data_inter_scaled * scale + z_min

        # 通过 INN 反向传播得到对应的 "原始空间" 数据
        data_rev = inn_model(data_inter, rev=True)[0] # 假设 inn 在 z.device

        # 计算插值反演数据与真实数据之间的 MMD
        org_mmd = compute_linear_mmd(data_rev, data_true.to(data_rev.dtype)) # 确保类型匹配
        org_mmd_values.append(org_mmd)

    # 7. 计算最终损失
    loss_org_mmd_values = torch.stack(org_mmd_values)
    loss_org_mmd = loss_org_mmd_values.mean()

    return loss_org_mmd