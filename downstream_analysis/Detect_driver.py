import torch
import torch.nn as nn
import torch.nn.init as init
import matplotlib.pyplot as plt
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
import anndata
from scipy.spatial.distance import cdist
import pandas as pd
import ot
import Get_Probability_Measures
import random
from tqdm import tqdm
from torch import autograd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def scaled_output(z):
    """
    对 z 进行普通归一化处理。

    参数:
    z (torch.Tensor): 输入张量。

    返回:
    torch.Tensor: 归一化后的张量。
    """
    z_min = z.min(dim=0, keepdim=True)[0]
    z_max = z.max(dim=0, keepdim=True)[0]

    # 避免分母为零的情况
    z_norm = (z - z_min) / (z_max - z_min + 1e-8)

    return z_norm

# 行归一化函数
def row_normalize(matrix):
    """将矩阵内部每一行归一化为均值=0，标准差=1"""
    return (matrix - np.mean(matrix, axis=1, keepdims=True)) / np.std(matrix, axis=1, keepdims=True)

# z, _ = inn_MET(data_MET)
# z = scaled_output(z)
# (output,) = autograd.grad(
#     z[:, 0].unsqueeze(1),
#     data_MET,
#     create_graph=True,
#     only_inputs=True,
#     grad_outputs=torch.ones((data_MET.size()[0], 1), device=data_MET.device).float(),
# )

# data_use=data_MET; t_use=t_MET; inn=inn_MET; source=4; target=5; num_inter=100; method = 'kde'; reg = 2e-2

def Detect_driver(data_use, t_use, inn, cell_fate, source=4, target=5, method='kde', reg=2e-2, name=''):
    N = data_use.shape[1]
    z, _ = inn(data_use)
    z_min = z.min(dim=0, keepdim=True)[0]
    z_max = z.max(dim=0, keepdim=True)[0]
    z = scaled_output(z)
    t_use = torch.from_numpy(t_use).to(device)
    l_unique, inverse_indices = torch.unique(t_use, return_inverse=True)
    t_target = l_unique.max().to(torch.float32)
    t_source = l_unique.min().to(torch.float32)

    data_t = z[target]
    data_t = pd.DataFrame(data_t.cpu().detach().numpy())
    data_s = z[source]
    data_s = pd.DataFrame(data_s.cpu().detach().numpy())
    C = cdist(data_s.values, data_t.values, metric='euclidean')
    if method == 'neighbor':
        mu = Get_Probability_Measures.Neighbor_Measures(data_s, 10, epsilon=1e-5)
        mu = mu.to_numpy()
        nu = Get_Probability_Measures.Neighbor_Measures(data_t, 10, epsilon=1e-5)
        nu = nu.to_numpy()
    else:
        mu = Get_Probability_Measures.kde_gene_expression(data_s)
        nu = Get_Probability_Measures.kde_gene_expression(data_t)
    P = ot.sinkhorn(mu, nu, C, reg=reg)
    P_normalized = P / P.sum(axis=1, keepdims=True)
    data_trans = np.dot(P_normalized, data_t.values)
    data_trans = torch.from_numpy(data_trans).to(device).to(torch.float32)
    data_s = torch.from_numpy(data_s.values).to(device)
    data_trans = data_trans[cell_fate == name, :]
    data_s = data_s[cell_fate == name, :]
    # t = (t_start-t_source) / (t_target - t_source)
    # data_s = (1 - t) * data_s + t * data_trans

    num_steps_tensor = (t_target - t_source) * 10 + 1
    num_steps = int(num_steps_tensor.item())
    t_values = torch.linspace(0.0, 1.0, num_steps)
    t_values = t_values[: -1]
    V = (data_trans - data_s) / (t_target - t_source)
    # V = V.mean(0, keepdims=True)
    # pca = PCA(n_components=1)
    # pca.fit(z.cpu().detach().numpy())
    # loadings = pca.components_
    # V_weighted = V * torch.from_numpy(loadings).to(device)
    dynamic_driver_index = []
    for i, t in tqdm(enumerate(t_values), total=len(t_values), desc="Processing"):
        with torch.no_grad():
            data_inter = (1 - t) * data_s + t * data_trans
            # data_inter = data_inter.mean(0, keepdims=True)
            data_inter = data_inter * (z_max - z_min + 1e-8) + z_min
            data_org = inn(data_inter.to(torch.float).to(device), rev=True)[0]

        # data_org = data_use[inverse_indices == source-1]
        # data_org = data_org.sum(0, keepdims=True)
        data_org.requires_grad = True
        # data_org = data_org.mean(0, keepdims=True)
        data_emb = inn(data_org)[0]
        data_emb_scaled = (data_emb - z_min) / (z_max - z_min + 1e-8)
        # scalar_output = data_emb_scaled.sum()
        # data_emb_scaled = data_emb_scaled.mean(0, keepdims=True)
        # outputs = [data_emb_scaled[:, i].unsqueeze(1) for i in range(N)]
        # grad_outputs = [V[:, i].unsqueeze(1) for i in range(N)]
        weighted_gradients_summed_over_features, = autograd.grad(
            outputs=data_emb_scaled,
            inputs=data_org, grad_outputs=V,
            create_graph=False,
            only_inputs=True,
        )
        driver_index = weighted_gradients_summed_over_features.sum(dim=0)
        dynamic_driver_index.append(driver_index.detach())
    return dynamic_driver_index


import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster


def plot_phase_heatmap(data_matrix, col_color, num_gene_clusters, driver_genes_list,  title, cmap="coolwarm", figsize=(1.8, 3), fontsize=4.9, title_sz=16, path=''):

    num_cols = data_matrix.shape[1]
    col_colors = [col_color] * num_cols

    # 计算基因的行聚类：基于行数据计算层次聚类
    linkage_matrix = linkage(data_matrix, method='ward', metric='euclidean')  # 层次聚类
    ordered_rows = leaves_list(linkage_matrix)  # 聚类后得到行顺序

    # 根据聚类结果重新排列表
    sorted_matrix = data_matrix[ordered_rows, :]
    sorted_driver_genes_list = driver_genes_list[ordered_rows]
    # 使用 fcluster 根据 linkage_matrix 切割 dendrogram，得到每个原始基因的簇ID
    cluster_ids_original = fcluster(linkage_matrix, t=num_gene_clusters, criterion='maxclust')
    palette = sns.color_palette("tab10", num_gene_clusters) # 使用 tab10 调色板
    # 创建颜色列表，顺序与原始基因顺序一致
    # cluster_ids_original 是 1-based，所以需要减 1 来索引 palette (0-based)
    row_colors = [palette[cluster_id - 1] for cluster_id in cluster_ids_original[ordered_rows]]
    # 绘制热图
    g = sns.clustermap(
        sorted_matrix,
        row_colors=row_colors,      # 添加行颜色注释
        row_cluster=False,          # 不对行进行聚类
        col_cluster=False,          # 不对列进行聚类
        col_colors=col_colors,      # 添加列颜色注释
        cmap=cmap,            # 热图颜色映射
        xticklabels=False,          # 不显示x轴刻度标签
        yticklabels=sorted_driver_genes_list,          # 不显示y轴刻度标签
        figsize=figsize,            # 整个图的大小 (宽度, 高度)，增加宽度和高度
        cbar_pos=(0.65, 0.6, 0.015, 0.3),  # 调整颜色条的位置 (右侧位置与大小)
        dendrogram_ratio=(0.0001, 0.0001)      # 调整树状图比例，避免过大影响
    )

    # 添加标题
    g.fig.suptitle(title, fontsize=title_sz, x=0.24, y=0.931, color='white') # y=1.02 将标题稍微抬高一点
    plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=fontsize)  # 将字体大小设置为 fontsize
    # cbar_ax = g.cax  # 获取颜色条的Axes对象
    # pos = cbar_ax.get_position()  # 获取默认位置
    # cbar_ax.set_position([0.92, 0.2, 0.03, 0.6])  # 手动设置颜色条位置 (x, y, width, height)
    # 添加轴标签 (可选，因为刻度标签已关闭)
    # 可以通过访问 ClusterGrid 对象的 ax_heatmap 属性来获取热图的 Axes 对象
    g.ax_heatmap.set_xlabel("Time Processing")
    g.ax_heatmap.set_ylabel("Genes")
    # plt.subplots_adjust(right=0.8)
    df = pd.DataFrame(cluster_ids_original[ordered_rows])
    df.index = sorted_driver_genes_list
    df.to_csv(f'{path}/{title}.csv')
    # plt.show(block=True)
    pdf_path = f"{path}/{title}.pdf"
    g.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(g.fig)  # 关闭图像，释放内存



def plot_top_Dynamic_driver(data_All, t_All, inn, HVG, cell_fates, source, target, method='kde', reg=2e-2, name='', n_top=100, figsize=(1.8, 3), fontsize=4.9, title_sz=16, num_gene_clusters=3, path=''):
    driver_index = Detect_driver(data_All, t_All, inn, cell_fates, source=source,
                                    target=target, method=method, reg=reg, name=name)
    dynamic_driver_index_matrix = torch.stack(driver_index).cpu().numpy()
    averaged_driver_index = torch.mean(torch.stack(driver_index), dim=0)

    sorted_results = torch.sort(averaged_driver_index, descending=True)
    original_indices = sorted_results[1]  # 排序后的值对应的原始索引

    driver_genes = HVG[original_indices[0:n_top].cpu().detach().numpy()]

    data = dynamic_driver_index_matrix[:, original_indices[0:n_top].cpu().detach().numpy()].T
    data_norm = row_normalize(data)

    plot_phase_heatmap(data_norm, 'purple', num_gene_clusters, driver_genes, f"Driver Gene Dynamics with {name} path", cmap="coolwarm", figsize=figsize, fontsize=fontsize, title_sz=title_sz, path=path)
    return dynamic_driver_index_matrix, original_indices[0:n_top]


def plot_driver_exp_cor(driver_index_list, target_list, path=''):
    for i in range(len(driver_index_list)):
        df = pd.read_csv(f'{path}/{target_list[i]}.csv', index_col=0)
        df = np.array(df)
        df = df[0:(df.shape[0]-1), :]
        # --- 计算每列的相关性 ---
        correlation_coefficients = []
        for j in range(df.shape[1]):
            # 获取对应列的 Series
            true_col = driver_index_list[i][:, j]
            pseudo_col = df[:, j]
            # 计算相关性系数
            corr_value = np.corrcoef(true_col, pseudo_col)[0, 1]
            # 将相关性系数添加到列表中
            correlation_coefficients.append(corr_value)
        # --- 绘制概率密度图 (KDE) ---
        plt.figure(figsize=(10, 6))  # 设置图的大小
        sns.kdeplot(data=correlation_coefficients, fill=True)
        # 添加图的标题和标签
        plt.title(f'Distribution of Correlation Coefficients\n(Driver Index vs Gene EXpression)')
        plt.xlabel('Pearson Correlation Coefficient')
        plt.ylabel('Density')
        # 设置 X 轴范围为 -1 到 1，因为相关性系数总是介于这两个值之间
        plt.xlim(-1, 1)
        # 添加网格线 (可选)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(f"{path}/driver_exp_cor_{target_list[i]}.png", format='png')
        plt.close()  # 关闭当前图，释放内存









