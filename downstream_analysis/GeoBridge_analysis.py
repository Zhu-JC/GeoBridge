import ot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import Get_Probability_Measures
from scipy.spatial.distance import cdist
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
import matplotlib
from tqdm import tqdm
import seaborn as sns
import FISTA_OT
import os
import matplotlib.cm as cm


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

def linear_interpolation(data1, data2, num_steps=100):
    t_values = torch.linspace(0.0, 1.0, num_steps)
    interpolated_data = [(1 - t) * data1 + t * data2 for t in t_values]
    return torch.stack(interpolated_data)

def S_T_plan(data_use, inn, method='kde', source=1, target=5, reg = 2e-2):
    z, _ = inn(data_use)
    z = scaled_output(z)
    # l_unique, inverse_indices = np.unique(t_use, return_inverse=True)

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
    return P_normalized

def decide_fate(MET_weights, EMT_weights, threshold_score, threshold_weight, name1, name2):
    epsilon = 1e-8  # 防止除以零

    # 1. 计算倾向性分数 (使用对数比值)
    score = np.log2((MET_weights + epsilon) / (EMT_weights + epsilon))

    # 2. 计算最大传输量
    max_weights = np.maximum(MET_weights, EMT_weights)

    # 3. 决策规则
    cell_fate = np.full(len(MET_weights), 'Undetermined', dtype='object')  # 初始化所有细胞为未定型

    # 明确倾向 MET (高倾向, 高置信)
    met_mask = (score > threshold_score) & (max_weights > threshold_weight)
    cell_fate[met_mask] = name1

    # 明确倾向 EMT (低倾向, 高置信)
    emt_mask = (score < -threshold_score) & (max_weights > threshold_weight)
    cell_fate[emt_mask] = name2

    # 中间态/双向倾向 (倾向不明确, 高置信) - 处理“都高”的一部分
    intermediate_mask = (np.abs(score) <= threshold_score) & (max_weights > threshold_weight)
    cell_fate[intermediate_mask] = 'Intermediate'

    # 低置信/未定型 (总传输量低) - 处理“都低”和部分“都高”但总和不高的细胞
    low_confidence_mask = (max_weights <= threshold_weight)
    # 确保 low_confidence_mask 不会覆盖上面已经标记的细胞 (虽然按逻辑不应该发生，但安全起见)
    cell_fate[low_confidence_mask] = 'Low_Confidence'
    # 查看各类别细胞数量
    unique_fates, counts = np.unique(cell_fate, return_counts=True)
    for fate, count in zip(unique_fates, counts):
        print(f'{fate}: {count} cells')
    return cell_fate

def get_fate_list(All_data, All_label, inn_MET, inn_EMT, source_list, target_list, target, name1, name2):
    cell_fate_list = []
    All_source_label = []
    MET_weights_list = []
    EMT_weights_list = []
    target_type = All_label[target]
    MET_indices = np.where(target_type == target_list[0])[0]
    EMT_indices = np.where(target_type == target_list[1])[0]
    for i in source_list:
        source = (All_label == i)
        P_MET = S_T_plan(All_data, inn_MET, method='kde', source=source, target=target, reg=2e-2)
        P_EMT = S_T_plan(All_data, inn_EMT, method='kde', source=source, target=target, reg=2e-2)

        MET_weights = np.sum(P_MET[:, MET_indices], axis=1)  # 源点到目标域 MET 的总传输权重
        EMT_weights = np.sum(P_EMT[:, EMT_indices], axis=1)  # 源点到目标域 EMT 的总传输权重
        cell_fate = decide_fate(MET_weights, EMT_weights, threshold_score=0.5, threshold_weight=0.5,
                                                   name1=name1, name2=name2)
        MET_weights_list.append(MET_weights)
        EMT_weights_list.append(EMT_weights)
        cell_fate_list.append(cell_fate)
        All_source_label.extend([i] * np.sum(source))
    return cell_fate_list, All_source_label, MET_weights_list, EMT_weights_list

def fate_weights_plot(data_All, All_label, unique_labels, label_names, source_list, weights_list, inn, solid_color_map, PCs = [], spot_size=5, plot='org', path=''):
    pca = PCA(n_components=5)
    if plot == 'org':
        All_data_np = data_All.cpu().detach().numpy()
    else:
        z = inn(data_All)[0]
        z = scaled_output(z)
        All_data_np = z.cpu().detach().numpy()
    data_pca_all = pca.fit_transform(All_data_np)
    gradient_cmap = plt.cm.viridis
    target_labels_values = [unique_labels[4], unique_labels[5]]  # Get the actual label values for 30-MET and 30-EMT
    fig, ax = plt.subplots(figsize=(8, 6))

    # Store the scatter plot object for the colorbar (only need one from the gradient plots)
    gradient_scatter = None

    # Iterate through each unique label found in the data
    for i in range(len(unique_labels)):

        # Check if the current label is one of the source populations we want to color by MET weight
        if i < len(source_list):
            current_label_value = source_list[i]
            indices_with_current_label = np.where(All_label == current_label_value)[0]
            # Get the calculated MET weights for this source population
            weights = weights_list[i]

            # Plot with gradient color based on MET weights
            # 'c' argument takes an array of values for coloring
            scatter = ax.scatter(
                data_pca_all[indices_with_current_label, PCs[0]],
                data_pca_all[indices_with_current_label, PCs[1]],
                c=weights,  # Use the weights array for coloring
                cmap=gradient_cmap,  # Specify the colormap
                s=spot_size,
                alpha=1
            )
            # Store the scatter object for the colorbar (only need one)
            if gradient_scatter is None:
                gradient_scatter = scatter

        # Check if the current label is one of the target populations with a solid color
        elif i >= len(source_list):
            current_label_value = unique_labels[i]
            indices_with_current_label = np.where(All_label == current_label_value)[0]
            alpha=0.1
            ax.scatter(
                data_pca_all[indices_with_current_label, PCs[0]],
                data_pca_all[indices_with_current_label, PCs[1]],
                color=solid_color_map[label_names[i]],
                s=spot_size,
                # label=label_names[i],
                alpha=alpha
                # Get legend label from map
            )

    # Add a colorbar for the gradient points if any were plotted
    if gradient_scatter:
        cbar = fig.colorbar(gradient_scatter, ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('grey')  # 改边框颜色
        spine.set_linewidth(0)  # 线宽可选

    # Show the plotplt.grid(True, linestyle='--', alpha=0.6) # Optional: Add grid
    plt.tight_layout()  # Adjust layout to prevent labels overlapping
    plt.savefig(f"{path}_{plot}.png", format='png')
    plt.close(fig)  # 关闭当前图，释放内存


def fate_distribution_PCA(data_use, All_t, All_labels, All_source_labels, All_fate, key_label, key_fates, colors_order, PCs=[], spot_size=5, path=''):
    pca = PCA(n_components=5)
    All_data_np = data_use.cpu().detach().numpy()
    uni_fate_clusters = np.unique(All_fate)
    uni_t = np.unique(All_t)
    # PCA变换
    data_pca_all = pca.fit_transform(All_data_np)
    color_map = {fate: color for fate, color in zip(uni_fate_clusters, colors_order)}
    cmap = plt.cm.plasma
    # 绘制
    fig, ax = plt.subplots()
    for i, current_label in enumerate(uni_t):
        indices = [idx for idx, l in enumerate(All_t) if l == current_label]
        ax.scatter(data_pca_all[indices, PCs[0]],
                   data_pca_all[indices, PCs[1]],
                   c=All_t[indices], cmap=cmap, vmin=All_t.min(), vmax=All_t.max(), s=spot_size, alpha=0.1)

    indices_key = [idx for idx, l in enumerate(All_labels) if l == key_label]
    indices_key_fate = [idx for idx, l in enumerate(All_fate[All_source_labels == key_label]) if l in key_fates]
    indices = [indices_key[i] for i in indices_key_fate]
    ax.scatter(data_pca_all[indices, PCs[0]],
               data_pca_all[indices, PCs[1]],
               color=[color_map[fate] for fate in All_fate[All_source_labels == key_label][indices_key_fate]], s=spot_size, alpha=1)
    for fate in key_fates:
        ax.scatter([], [], color=color_map[fate], label=fate)
    unique, counts = np.unique(All_fate[All_source_labels == key_label][indices_key_fate], return_counts=True)
    fate_counts = dict(zip(unique, counts))
    fate_count_str = ", ".join([f"{fate}: {count}" for fate, count in fate_counts.items()])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('grey')
        spine.set_linewidth(0)
    ax.set_title(f"{key_label} | Fate counts → {fate_count_str}", fontsize=12)
    plt.legend()
    plt.savefig(f"{path}/Fate_{key_label}.png", format='png')
    plt.close(fig)  # 关闭当前图，释放内存


def plot_PCA(data_use, All_label, labels, label_names, colors_order, inn, plot='org', PCs=[], spot_size=5, is_3D=False, path=''):
    pca = PCA(n_components=5)

    # 数据选择
    if plot == 'org':
        All_data_np = data_use.cpu().detach().numpy()
    else:
        z = inn(data_use)[0]
        z = scaled_output(z)
        All_data_np = z.cpu().detach().numpy()

    # PCA变换
    data_pca_all = pca.fit_transform(All_data_np)

    # 绘制
    if is_3D:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i, color in enumerate(colors_order):
            current_label = labels[i]
            indices = [idx for idx, l in enumerate(All_label) if l == current_label]
            ax.scatter(data_pca_all[indices, PCs[0]],
                       data_pca_all[indices, PCs[1]],
                       data_pca_all[indices, PCs[2]],
                       color=color, s=spot_size, label=label_names[i])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        # 3D 的边框调整略不同，这里不设置边框线
    else:
        fig, ax = plt.subplots()
        for i, color in enumerate(colors_order):
            current_label = labels[i]
            indices = [idx for idx, l in enumerate(All_label) if l == current_label]
            ax.scatter(data_pca_all[indices, PCs[0]],
                       data_pca_all[indices, PCs[1]],
                       color=color, s=spot_size, label=label_names[i])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('grey')
            spine.set_linewidth(0)

    plt.legend()
    plt.savefig(f"{path}/PCA_{plot}.png", format='png')
    plt.close(fig)  # 关闭当前图，释放内存


def Dynamic_plot(All_data, All_label, labels, All_t, inn_MET, inn_EMT, cell_fate, method='kde', source=1, target=5, num_inter=300, reg1=2e-2, reg2=2e-2, show_velocity=10, velocity_length=3, plot='org', PCs=[], path='', name1='', name2=''):
    with torch.no_grad():
        ###MET
        z1, _ = inn_MET(All_data)
        z1_min = z1.min(dim=0, keepdim=True)[0]
        z1_max = z1.max(dim=0, keepdim=True)[0]
        z1 = scaled_output(z1)

        data_t1 = z1[target]
        data_t1 = pd.DataFrame(data_t1.cpu().detach().numpy())
        data_s1 = z1[source]
        data_s1 = pd.DataFrame(data_s1.cpu().detach().numpy())
        C = cdist(data_s1.values, data_t1.values, metric='euclidean')
        if method == 'neighbor':
            mu = Get_Probability_Measures.Neighbor_Measures(data_s1, 10, epsilon=1e-5)
            mu = mu.to_numpy()
            nu = Get_Probability_Measures.Neighbor_Measures(data_t1, 10, epsilon=1e-5)
            nu = nu.to_numpy()
        else:
            mu = Get_Probability_Measures.kde_gene_expression(data_s1)
            nu = Get_Probability_Measures.kde_gene_expression(data_t1)
        P = ot.sinkhorn(mu, nu, C, reg=reg1)
        P_normalized = P / P.sum(axis=1, keepdims=True)
        data_trans1 = np.dot(P_normalized, data_t1.values)
        data_trans1 = torch.from_numpy(data_trans1).to(device).to(torch.float32)
        data_trans1 = data_trans1[cell_fate == name1, :]
        data_s1 = torch.from_numpy(data_s1.values).to(device)
        data_s1 = data_s1[cell_fate == name1, :]
        if plot == 'org':
            data_trans1 = data_trans1 * (z1_max - z1_min + 1e-8) + z1_min
            data_s1 = data_s1 * (z1_max - z1_min + 1e-8) + z1_min

        ###EMT
        z2, _ = inn_EMT(All_data)
        z2_min = z2.min(dim=0, keepdim=True)[0]
        z2_max = z2.max(dim=0, keepdim=True)[0]
        z2 = scaled_output(z2)

        data_t2 = z2[target]
        data_t2 = pd.DataFrame(data_t2.cpu().detach().numpy())
        data_s2 = z2[source]
        data_s2 = pd.DataFrame(data_s2.cpu().detach().numpy())
        C = cdist(data_s2.values, data_t2.values, metric='euclidean')
        if method == 'neighbor':
            mu = Get_Probability_Measures.Neighbor_Measures(data_s2, 10, epsilon=1e-5)
            mu = mu.to_numpy()
            nu = Get_Probability_Measures.Neighbor_Measures(data_t2, 10, epsilon=1e-5)
            nu = nu.to_numpy()
        else:
            mu = Get_Probability_Measures.kde_gene_expression(data_s2)
            nu = Get_Probability_Measures.kde_gene_expression(data_t2)
        P = ot.sinkhorn(mu, nu, C, reg=reg2)
        P_normalized = P / P.sum(axis=1, keepdims=True)
        data_trans2 = np.dot(P_normalized, data_t2.values)
        data_trans2 = torch.from_numpy(data_trans2).to(device).to(torch.float32)
        data_trans2 = data_trans2[cell_fate == name2, :]
        data_s2 = torch.from_numpy(data_s2.values).to(device)
        data_s2 = data_s2[cell_fate == name2, :]
        if plot == 'org':
            data_trans2 = data_trans2 * (z2_max - z2_min + 1e-8) + z2_min
            data_s2 = data_s2 * (z2_max - z2_min + 1e-8) + z2_min

        pca = PCA(n_components=5)
        if plot == 'org':
            All_data_np = All_data.cpu().detach().numpy()
        elif plot == f'{name1}_eu':
            All_data_np = z1.cpu().detach().numpy()
        elif plot == f'{name2}_eu':
            All_data_np = z2.cpu().detach().numpy()
        data_pca_all = pca.fit_transform(All_data_np)

        # 绘制散点图
        fig, ax = plt.subplots()  # 使用 plt.subplots 创建二维图
        cmap = plt.cm.plasma
        for i, color in enumerate(labels):
            # 找出当前颜色对应的标签
            current_label = labels[i]
            indices = [idx for idx, l in enumerate(All_label) if l == current_label]

            # 绘制二维散点图，去掉 z 轴
            ax.scatter(data_pca_all[indices, PCs[0]], data_pca_all[indices, PCs[1]], c=All_t[indices], s=5, alpha=0.1, cmap=cmap, vmin=All_t.min(), vmax=All_t.max())

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('grey')  # 改边框颜色
            spine.set_linewidth(0)  # 线宽可选
        t_values = torch.linspace(0.0, 1.0, num_inter+1)
        t_values = t_values[0:]
        # mmd_results = []
        # 打印进度的间隔
        print_interval = max(1, num_inter // 10)  # 每10%打印一次
        # 初始化两个 scatter 对象
        sc1 = ax.scatter([], [], c='green', s=5, label=f'Path {name1}')
        sc2 = ax.scatter([], [], c='orange', s=5, label=f'Path {name2}')
        quiv1 = ax.quiver([0] * show_velocity, [0] * show_velocity, [0] * show_velocity, [0] * show_velocity,
                          angles='xy', scale_units='xy', scale=1, color='black')
        quiv2 = ax.quiver([0] * show_velocity, [0] * show_velocity, [0] * show_velocity, [0] * show_velocity,
                          angles='xy', scale_units='xy', scale=1, color='black')
        def init():
            """初始化动画背景"""
            sc1.set_offsets(np.empty((0, 2)))
            sc2.set_offsets(np.empty((0, 2)))
            quiv1.set_offsets(np.empty((0, 2)))
            quiv1.set_UVC([0] * show_velocity, [0] * show_velocity)  # 清空方向
            quiv2.set_offsets(np.empty((0, 2)))
            quiv2.set_UVC([0] * show_velocity, [0] * show_velocity)
            return sc1, sc2, quiv1, quiv2

        def update(frame):
            """更新每一帧的函数"""
            t = t_values[frame].item()
            with torch.no_grad():
                # 计算插值数据
                data_inter1 = (1 - t) * data_s1 + t * data_trans1
                data_inter2 = (1 - t) * data_s2 + t * data_trans2
                if plot == 'org':
                    data_inter1 = inn_MET(data_inter1.to(torch.float).to(device), rev=True)[0]
                    data_inter2 = inn_EMT(data_inter2.to(torch.float).to(device), rev=True)[0]

                # 转换为 NumPy 数组
                data_org_np1 = data_inter1.cpu().detach().numpy()
                data_org_np2 = data_inter2.cpu().detach().numpy()

                # PCA 降维
                data_pca_org1 = pca.transform(data_org_np1)
                data_pca_org2 = pca.transform(data_org_np2)

                # 更新 scatter 的数据
                if plot == 'org':
                    sc1.set_offsets(data_pca_org1[:, PCs])
                    sc2.set_offsets(data_pca_org2[:, PCs])
                elif plot == f'{name1}_eu':
                    sc1.set_offsets(data_pca_org1[:, PCs])
                elif plot == f'{name2}_eu':
                    sc2.set_offsets(data_pca_org2[:, PCs])


                # 同样步骤计算下一时刻，为了求速度
                t_next = t+0.01
                data_inter1_next = (1 - t_next) * data_s1[0:show_velocity, :] + t_next * data_trans1[0:show_velocity, :]
                data_inter2_next = (1 - t_next) * data_s2[0:show_velocity, :] + t_next * data_trans2[0:show_velocity, :]
                if plot == 'org':
                    data_inter1_next = inn_MET(data_inter1_next.to(torch.float).to(device), rev=True)[0]
                    data_inter2_next = inn_EMT(data_inter2_next.to(torch.float).to(device), rev=True)[0]
                data_org_np1_next = data_inter1_next.cpu().detach().numpy()
                data_org_np2_next = data_inter2_next.cpu().detach().numpy()
                data_pca_org1_next = pca.transform(data_org_np1_next)
                data_pca_org2_next = pca.transform(data_org_np2_next)
                velocity1 = (data_pca_org1_next[:, PCs] - data_pca_org1[0:show_velocity, PCs]) / 0.01
                velocity2 = (data_pca_org2_next[:, PCs] - data_pca_org2[0:show_velocity, PCs]) / 0.01
                norms1 = np.linalg.norm(velocity1, axis=1, keepdims=True)  # 每个向量长度
                norms1[norms1 == 0] = 1  # 防止除以零
                velocity1_norm = (velocity1 / norms1)*velocity_length
                norms2 = np.linalg.norm(velocity2, axis=1, keepdims=True)
                norms2[norms2 == 0] = 1
                velocity2_norm = (velocity2 / norms2)*velocity_length
                # 更新箭头（quiver）
                if plot == 'org':
                    quiv1.set_offsets(data_pca_org1[:show_velocity, PCs])
                    quiv1.set_UVC(velocity1_norm[:, 0], velocity1_norm[:, 1])
                    quiv2.set_offsets(data_pca_org2[:show_velocity, PCs])
                    quiv2.set_UVC(velocity2_norm[:, 0], velocity2_norm[:, 1])
                elif plot == f'{name1}_eu':
                    quiv1.set_offsets(data_pca_org1[:show_velocity, PCs])
                    quiv1.set_UVC(velocity1_norm[:, 0], velocity1_norm[:, 1])
                elif plot == f'{name2}_eu':
                    quiv2.set_offsets(data_pca_org2[:show_velocity, PCs])
                    quiv2.set_UVC(velocity2_norm[:, 0], velocity2_norm[:, 1])

            # 打印进度
            if (frame + 1) % print_interval == 0 or frame == num_inter - 1:
                print(f'[{frame + 1}/{num_inter}]')

            return sc1, sc2

        ani = FuncAnimation(
            fig,
            update,
            frames=num_inter+1,
            init_func=init,
            blit=True,
            interval=100
        )
        # plt.show(block=True)  # 这行确保动画运行完毕后再执行后续代码
        ani.save(f'{path}.gif', writer='imagemagick')
        plt.close()


def get_data_trans(data_s, data_t, reg, method='neighbor'):

    data_t = pd.DataFrame(data_t.cpu().detach().numpy())
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
    return data_trans



def OT_gene_Mean_Dynamic_compair(data_use, t_use, inn_MET, inn_EMT, cell_fate, genenames, method='kde', source=1, target=5, num_inter=300, use='org', reg1=1e-2, reg2=2e-2, path='', name1='', name2=''):
    ###MET
    z1, _ = inn_MET(data_use)
    z1_min = z1.min(dim=0, keepdim=True)[0]
    z1_max = z1.max(dim=0, keepdim=True)[0]
    z1 = scaled_output(z1)

    data_t1 = z1[target]
    data_t1 = pd.DataFrame(data_t1.cpu().detach().numpy())
    data_s1 = z1[source]
    data_s1 = pd.DataFrame(data_s1.cpu().detach().numpy())
    C = cdist(data_s1.values, data_t1.values, metric='euclidean')
    if method == 'neighbor':
        mu = Get_Probability_Measures.Neighbor_Measures(data_s1, 10, epsilon=1e-5)
        mu = mu.to_numpy()
        nu = Get_Probability_Measures.Neighbor_Measures(data_t1, 10, epsilon=1e-5)
        nu = nu.to_numpy()
    else:
        mu = Get_Probability_Measures.kde_gene_expression(data_s1)
        nu = Get_Probability_Measures.kde_gene_expression(data_t1)
    P = ot.sinkhorn(mu, nu, C, reg=reg1)
    P_normalized = P / P.sum(axis=1, keepdims=True)
    data_trans1 = np.dot(P_normalized, data_t1.values)
    data_trans1 = torch.from_numpy(data_trans1).to(device).to(torch.float32)
    data_trans1 = data_trans1[cell_fate == name1, :]
    data_s1 = torch.from_numpy(data_s1.values).to(device)
    data_s1 = data_s1[cell_fate == name1, :]
    data_trans1 = data_trans1 * (z1_max - z1_min + 1e-8) + z1_min
    data_s1 = data_s1 * (z1_max - z1_min + 1e-8) + z1_min

    ###EMT
    z2, _ = inn_EMT(data_use)
    z2_min = z2.min(dim=0, keepdim=True)[0]
    z2_max = z2.max(dim=0, keepdim=True)[0]
    z2 = scaled_output(z2)
    l_unique, inverse_indices = torch.unique(t_use, return_inverse=True)

    data_t2 = z2[target]
    data_t2 = pd.DataFrame(data_t2.cpu().detach().numpy())
    data_s2 = z2[source]
    data_s2 = pd.DataFrame(data_s2.cpu().detach().numpy())
    C = cdist(data_s2.values, data_t2.values, metric='euclidean')
    if method == 'neighbor':
        mu = Get_Probability_Measures.Neighbor_Measures(data_s2, 10, epsilon=1e-5)
        mu = mu.to_numpy()
        nu = Get_Probability_Measures.Neighbor_Measures(data_t2, 10, epsilon=1e-5)
        nu = nu.to_numpy()
    else:
        mu = Get_Probability_Measures.kde_gene_expression(data_s2)
        nu = Get_Probability_Measures.kde_gene_expression(data_t2)
    P = ot.sinkhorn(mu, nu, C, reg=reg2)
    P_normalized = P / P.sum(axis=1, keepdims=True)
    data_trans2 = np.dot(P_normalized, data_t2.values)
    data_trans2 = torch.from_numpy(data_trans2).to(device).to(torch.float32)
    data_trans2 = data_trans2[cell_fate == name2, :]
    data_s2 = torch.from_numpy(data_s2.values).to(device)
    data_s2 = data_s2[cell_fate == name2, :]
    data_trans2 = data_trans2 * (z2_max - z2_min + 1e-8) + z2_min
    data_s2 = data_s2 * (z2_max - z2_min + 1e-8) + z2_min

    t_values = torch.linspace(0.0, 1.0, num_inter+1)
    mean_data1 = []
    mean_data2 = []
    std_data1 = []
    std_data2 = []
    for i, t in enumerate(t_values):
        with torch.no_grad():
            data_inter1 = (1 - t) * data_s1 + t * data_trans1
            data_inter2 = (1 - t) * data_s2 + t * data_trans2
            if use == 'org':
                data_inter_org1 = inn_MET(data_inter1.to(torch.float).to(device), rev=True)[0]
                mean_inter1 = data_inter_org1.mean(dim=0)
                std_inter1 = data_inter_org1.std(axis=0)
                data_inter_org2 = inn_EMT(data_inter2.to(torch.float).to(device), rev=True)[0]
                std_inter2 = data_inter_org2.std(axis=0)
                mean_inter2 = data_inter_org2.mean(dim=0)
            else:
                mean_inter1 = data_inter1.mean(dim=0).unsqueeze(0)
                mean_inter1 = inn_MET(mean_inter1.to(torch.float).to(device), rev=True)[0][0]
                mean_inter2 = data_inter2.mean(dim=0).unsqueeze(0)
                mean_inter2 = inn_EMT(mean_inter2.to(torch.float).to(device), rev=True)[0][0]
            mean_data1.append(mean_inter1)
            mean_data2.append(mean_inter2)
            std_data1.append(std_inter1)
            std_data2.append(std_inter2)
    mean_data1 = torch.stack(mean_data1)
    mean_data2 = torch.stack(mean_data2)
    std_data1 = torch.stack(std_data1).detach().cpu().numpy()
    std_data2 = torch.stack(std_data2).detach().cpu().numpy()
    t_target = l_unique.max().to(torch.float32)
    t_source = l_unique.min().to(torch.float32)
    t_np = (t_values * (t_target - t_source).cpu()).detach().cpu().numpy()
    # 绘制散点图
    os.makedirs(path, exist_ok=True)
    # 将 tensor 转换为 NumPy 数组以便 matplotlib 使用
    filtered_data_np1 = mean_data1.detach().cpu().numpy()
    filtered_data_np2 = mean_data2.detach().cpu().numpy()
    # filtered_data_np1[filtered_data_np1 < 0] = 0
    pd.DataFrame(filtered_data_np1).to_csv(f"{path}/{name1}.csv")
    # filtered_data_np2[filtered_data_np2 < 0] = 0
    pd.DataFrame(filtered_data_np2).to_csv(f"{path}/{name2}.csv")
    data = data_use.cpu().detach().numpy()
    t = t_use
    means = np.array([data[t == lbl].mean(axis=0) for lbl in np.unique(t)])
    matplotlib.use('Agg')
    # 绘制每个特征的散点图
    for i in tqdm(range(filtered_data_np1.shape[1])):
        fig = plt.figure(figsize=(6, 4))
        ax = plt.gca()
        # 绘制 filtered_data_np1，标签为 "MET"
        plt.plot(t_np, filtered_data_np1[:, i], alpha=0.6, linestyle='-', color='orange', linewidth=3)
        ax.fill_between(t_np,
                        filtered_data_np1[:, i] - std_data1[:, i],
                        filtered_data_np1[:, i] + std_data1[:, i],
                        color='orange', alpha=0.2)
        gene_name = genenames[i]
        sanitized_gene_name = gene_name.replace('/', '_').replace('\\', '_')
        # 绘制 filtered_data_np2，标签为 "EMT"
        plt.plot(t_np, filtered_data_np2[:, i], alpha=0.6, linestyle='-', color='green', linewidth=3)
        ax.fill_between(t_np,
                        filtered_data_np2[:, i] - std_data2[:, i],
                        filtered_data_np2[:, i] + std_data2[:, i],
                        color='green', alpha=0.2)
        plt.scatter(np.unique(t)-t.min(), means[:, i], alpha=0.6, label='True Data', marker='o', s=40, color='red')
        plt.title(sanitized_gene_name)  # 设置图标题为特征名
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        # plt.xlabel("t")
        # plt.ylabel(f"{sanitized_gene_name} expression")
        plt.grid(False)
        for spine in ax.spines.values():
            spine.set_edgecolor('grey')  # 改边框颜色
            spine.set_linewidth(1)  # 线宽可选
        # plt.legend()  # 显示图例
        # 保存图片，文件名为特征名
        plt.savefig(f"{path}/{sanitized_gene_name}.png", format='png')
        plt.close(fig)  # 关闭当前图，释放内存



def Plot_fate_path(All_data, t_use, All_label, unique_labels, label_names, source_list, weights_list, inn, cell_fate, solid_color_map, method = 'kde', source=4, target=5, reg = 2e-2, PCs=[], name='', path='', plot='org', n_path=50, color='pink', traj_alpha=1, traj_width=1, linewidth=2, arrow_size=0.4):
    with torch.no_grad():
        z, _ = inn(All_data)
        z_min = z.min(dim=0, keepdim=True)[0]
        z_max = z.max(dim=0, keepdim=True)[0]
        z = scaled_output(z)
        l_unique, inverse_indices = torch.unique(t_use, return_inverse=True)

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
        data_trans = data_trans[cell_fate == name, :]
        data_s = torch.from_numpy(data_s.values).to(device)
        data_s = data_s[cell_fate == name, :]
        indices = torch.randperm(data_trans.shape[0])[:n_path]
        data_trans = data_trans[indices, :]
        data_s = data_s[indices, :]
        if plot == 'org':
            data_trans = (data_trans * (z_max - z_min + 1e-8) + z_min)
            data_s = (data_s * (z_max - z_min + 1e-8) + z_min)

        pca = PCA(n_components=5)
        if plot == 'org':
            All_data_np = All_data.cpu().detach().numpy()
        else:
            All_data_np = z.cpu().detach().numpy()

        data_pca_all = pca.fit_transform(All_data_np)

        t_target = l_unique.max().to(torch.float32)
        t_source = l_unique.min().to(torch.float32)
        num_steps_tensor = (t_target - t_source) * 50 + 1
        num_steps = int(num_steps_tensor.item())
        t_values = torch.linspace(0.0, 1.0, num_steps+1)
        t_values = t_values[0:].to(device)
        matplotlib.use('Agg')
        fig, ax = plt.subplots()  # 使用 plt.subplots 创建二维图
        # 绘制散点图
        indice = np.where(source_list == np.unique(All_label[source]))
        for i in range(len(unique_labels)):
            if i < indice[0]:
                continue
            if i < len(source_list):
                current_label_value = source_list[i]
                indices_with_current_label = np.where(All_label == current_label_value)[0]
                # Get the calculated MET weights for this source population
                weights = weights_list[i]

                scatter = ax.scatter(
                    data_pca_all[indices_with_current_label, PCs[0]],
                    data_pca_all[indices_with_current_label, PCs[1]],
                    c='grey',  # Use the weights array for coloring
                    s=5,
                    alpha=weights*0.5
                )

            # Check if the current label is one of the target populations with a solid color
            elif i >= len(source_list):
                current_label_value = unique_labels[i]
                indices_with_current_label = np.where(All_label == current_label_value)[0]
                # Plot with the predefined solid color
                # color = solid_color_map[current_label_value]
                if label_names[i] == name:
                    alpha = 0.4
                else:
                    alpha = 0.1
                ax.scatter(
                    data_pca_all[indices_with_current_label, PCs[0]],
                    data_pca_all[indices_with_current_label, PCs[1]],
                    color=solid_color_map[label_names[i]],
                    s=5,
                    # label=label_names[i],
                    alpha=alpha
                    # Get legend label from map
                )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('grey')  # 改边框颜色
            spine.set_linewidth(0)  # 线宽可选
        avg_points = 0
        trajectories = []
        # for i, t in enumerate(t_values):
        with torch.no_grad():
            for i in range(n_path):
                interpolated_data = (1 - t_values.unsqueeze(1)) * data_s[i, :] + t_values.unsqueeze(1) * data_trans[i, :]
                avg_points = avg_points + interpolated_data
                if plot == 'org':
                    org_z = inn(interpolated_data, rev=True)[0]
                    trajectory = pca.transform(org_z.cpu().detach().numpy())
                    trajectories.append(trajectory)
            avg_points = avg_points / n_path
            avg_points_org = inn(avg_points.to(torch.float).to(device), rev=True)[0]
            for trajectory in trajectories:
                plt.plot(trajectory[:, PCs[0]], trajectory[:, PCs[1]], alpha=traj_alpha, color=color, linewidth=traj_width);
            # avg_points = np.vstack(avg_points)  # shape = (len(t_values), features)
            avg_path_pca = pca.transform(avg_points_org.cpu().detach().numpy())
            plt.plot(avg_path_pca[:, PCs[0]], avg_path_pca[:, PCs[1]], linestyle='--', color='black', linewidth=linewidth)

            # 加箭头
            j = len(avg_path_pca) - 2  # 倒数第二个点（作为箭头起点）
            dx = avg_path_pca[-1, PCs[0]] - avg_path_pca[j, PCs[0]]
            dy = avg_path_pca[-1, PCs[1]] - avg_path_pca[j, PCs[1]]
            plt.arrow(avg_path_pca[j, PCs[0]], avg_path_pca[j, PCs[1]], dx, dy,
                      shape='full', head_width=arrow_size, color='black',
                      length_includes_head=True, zorder=10)

            plt.savefig(f'{path}_{name}_single_path_{plot}.png', bbox_inches='tight')
            plt.close(fig)


def plot_pseudotime(data_use, inn, cell_fate_list, All_label, labels, source_list, source=1, target=5, reg = 2e-2, target_label='', name='', PCs=[]):
    z, _ = inn(data_use)
    z = scaled_output(z)
    # l_unique, inverse_indices = np.unique(t_use, return_inverse=True)

    data_t = z[target]
    data_t = pd.DataFrame(data_t.cpu().detach().numpy())
    data_s = z[source]
    data_s = pd.DataFrame(data_s.cpu().detach().numpy())
    mu = Get_Probability_Measures.kde_gene_expression(data_s)
    nu = Get_Probability_Measures.kde_gene_expression(data_t)
    C = cdist(data_s.values, data_t.values, metric='euclidean')
    KP = -FISTA_OT.fista_ot2(C, mu, nu, lambda_reg=0.1, max_iter=1000, tol=1e-6)[1]

    KP_max = KP.max()
    KP_min = KP.min()
    KP_scaled = (KP-KP_min)/(KP_max-KP_min)


    pca = PCA(n_components=5)
    All_data_np = data_use.cpu().detach().numpy()
    data_pca_all = pca.fit_transform(All_data_np)

    # 绘制散点图

    gradient_cmap = plt.cm.viridis  # You can choose other colormaps, e.g., 'plasma', 'inferno', 'Reds'
    # Map unique label values to desired solid colors and legend labels
    solid_color_map = {
        labels[4]: 'brown',  # Color for unique_labels[4] (assuming 30-MET)
        labels[5]: 'purple'  # Color for unique_labels[5] (assuming 30-EMT)
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    # Store the scatter plot object for the colorbar (only need one from the gradient plots)
    gradient_scatter = None

    # Iterate through each unique label found in the data
    for i in range(len(labels)-1):

        # Check if the current label is one of the source populations we want to color by MET weight
        if i < len(source_list):
            current_label_value = source_list[i]
            indices_with_current_label = np.where(All_label == current_label_value)[0]
            sub_indices = indices_with_current_label[cell_fate_list[i] == name]
            # Get the calculated MET weights for this source population
            weights = KP_scaled[indices_with_current_label]

            # Plot with gradient color based on MET weights
            # 'c' argument takes an array of values for coloring
            scatter = ax.scatter(
                data_pca_all[indices_with_current_label, PCs[0]],
                data_pca_all[indices_with_current_label, PCs[1]],
                c=weights,  # Use the weights array for coloring
                cmap=gradient_cmap,  # Specify the colormap
                s=5
            )
            # Store the scatter object for the colorbar (only need one)
            if gradient_scatter is None:
                gradient_scatter = scatter

        # Check if the current label is one of the target populations with a solid color
        elif i >= len(source_list):
            indices_with_current_label = np.where((All_label == labels[4]) | (All_label == labels[5]))[0]
            sub_indices = indices_with_current_label[cell_fate_list[i] == target_label]
            # Plot with the predefined solid color
            color = solid_color_map[target_label]
            ax.scatter(
                data_pca_all[indices_with_current_label, PCs[0]],
                data_pca_all[indices_with_current_label, PCs[1]],
                color=color,
                s=5,
                label=name
                # Get legend label from map
            )

    # Add a colorbar for the gradient points if any were plotted
    if gradient_scatter:
        cbar = fig.colorbar(gradient_scatter, ax=ax)
        cbar.set_label(f'{name} Weight')

    # Add legend (will show both gradient colormaps and solid color swatches)
    ax.legend(loc='best')  # Adjust legend location as needed

    # Set title and labels
    ax.set_title(f'PCA Embedding (2D) Colored by {name} Weight')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    # Show the plotplt.grid(True, linestyle='--', alpha=0.6) # Optional: Add grid
    plt.tight_layout()  # Adjust layout to prevent labels overlapping
    plt.show(block=True)


def new_plot_comparisions(
        df, trajectories, avg_list,
        palette='viridis',
        df_time_key='samples',
        x=0, y=1, plot_avg=False, use_color=[], traj_alpha=0.3, traj_width=1.5,
        groups=None, arrow_size=0.5, avg_width=3
):
    if groups is None:
        groups = sorted(df[df_time_key].unique())
    cmap = cm.get_cmap(palette)
    sns.set_palette(palette)
    plt.rcParams.update({
        'axes.prop_cycle': plt.cycler(color=cmap(np.linspace(0, 1, len(groups) + 1))),
        'axes.axisbelow': False,
        'axes.edgecolor': 'lightgrey',
        'axes.facecolor': 'None',
        'axes.grid': False,
        'axes.labelcolor': 'dimgrey',
        'axes.spines.right': False,
        'axes.spines.top': False,
        'figure.facecolor': 'white',
        'lines.solid_capstyle': 'round',
        'patch.edgecolor': 'w',
        'patch.force_edgecolor': True,
        'text.color': 'dimgrey',
        'xtick.bottom': False,
        'xtick.color': 'dimgrey',
        'xtick.direction': 'out',
        'xtick.top': False,
        'ytick.color': 'dimgrey',
        'ytick.direction': 'out',
        'ytick.left': False,
        'ytick.right': False,
        'font.size': 12,
        'axes.titlesize': 10,
        'axes.labelsize': 12
    })

    n_cols = 1
    n_rols = 1

    grid_figsize = [12, 8]
    dpi = 80
    grid_figsize = (grid_figsize[0] * n_cols, grid_figsize[1] * n_rols)
    fig = plt.figure(None, grid_figsize, dpi=dpi)

    hspace = 0.3
    wspace = None
    gspec = plt.GridSpec(n_rols, n_cols, fig, hspace=hspace, wspace=wspace)

    outline_width = (0.3, 0.05)
    size = 80
    bg_width, gap_width = outline_width
    point = np.sqrt(size)

    gap_size = (point + (point * gap_width) * 2) ** 2
    bg_size = (np.sqrt(gap_size) + (point * bg_width) * 2) ** 2

    # plt.legend(frameon=False)
    states = sorted(df[df_time_key].unique())
    vmin = min(states)
    vmax = max(states)
    axs = []
    for i, gs in enumerate(gspec):
        ax = plt.subplot(gs)

        ax.scatter(
            df[x], df[y],
            c=df[df_time_key],
            s=size,
            alpha=0.1,
            marker='X',
            linewidths=0,
            edgecolors=None,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )

        for i, trajectory in enumerate(trajectories):
            for single_trajectory in trajectory:
                if len(use_color)>0:
                    color = use_color[i]
                else:
                    color = 'black'
                plt.plot(single_trajectory[:, x], single_trajectory[:, y], alpha=traj_alpha, color=color, linewidth=traj_width);
        if plot_avg == True:
            for avg_path in avg_list:
                plt.plot(avg_path[:, x], avg_path[:, y], linestyle='--', color='black', linewidth=avg_width)
                j = len(avg_path) - 2  # 倒数第二个点（作为箭头起点）
                dx = avg_path[-1, x] - avg_path[j, x]
                dy = avg_path[-1, y] - avg_path[j, y]
                plt.arrow(avg_path[j, x], avg_path[j, y], dx, dy,
                          shape='full', head_width=arrow_size, color='black',
                          length_includes_head=True)

        # legend_elements = [
        #     Line2D(
        #         [0], [0], marker='o',
        #         color=cmap((i) / (len(states) - 1)), label=f'Day {state}',
        #         markerfacecolor=cmap((i) / (len(states) - 1)), markersize=15,
        #     )
        #     for i, state in enumerate(states)
        # ]
        #
        # leg = ax.legend(handles=legend_elements, loc='upper left')
        # ax.add_artist(leg)

        # legend_elements = [
        #     Line2D([0], [0], color='black', lw=2, label='Trajectory')
        #
        # ]
        # leg = plt.legend(handles=legend_elements, loc='upper right')
        # ax.add_artist(leg)

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('grey')  # 改边框颜色
            spine.set_linewidth(0)  # 线宽可选
        ax.get_xaxis().get_major_formatter().set_scientific(False)
        ax.get_yaxis().get_major_formatter().set_scientific(False)
        # kwargs = dict(bottom=False, left=False, labelbottom=False, labelleft=False)
        # ax.tick_params(which="both", **kwargs)
        # ax.set_frame_on(False)
        ax.patch.set_alpha(0)

        axs.append(ax)
    return fig


def pair_path_plot(data_use, All_t, inn, num_inter, traj_width=1.5, cmap='plasma', PCs=[], path=''):
    z, _ = inn(data_use)
    z_min = z.min(dim=0, keepdim=True)[0]
    z_max = z.max(dim=0, keepdim=True)[0]
    z = scaled_output(z)
    indices = torch.randperm(z.shape[0])[:2]
    z1 = z[indices[0], :]
    z2 = z[indices[1], :]
    z_interp = linear_interpolation(z1, z2, num_steps=num_inter)

    data = z_interp * (z_max - z_min + 1e-8) + z_min
    data_org = inn(data, rev=True)[0]
    # 将 data_org 和 All_data 转换为 NumPy 数组
    All_data_np = data_use.cpu().detach().numpy()

    # 1. 对 All_data 进行 PCA 降维
    pca = PCA(n_components=5)
    data_pca_all = pca.fit_transform(All_data_np)

    # 2. 使用相同的 PCA 模型对 data_org 降维
    data_org_np = data_org.cpu().detach().numpy()
    data_pca_org = pca.transform(data_org_np)
    df = pd.DataFrame(data_pca_all)
    df.insert(0, 'samples', All_t)
    trajectories = []
    trajectories.append(data_pca_org)
    fig = new_plot_comparisions(df, [trajectories], avg_list=[], palette=cmap, x=PCs[0], y=PCs[1], plot_avg=False,
                                use_color=[], traj_alpha=1, traj_width=traj_width)
    save_path = f"{path}/sc_navigation"
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(f"{save_path}/{indices[0]}_{indices[1]}_org.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    data_pca_eu = pca.fit_transform(z.cpu().detach().numpy())
    data_traj_eu = pca.transform(z_interp.cpu().detach().numpy())
    df = pd.DataFrame(data_pca_eu)
    df.insert(0, 'samples', All_t)
    trajectories = []
    trajectories.append(data_traj_eu)
    fig = new_plot_comparisions(df, [trajectories], avg_list=[], palette=cmap, x=PCs[0], y=PCs[1], plot_avg=False,
                                use_color=[], traj_alpha=1, traj_width=traj_width)

    fig.savefig(f"{save_path}/{indices[0]}_{indices[1]}_eu.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def inter_plot(All_data, All_t, inn_MET, inn_EMT, cell_fate, traj_num=30, method='kde', source=1, target=5, num_inter=70, reg1=2e-2, reg2=2e-2, PCs=[], name1='', name2='', plot='org', cmap='plasma', arrow_size=0.5, avg_width=3, path=''):
    with torch.no_grad():
        ###MET
        z1, _ = inn_MET(All_data)
        z1_min = z1.min(dim=0, keepdim=True)[0]
        z1_max = z1.max(dim=0, keepdim=True)[0]
        z1 = scaled_output(z1)

        data_t1 = z1[target]
        data_t1 = pd.DataFrame(data_t1.cpu().detach().numpy())
        data_s1 = z1[source]
        data_s1 = pd.DataFrame(data_s1.cpu().detach().numpy())
        C = cdist(data_s1.values, data_t1.values, metric='euclidean')
        if method == 'neighbor':
            mu = Get_Probability_Measures.Neighbor_Measures(data_s1, 10, epsilon=1e-5)
            mu = mu.to_numpy()
            nu = Get_Probability_Measures.Neighbor_Measures(data_t1, 10, epsilon=1e-5)
            nu = nu.to_numpy()
        else:
            mu = Get_Probability_Measures.kde_gene_expression(data_s1)
            nu = Get_Probability_Measures.kde_gene_expression(data_t1)
        P = ot.sinkhorn(mu, nu, C, reg=reg1)
        P_normalized = P / P.sum(axis=1, keepdims=True)
        data_trans1 = np.dot(P_normalized, data_t1.values)
        data_trans1 = torch.from_numpy(data_trans1).to(device).to(torch.float32)
        data_trans1 = data_trans1[cell_fate == name1, :]
        data_s1 = torch.from_numpy(data_s1.values).to(device)
        data_s1 = data_s1[cell_fate == name1, :]
        if plot == 'org':
            data_trans1 = data_trans1 * (z1_max - z1_min + 1e-8) + z1_min
            data_s1 = data_s1 * (z1_max - z1_min + 1e-8) + z1_min

        ###EMT
        z2, _ = inn_EMT(All_data)
        z2_min = z2.min(dim=0, keepdim=True)[0]
        z2_max = z2.max(dim=0, keepdim=True)[0]
        z2 = scaled_output(z2)

        data_t2 = z2[target]
        data_t2 = pd.DataFrame(data_t2.cpu().detach().numpy())
        data_s2 = z2[source]
        data_s2 = pd.DataFrame(data_s2.cpu().detach().numpy())
        C = cdist(data_s2.values, data_t2.values, metric='euclidean')
        if method == 'neighbor':
            mu = Get_Probability_Measures.Neighbor_Measures(data_s2, 10, epsilon=1e-5)
            mu = mu.to_numpy()
            nu = Get_Probability_Measures.Neighbor_Measures(data_t2, 10, epsilon=1e-5)
            nu = nu.to_numpy()
        else:
            mu = Get_Probability_Measures.kde_gene_expression(data_s2)
            nu = Get_Probability_Measures.kde_gene_expression(data_t2)
        P = ot.sinkhorn(mu, nu, C, reg=reg2)
        P_normalized = P / P.sum(axis=1, keepdims=True)
        data_trans2 = np.dot(P_normalized, data_t2.values)
        data_trans2 = torch.from_numpy(data_trans2).to(device).to(torch.float32)
        data_trans2 = data_trans2[cell_fate == name2, :]
        data_s2 = torch.from_numpy(data_s2.values).to(device)
        data_s2 = data_s2[cell_fate == name2, :]
        if plot == 'org':
            data_trans2 = data_trans2 * (z2_max - z2_min + 1e-8) + z2_min
            data_s2 = data_s2 * (z2_max - z2_min + 1e-8) + z2_min
        # 1. 对 All_data 进行 PCA 降维
        pca = PCA(n_components=5)
        if plot == 'org':
            All_data_np = All_data.cpu().detach().numpy()
        elif plot == f'{name1}_eu':
            All_data_np = z1.cpu().detach().numpy()
        elif plot == f'{name2}_eu':
            All_data_np = z2.cpu().detach().numpy()

        data_pca_all = pca.fit_transform(All_data_np)
        t_values = torch.linspace(0.0, 1.0, num_inter+1)
        t_values = t_values[0:]
        df = pd.DataFrame(data_pca_all)
        df.insert(0, 'samples', All_t)

        t_values = torch.linspace(0.0, 1.0, num_inter+1)
        t_values = t_values[0:].to(device)
        trajectories1 = []
        trajectories2 = []
        avg_points1 = 0
        avg_points2 = 0
        with torch.no_grad():
            for i in range(traj_num):
                interpolated_data1 = (1 - t_values.unsqueeze(1)) * data_s1[i, :] + t_values.unsqueeze(1) * data_trans1[i, :]
                interpolated_data2 = (1 - t_values.unsqueeze(1)) * data_s2[i, :] + t_values.unsqueeze(1) * data_trans2[i, :]
                avg_points1 = avg_points1 + interpolated_data1
                avg_points2 = avg_points2 + interpolated_data2

                if plot == 'org':
                    org_z1 = inn_MET(interpolated_data1, rev=True)[0]
                    org_z2 = inn_EMT(interpolated_data2, rev=True)[0]
                    trajectory1 = pca.transform(org_z1.cpu().detach().numpy())
                    trajectories1.append(trajectory1)
                    trajectory2 = pca.transform(org_z2.cpu().detach().numpy())
                    trajectories2.append(trajectory2)
                elif plot == f'{name1}_eu':
                    org_z = interpolated_data1
                    trajectory = pca.transform(org_z.cpu().detach().numpy())
                    trajectories1.append(trajectory)
                elif plot == f'{name2}_eu':
                    org_z = interpolated_data2
                    trajectory = pca.transform(org_z.cpu().detach().numpy())
                    trajectories2.append(trajectory)
            if plot == 'org':
                avg_points1 = avg_points1 / traj_num
                avg_points2 = avg_points2 / traj_num
                org_avg1 = inn_MET(avg_points1, rev=True)[0]
                org_avg2 = inn_EMT(avg_points2, rev=True)[0]
                pca_avg1 = pca.transform(org_avg1.cpu().detach().numpy())
                pca_avg2 = pca.transform(org_avg2.cpu().detach().numpy())
                avg_list = [pca_avg1, pca_avg2]
                # colors = ['brown', 'green']
                trajectories = [trajectories1, trajectories2]
            elif plot == f'{name1}_eu':
                avg_points = avg_points1 / traj_num
                pca_avg = pca.transform(avg_points.cpu().detach().numpy())
                avg_list = [pca_avg]
                # colors = ['brown']
                trajectories = [trajectories1]
            elif plot == f'{name2}_eu':
                avg_points = avg_points2 / traj_num
                pca_avg = pca.transform(avg_points.cpu().detach().numpy())
                avg_list = [pca_avg]
                # colors = ['green']
                trajectories = [trajectories2]

            fig = new_plot_comparisions(df, trajectories, avg_list, palette=cmap, x=PCs[0], y=PCs[1], plot_avg=True, use_color=[], arrow_size=arrow_size, avg_width=avg_width)
            fig.savefig(f"{path}/trajectories_{plot}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)


def Dynamic_plot_mt(data_use, All_data, All_cluster, labels, All_t, inn, method='kde', source=1, target=5, target_list=['MkP','MasP','NeuP'], color_list=['blue', 'red', 'orange'], num_inter=300, reg=2e-2, show_velocity=10, velocity_length=3, PCs=[], path='', plot='org'):
    with torch.no_grad():
        z, _ = inn(data_use)
        z_min = z.min(dim=0, keepdim=True)[0]
        z_max = z.max(dim=0, keepdim=True)[0]
        z = scaled_output(z)

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
        target_clusters = All_cluster[target]
        uni_target_clusters = np.unique(target_clusters)
        cell_fates = []
        for row in P_normalized:
            sums = np.zeros(len(set(target_clusters)))  # 初始化类别的求和值
            # 遍历类别
            for i, cluster in enumerate(uni_target_clusters):
                # 求和该类别对应的列数据
                sums[i] = row[target_clusters == cluster].sum()
            # 找到最大和的类别
            cell_fates.append(uni_target_clusters[np.argmax(sums)])
        cell_fates = np.stack(cell_fates)
        data_s = torch.from_numpy(data_s.values).to(device)
        data_sets = []
        for cluster in target_list:
            data_trans0 = data_trans[cell_fates == cluster, :]
            data_s0 = data_s[cell_fates == cluster, :]
            if plot == 'org':
                data_trans0 = data_trans0 * (z_max - z_min + 1e-8) + z_min
                data_s0 = data_s0 * (z_max - z_min + 1e-8) + z_min
            data_sets.append([data_s0, data_trans0])

        pca = PCA(n_components=5)
        if plot == 'org':
            All_data_np = All_data.cpu().detach().numpy()
        else:
            All_data_np = z.cpu().detach().numpy()
        data_pca_all = pca.fit_transform(All_data_np)

        # 绘制散点图
        fig, ax = plt.subplots()  # 使用 plt.subplots 创建二维图
        cmap = plt.cm.plasma
        for i, color in enumerate(labels):
            # 找出当前颜色对应的标签
            current_label = labels[i]
            indices = [idx for idx, l in enumerate(All_cluster) if l == current_label]

            # 绘制二维散点图，去掉 z 轴
            ax.scatter(data_pca_all[indices, PCs[0]], data_pca_all[indices, PCs[1]], c=All_t[indices], s=5, alpha=0.1, cmap=cmap, vmin=All_t.min(), vmax=All_t.max())

        # 添加图例
        # norm = Normalize(vmin=All_t.min(), vmax=All_t.max())
        # unique_t = np.unique(All_t)
        # legend_elements = [
        #     Line2D(
        #         [0], [0], marker='o',
        #         color=cmap(norm(t)), label=f'Day {t}',
        #         markerfacecolor=cmap(norm(t)), markersize=15,
        #     )
        #     for i, t in enumerate(unique_t)
        # ]
        # leg = ax.legend(handles=legend_elements, loc='upper left')
        # ax.add_artist(leg)
        # 设置标题和标签
        # ax.set_title('PCA Embedding (2D)')
        # ax.set_xlabel('PC1')
        # ax.set_ylabel('PC2')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('grey')  # 改边框颜色
            spine.set_linewidth(0)  # 线宽可选
        t_values = torch.linspace(0.0, 1.0, num_inter+1)
        t_values = t_values[0:]
        # 打印进度的间隔
        print_interval = max(1, num_inter // 10)  # 每10%打印一次
        # 初始化两个 scatter 对象
        scatter_objects = []
        quiver_objects = []
        for i, name in enumerate(target_list):
            color = color_list[i]
            sc = ax.scatter([], [], c=color, s=5, label=f'Path {name}')
            scatter_objects.append(sc)
            # quiver 箭头对象
            quiv = ax.quiver([0] * show_velocity, [0] * show_velocity,
                             [0] * show_velocity, [0] * show_velocity,
                             angles='xy', scale_units='xy', scale=1, color='black')
            quiver_objects.append(quiv)

        def init():
            """初始化动画背景"""
            for sc, quiv in zip(scatter_objects, quiver_objects):
                sc.set_offsets(np.empty((0, 2)))  # 清空散点
                quiv.set_offsets(np.empty((0, 2)))  # 清空箭头位置
                quiv.set_UVC([0] * show_velocity, [0] * show_velocity)  # 清空箭头方向
            return scatter_objects + quiver_objects

        def update(frame):
            """更新每一帧的函数"""
            t = t_values[frame].item()
            with torch.no_grad():
                for i, (sc, quiv, data) in enumerate(zip(scatter_objects, quiver_objects, data_sets)):
                    # 计算插值数据
                    data_inter = (1 - t) * data[0] + t * data[1]
                    if plot == 'org':
                        data_inter = inn(data_inter.to(torch.float).to(device), rev=True)[0]
                    # 转换为 NumPy 数组
                    data_org_np = data_inter.cpu().detach().numpy()
                    # PCA 降维
                    data_pca_org = pca.transform(data_org_np)
                    # 更新 scatter 的数据
                    sc.set_offsets(data_pca_org[:, PCs])

                    # 计算下一时刻的速度
                    t_next = t + 0.01
                    data_inter_next = (1 - t_next) * data[0][:show_velocity, :] + t_next * data[1][:show_velocity, :]
                    if plot == 'org':
                        data_inter_next = inn(data_inter_next.to(torch.float).to(device), rev=True)[0]
                    data_org_np_next = data_inter_next.cpu().detach().numpy()
                    data_pca_org_next = pca.transform(data_org_np_next)

                    velocity = (data_pca_org_next[:, PCs] - data_pca_org[:show_velocity, PCs]) / 0.01
                    # 归一化并放大箭头
                    norms = np.linalg.norm(velocity, axis=1, keepdims=True)
                    norms[norms == 0] = 1
                    velocity_norm = (velocity / norms) * velocity_length

                    # 更新箭头
                    quiv.set_offsets(data_pca_org[:show_velocity, PCs])
                    quiv.set_UVC(velocity_norm[:, 0], velocity_norm[:, 1])


            # 打印进度
            if (frame + 1) % print_interval == 0 or frame == num_inter - 1:
                print(f'[{frame + 1}/{num_inter}]')

            return scatter_objects + quiver_objects

        ani = FuncAnimation(
            fig,
            update,
            frames=num_inter+1,
            init_func=init,
            blit=True,
            interval=100
        )
        # plt.show(block=True)  # 这行确保动画运行完毕后再执行后续代码
        ani.save(f'{path}.gif', writer='imagemagick')
        plt.close()

def inter_plot_mt(data_use, All_data, All_cluster, All_t, inn, traj_num=30, method='kde', source=1, target=5, target_list=['MkP','MasP','NeuP'], color_list=['blue', 'red', 'orange'], num_inter=70, reg=2e-2, PCs=[], plot='org', cmap='plasma', arrow_size=0.5, avg_width=5, path=''):
    with torch.no_grad():
        z, _ = inn(data_use)
        z_min = z.min(dim=0, keepdim=True)[0]
        z_max = z.max(dim=0, keepdim=True)[0]
        z = scaled_output(z)

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
        target_clusters = All_cluster[target]
        uni_target_clusters = np.unique(target_clusters)
        cell_fates = []
        for row in P_normalized:
            sums = np.zeros(len(set(target_clusters)))  # 初始化类别的求和值
            # 遍历类别
            for i, cluster in enumerate(uni_target_clusters):
                # 求和该类别对应的列数据
                sums[i] = row[target_clusters == cluster].sum()
            # 找到最大和的类别
            cell_fates.append(uni_target_clusters[np.argmax(sums)])
        cell_fates = np.stack(cell_fates)
        data_s = torch.from_numpy(data_s.values).to(device)
        data_sets = []
        for cluster in target_list:
            data_trans0 = data_trans[cell_fates == cluster, :]
            data_s0 = data_s[cell_fates == cluster, :]
            if plot == 'org':
                data_trans0 = data_trans0 * (z_max - z_min + 1e-8) + z_min
                data_s0 = data_s0 * (z_max - z_min + 1e-8) + z_min
            data_sets.append([data_s0, data_trans0])

        # 1. 对 All_data 进行 PCA 降维
        pca = PCA(n_components=5)
        if plot == 'org':
            All_data_np = All_data.cpu().detach().numpy()
        else:
            All_data_np = z.cpu().detach().numpy()

        data_pca_all = pca.fit_transform(All_data_np)

        df = pd.DataFrame(data_pca_all)
        df.insert(0, 'samples', All_t)

        t_values = torch.linspace(0.0, 1.0, num_inter+1)
        t_values = t_values[0:].to(device)
        trajectories_list = []
        avg_list = []
        with torch.no_grad():
            for data in data_sets:
                avg_points = 0
                trajectories = []
                for i in range(traj_num):
                    interpolated_data = (1 - t_values.unsqueeze(1)) * data[0][i, :] + t_values.unsqueeze(1) * data[1][i, :]
                    avg_points = avg_points + interpolated_data
                    if plot == 'org':
                        org_z = inn(interpolated_data, rev=True)[0]
                        trajectory = pca.transform(org_z.cpu().detach().numpy())
                        trajectories.append(trajectory)

                    else:
                        org_z = interpolated_data
                        trajectory = pca.transform(org_z.cpu().detach().numpy())
                        trajectories.append(trajectory)
                if plot == 'org':
                    avg_points = avg_points / traj_num
                    org_avg = inn(avg_points, rev=True)[0]
                    pca_avg = pca.transform(org_avg.cpu().detach().numpy())
                    avg_list.append(pca_avg)
                    trajectories_list.append(trajectories)
                else:
                    avg_points = avg_points / traj_num
                    pca_avg = pca.transform(avg_points.cpu().detach().numpy())
                    avg_list.append(pca_avg)
                    trajectories_list.append(trajectories)


            fig = new_plot_comparisions(df, trajectories_list, avg_list, palette=cmap, x=PCs[0], y=PCs[1], plot_avg=True, use_color=color_list, arrow_size=arrow_size, avg_width=avg_width)
            plt.savefig(f"{path}/trajectoris_{plot}.png", format='png')
            plt.close()  # 关闭当前图，释放内存

def Decide_fate_mt(data_use, All_cluster, All_t, inn, source_list, method='kde', target=5, reg = 2e-2):
    with torch.no_grad():
        target_clusters = All_cluster[target]
        uni_target_clusters = np.unique(target_clusters)
        cell_fate_list = []
        All_source_label = []
        for i in tqdm(source_list):
            source = (All_t == i)
            fate_weights = S_T_plan(data_use, inn, method=method, source=source, target=target, reg=reg)
            cell_fates = []
            for row in fate_weights:
                sums = np.zeros(len(set(target_clusters)))  # 初始化类别的求和值
                # 遍历类别
                for j, cluster in enumerate(uni_target_clusters):
                    # 求和该类别对应的列数据
                    sums[j] = row[target_clusters == cluster].sum()
                # 找到最大和的类别
                cell_fates.append(uni_target_clusters[np.argmax(sums)])
            cell_fates = np.stack(cell_fates)
            cell_fate_list.append(cell_fates)
            All_source_label.extend([i] * np.sum(source))
        return cell_fate_list, All_source_label

def Fate_prop_plot(cell_fates, target_list, color_list, name, path):
    # 统计数量
    fate_num = [(cell_fates == cluster).sum() for cluster in target_list]

    # 生成 x 坐标
    x = np.arange(len(target_list))

    # 绘图
    plt.figure(figsize=(8, 4))
    plt.bar(x, fate_num, color=color_list)
    plt.xticks(x, target_list, rotation=45)
    plt.ylabel("Cell count")
    plt.xlabel("Fate")
    plt.title('Cell Fates Proportion')
    plt.tight_layout()
    save_dir = f"{path}/virtual_interventions"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/{name}_{fate_num}.png", format='png')
    plt.close()  # 关闭当前图，释放内存


def OT_gene_Mean_Dynamic_compair_mt(data_use, t_use, inn, cell_fate, genenames, method='kde', source=1, target=5, num_inter=300, color_list=[], reg=2e-2, path='', fate_list=[]):
    with torch.no_grad():
        matplotlib.use('Agg')
        l_unique, inverse_indices = torch.unique(t_use, return_inverse=True)
        z, _ = inn(data_use)
        z_min = z.min(dim=0, keepdim=True)[0]
        z_max = z.max(dim=0, keepdim=True)[0]
        z = scaled_output(z)
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

        # data_s_list = []
        # data_trans_list = []
        t_values = torch.linspace(0.0, 1.0, num_inter + 1)
        Dynamic_exp_list_mean = []
        Dynamic_exp_list_std = []
        for fate in fate_list:
            data_trans_fate = data_trans[cell_fate == fate, :]
            data_s_fate = data_s[cell_fate == fate, :]
            data_trans_fate = data_trans_fate * (z_max - z_min + 1e-8) + z_min
            data_s_fate = data_s_fate * (z_max - z_min + 1e-8) + z_min
            # data_s_list.append(data_s_fate)
            # data_trans_list.append(data_trans_fate)

            mean_data = []
            std_data = []
            for i, t in enumerate(t_values):
                with torch.no_grad():
                    data_inter = (1 - t) * data_s_fate + t * data_trans_fate

                    data_inter_org = inn(data_inter.to(torch.float).to(device), rev=True)[0]
                    mean_inter = data_inter_org.mean(dim=0)
                    std_inter = data_inter_org.std(axis=0)

                    mean_data.append(mean_inter)
                    std_data.append(std_inter)

            mean_data = torch.stack(mean_data)
            std_data = torch.stack(std_data).detach().cpu().numpy()
            save_dir = f'{path}/Dynamic_gene_exp'
            os.makedirs(save_dir, exist_ok=True)
            # 将 tensor 转换为 NumPy 数组以便 matplotlib 使用
            data_np = mean_data.detach().cpu().numpy()
            Dynamic_exp_list_mean.append(data_np)
            Dynamic_exp_list_std.append(std_data)
            pd.DataFrame(data_np).to_csv(f"{save_dir}/{fate}.csv")

        t_target = l_unique.max().to(torch.float32)
        t_source = l_unique.min().to(torch.float32)
        t_np = (t_values * (t_target - t_source).cpu()).detach().cpu().numpy()
        data = data_use.cpu().detach().numpy()
        t = t_use.cpu().detach().numpy()
        means = np.array([data[t == lbl].mean(axis=0) for lbl in np.unique(t)])
        matplotlib.use('Agg')
        # 绘制每个特征的散点图
        for i in tqdm(range(Dynamic_exp_list_mean[0].shape[1])):
            fig = plt.figure(figsize=(6, 4))
            ax = plt.gca()
            for j in range(len(Dynamic_exp_list_mean)):
                data_np_fate = Dynamic_exp_list_mean[j]
                std_np_fate = Dynamic_exp_list_std[j]
                plt.plot(t_np, data_np_fate[:, i], alpha=0.6, linestyle='-', color=color_list[j], linewidth=3)
                ax.fill_between(t_np,
                                data_np_fate[:, i] - std_np_fate[:, i],
                                data_np_fate[:, i] + std_np_fate[:, i],
                                color=color_list[j], alpha=0.2)

            gene_name = genenames[i]
            sanitized_gene_name = gene_name.replace('/', '_').replace('\\', '_')
            plt.scatter(np.unique(t) - t.min(), means[:, i], alpha=0.6, label='True Data', marker='o', s=40,
                        color='red')
            plt.title(sanitized_gene_name)  # 设置图标题为特征名
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
            # plt.xlabel("t")
            # plt.ylabel(f"{sanitized_gene_name} expression")
            plt.grid(False)
            for spine in ax.spines.values():
                spine.set_edgecolor('grey')  # 改边框颜色
                spine.set_linewidth(1)  # 线宽可选
            # plt.legend()  # 显示图例
            # 保存图片，文件名为特征名
            plt.savefig(f"{save_dir}/{sanitized_gene_name}.png", format='png')
            plt.close(fig)  # 关闭当前图，释放内存

def Cell_Navigation(All_data, All_t, inn, cell_fates, navi_org, navi_target, HVG, source, target, PCs, method='kde', reg=2e-2, num_inter=200, path=''):
    with torch.no_grad():
        matplotlib.use('Agg')
        z, _ = inn(All_data)
        z_min = z.min(dim=0, keepdim=True)[0]
        z_max = z.max(dim=0, keepdim=True)[0]
        z = scaled_output(z)
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

        data_trans_org = data_trans[cell_fates == navi_org, :]
        data_s_org = data_s[cell_fates == navi_org, :]
        data_trans_target = data_trans[cell_fates == navi_target, :]
        data_s_target = data_s[cell_fates == navi_target, :]

        mean_s_org_eu = data_s_org.mean(0).unsqueeze(1)
        mean_s_target_eu = data_s_target.mean(0).unsqueeze(1)
        mean_trans_org_eu = data_trans_org.mean(0).unsqueeze(1)
        mean_trans_target_eu = data_trans_target.mean(0).unsqueeze(1)

        data_trans_org_eu = data_trans_org * (z_max - z_min + 1e-8) + z_min
        data_s_org_eu = data_s_org * (z_max - z_min + 1e-8) + z_min
        data_trans_target_eu = data_trans_target * (z_max - z_min + 1e-8) + z_min
        data_s_target_eu = data_s_target * (z_max - z_min + 1e-8) + z_min

        t_values = torch.linspace(0.0, 1.0, num_inter + 1)
        t_values = t_values[0:].to(device)

        traj_org_org = []
        std_org_org = []
        for i, t in enumerate(t_values):
            with torch.no_grad():
                data_inter = (1 - t) * data_s_org_eu + t * data_trans_org_eu

                data_inter_org = inn(data_inter.to(torch.float).to(device), rev=True)[0]
                mean_inter = data_inter_org.mean(dim=0)
                std_inter = data_inter_org.std(axis=0)

                traj_org_org.append(mean_inter)
                std_org_org.append(std_inter)
        traj_org_org = torch.stack(traj_org_org)
        std_org_org = torch.stack(std_org_org)

        traj_target_target = []
        std_target_target = []
        for i, t in enumerate(t_values):
            with torch.no_grad():
                data_inter = (1 - t) * data_s_target_eu + t * data_trans_target_eu

                data_inter_org = inn(data_inter.to(torch.float).to(device), rev=True)[0]
                mean_inter = data_inter_org.mean(dim=0)
                std_inter = data_inter_org.std(axis=0)

                traj_target_target.append(mean_inter)
                std_target_target.append(std_inter)
        traj_target_target = torch.stack(traj_target_target)
        std_target_target = torch.stack(std_target_target)

        traj_org_target = []
        std_org_target = []
        for i, t in enumerate(t_values):
            with torch.no_grad():
                data_inter = (1 - t) * data_s_org_eu + t * data_trans_target_eu.mean(0)

                data_inter_org = inn(data_inter.to(torch.float).to(device), rev=True)[0]
                mean_inter = data_inter_org.mean(dim=0)
                std_inter = data_inter_org.std(axis=0)

                traj_org_target.append(mean_inter)
                std_org_target.append(std_inter)
        traj_org_target = torch.stack(traj_org_target)
        std_org_target = torch.stack(std_org_target)

        traj_org_org_eu = ((1 - t_values) * mean_s_org_eu + t_values * mean_trans_org_eu).T
        traj_target_target_eu = ((1 - t_values) * mean_s_target_eu + t_values * mean_trans_target_eu).T
        traj_org_target_eu = ((1 - t_values) * mean_s_org_eu + t_values * mean_trans_target_eu).T

        unique_labels = np.unique(All_t)
        pca = PCA(n_components=50)
        z = z.cpu().detach().numpy()
        data_pca_all = pca.fit_transform(z)
        cmap = plt.cm.plasma
        fig, ax = plt.subplots()  # 使用 plt.subplots 创建二维图
        for i, color in enumerate(unique_labels):
            # 找出当前颜色对应的标签
            current_label = unique_labels[i]
            indices = [idx for idx, l in enumerate(All_t) if l == current_label]

            # 绘制二维散点图，去掉 z 轴
            ax.scatter(data_pca_all[indices, PCs[0]], data_pca_all[indices, PCs[1]], c=All_t[indices], s=5, alpha=0.1,
                       cmap=cmap, vmin=unique_labels.min(), vmax=unique_labels.max())
        # 添加图例
        ax.legend()
        # 设置标题和标签
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('grey')  # 改边框颜色
            spine.set_linewidth(1)  # 线宽可选

        pca_eu_org_org = pca.transform(traj_org_org_eu.cpu().detach().numpy())
        pca_eu_org_target = pca.transform(traj_org_target_eu.cpu().detach().numpy())
        pca_eu_target_target = pca.transform(traj_target_target_eu.cpu().detach().numpy())

        j = len(pca_eu_org_org) - 2  # 倒数第二个点（作为箭头起点）
        plt.plot(pca_eu_org_org[:, PCs[0]], pca_eu_org_org[:, PCs[1]], alpha=1, linestyle='-', color='green', linewidth=3,
                 label=f'{navi_org} Original Path')
        dx = pca_eu_org_org[-1, PCs[0]] - pca_eu_org_org[j, PCs[0]]
        dy = pca_eu_org_org[-1, PCs[1]] - pca_eu_org_org[j, PCs[1]]
        plt.arrow(pca_eu_org_org[j, 0], pca_eu_org_org[j, 1], dx, dy,
                  shape='full', head_width=0.1, color='green',
                  length_includes_head=True)

        plt.plot(pca_eu_org_target[:, PCs[0]], pca_eu_org_target[:, PCs[1]], alpha=1, linestyle='-', color='orange', linewidth=3,
                 label=f'{navi_target} Navigation Compass')
        dx = pca_eu_org_target[-1, PCs[0]] - pca_eu_org_target[j, PCs[0]]
        dy = pca_eu_org_target[-1, PCs[1]] - pca_eu_org_target[j, PCs[1]]
        plt.arrow(pca_eu_org_target[j, PCs[0]], pca_eu_org_target[j, PCs[1]], dx, dy,
                  shape='full', head_width=0.1, color='orange',
                  length_includes_head=True)

        plt.plot(pca_eu_target_target[:, PCs[0]], pca_eu_target_target[:, PCs[1]], alpha=1, linestyle='-', color='red', linewidth=3,
                 label=f'{navi_target} Original Path')
        dx = pca_eu_target_target[-1, PCs[0]] - pca_eu_target_target[j, PCs[0]]
        dy = pca_eu_target_target[-1, PCs[1]] - pca_eu_target_target[j, PCs[1]]
        plt.arrow(pca_eu_target_target[j, PCs[0]], pca_eu_target_target[j, PCs[1]], dx, dy,
                  shape='full', head_width=0.1, color='red',
                  length_includes_head=True)
        # 保存图形
        # plt.legend()
        plt.savefig(f"{path}/{navi_org}_{navi_target}_eu_Navigation.png", format='png')
        plt.close(fig)  # 关闭当前图，释放内存

        All_data_np = All_data.cpu().detach().numpy()
        data_pca_all = pca.fit_transform(All_data_np)
        cmap = plt.cm.plasma
        fig, ax = plt.subplots()  # 使用 plt.subplots 创建二维图
        for i, color in enumerate(unique_labels):
            # 找出当前颜色对应的标签
            current_label = unique_labels[i]
            indices = [idx for idx, l in enumerate(All_t) if l == current_label]

            # 绘制二维散点图，去掉 z 轴
            ax.scatter(data_pca_all[indices, PCs[0]], data_pca_all[indices, PCs[1]], c=All_t[indices], s=5, alpha=0.1,
                       cmap=cmap, vmin=unique_labels.min(), vmax=unique_labels.max())
        # 添加图例
        ax.legend()
        # 设置标题和标签
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('grey')  # 改边框颜色
            spine.set_linewidth(1)  # 线宽可选

        pca_org_org = pca.transform(traj_org_org.cpu().detach().numpy())
        pca_org_target = pca.transform(traj_org_target.cpu().detach().numpy())
        pca_target_target = pca.transform(traj_target_target.cpu().detach().numpy())

        j = len(pca_org_org) - 2  # 倒数第二个点（作为箭头起点）
        plt.plot(pca_org_org[:, PCs[0]], pca_org_org[:, PCs[1]], alpha=1, linestyle='-', color='green', linewidth=3,
                 label=f'{navi_org} Original Path')
        dx = pca_org_org[-1, PCs[0]] - pca_org_org[j, PCs[0]]
        dy = pca_org_org[-1, PCs[1]] - pca_org_org[j, PCs[1]]
        plt.arrow(pca_org_org[j, PCs[0]], pca_org_org[j, PCs[1]], dx, dy,
                  shape='full', head_width=2, color='green',
                  length_includes_head=True)

        plt.plot(pca_org_target[:, PCs[0]], pca_org_target[:, PCs[1]], alpha=1, linestyle='-', color='orange', linewidth=3,
                 label=f'{navi_target} Navigation Compass')
        dx = pca_org_target[-1, PCs[0]] - pca_org_target[j, PCs[0]]
        dy = pca_org_target[-1, PCs[1]] - pca_org_target[j, PCs[1]]
        plt.arrow(pca_org_target[j, PCs[0]], pca_org_target[j, PCs[1]], dx, dy,
                  shape='full', head_width=2, color='orange',
                  length_includes_head=True)

        plt.plot(pca_target_target[:, PCs[0]], pca_target_target[:, PCs[1]], alpha=1, linestyle='-', color='red', linewidth=3,
                 label=f'{navi_target} Original Path')
        dx = pca_target_target[-1, PCs[0]] - pca_target_target[j, PCs[0]]
        dy = pca_target_target[-1, PCs[1]] - pca_target_target[j, PCs[1]]
        plt.arrow(pca_target_target[j, PCs[0]], pca_target_target[j, PCs[1]], dx, dy,
                  shape='full', head_width=2, color='red',
                  length_includes_head=True)
        # 保存图形
        # plt.legend()
        plt.savefig(f"{path}/{navi_org}_{navi_target}_org_Navigation.png", format='png')
        plt.close(fig)  # 关闭当前图，释放内存

    t_np = t_values.cpu().numpy()
    save_dir = f"{path}/gene_dynamic"
    os.makedirs(save_dir, exist_ok=True)
    # 绘制每个特征的散点图
    for i in tqdm(range(traj_org_org.shape[1])):
        fig = plt.figure(figsize=(6, 4))
        ax = plt.gca()
        # 绘制 filtered_data_np1，标签为 "MET"
        plt.scatter(t_np, traj_org_org[:, i].cpu().detach().numpy(), alpha=1, label=f'{navi_org} Original Path', marker='o', s=20, color='green')
        plt.scatter(t_np, traj_org_target[:, i].cpu().detach().numpy(), alpha=1, label=f'{navi_target} Navigation Compass', marker='o', s=20, color='orange')
        plt.scatter(t_np, traj_target_target[:, i].cpu().detach().numpy(), alpha=1, label=f'{navi_target} Original Path', marker='o', s=20, color='red')

        gene_name = HVG[i]
        sanitized_gene_name = gene_name.replace('/', '_').replace('\\', '_')

        # plt.legend()  # 显示图例
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        plt.grid(False)
        for spine in ax.spines.values():
            spine.set_edgecolor('grey')  # 改边框颜色
            spine.set_linewidth(1)  # 线宽可选

        # 保存图片，文件名为特征名
        plt.savefig(f"{save_dir}/{sanitized_gene_name}.png", format='png')
        plt.close(fig)  # 关闭当前图，释放内存












































