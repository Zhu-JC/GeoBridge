import ot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import Get_Probability_Measures
from scipy.spatial.distance import cdist
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
import matplotlib
import os

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

def get_data_trans(data_use, t_use, inn, method = 'kde', num_inter = 300, reg = 1e-2):
    z, _ = inn(data_use)
    z = scaled_output(z)

    data_t = z[t_use == t_use.max()]
    data_t = pd.DataFrame(data_t.cpu().detach().numpy())
    data_s = z[t_use == t_use.min()]
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

    return data_s, data_trans


def Dynamic_plot(data_use, inn_true, inn_pseudo, True_t, Pseudo_t, All_data, All_label, labels, method='kde', num_inter=300, show_velocity=10, reg1=2e-2, reg2=2e-2, PCs=[], path='', name1='', name2=''):
    with torch.no_grad():
        # 1. 对 All_data 进行 PCA 降维
        pca = PCA(n_components=2)
        data_pca_all = pca.fit_transform(All_data)

        # 绘制散点图
        fig, ax = plt.subplots()  # 使用 plt.subplots 创建二维图
        cmap = plt.cm.plasma
        for i, color in enumerate(labels):
            # 找出当前颜色对应的标签
            current_label = labels[i]
            indices = [idx for idx, l in enumerate(All_label) if l == current_label]

            # 绘制二维散点图，去掉 z 轴
            ax.scatter(data_pca_all[indices, PCs[0]], data_pca_all[indices, PCs[1]], c=True_t[indices], s=5, alpha=0.1, cmap=cmap, vmin=True_t.min(), vmax=True_t.max())
        # 添加图例
        ax.legend()
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('grey')  # 改边框颜色
            spine.set_linewidth(0)  # 线宽可选
        data_use = torch.from_numpy(data_use).to(device)
        z, _ = inn_true(data_use)
        z1_min = z.min(dim=0, keepdim=True)[0]
        z1_max = z.max(dim=0, keepdim=True)[0]
        z2, _ = inn_pseudo(data_use)
        z2_min = z2.min(dim=0, keepdim=True)[0]
        z2_max = z2.max(dim=0, keepdim=True)[0]

        data_s1, data_trans1 = get_data_trans(data_use, True_t, inn_true, method=method, reg=reg1)
        data_s2, data_trans2 = get_data_trans(data_use, Pseudo_t, inn_pseudo, method=method, reg=reg2)

        t_values = torch.linspace(0.0, 1.0, num_inter+1)
        t_values = t_values[1:]
        # 初始化两个 scatter 对象
        sc1 = ax.scatter([], [], c='green', s=1, label=f'Path {name1}')
        sc2 = ax.scatter([], [], c='brown', s=1, label=f'Path {name2}')
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

                data_inter1 = data_inter1 * (z1_max - z1_min + 1e-8) + z1_min
                data_inter2 = data_inter2 * (z2_max - z2_min + 1e-8) + z2_min
                data_inter1 = inn_true(data_inter1.to(torch.float).to(device), rev=True)[0]
                data_inter2 = inn_pseudo(data_inter2.to(torch.float).to(device), rev=True)[0]

                # 转换为 NumPy 数组
                data_org_np1 = data_inter1.cpu().detach().numpy()
                data_org_np2 = data_inter2.cpu().detach().numpy()

                # PCA 降维
                data_pca_org1 = pca.transform(data_org_np1)
                data_pca_org2 = pca.transform(data_org_np2)

                # 更新 scatter 的数据
                sc1.set_offsets(data_pca_org1[:, :2])
                sc2.set_offsets(data_pca_org2[:, :2])
                t_next = t + 0.01
                data_inter1_next = (1 - t_next) * data_s1[0:show_velocity, :] + t_next * data_trans1[0:show_velocity, :]
                data_inter2_next = (1 - t_next) * data_s2[0:show_velocity, :] + t_next * data_trans2[0:show_velocity, :]
                data_inter1_next = data_inter1_next * (z1_max - z1_min + 1e-8) + z1_min
                data_inter2_next = data_inter2_next * (z2_max - z2_min + 1e-8) + z2_min
                data_inter1_next = inn_true(data_inter1_next.to(torch.float).to(device), rev=True)[0]
                data_inter2_next = inn_pseudo(data_inter2_next.to(torch.float).to(device), rev=True)[0]
                data_org_np1_next = data_inter1_next.cpu().detach().numpy()
                data_org_np2_next = data_inter2_next.cpu().detach().numpy()
                data_pca_org1_next = pca.transform(data_org_np1_next)
                data_pca_org2_next = pca.transform(data_org_np2_next)
                velocity1 = (data_pca_org1_next[:, PCs] - data_pca_org1[0:show_velocity, PCs]) / 0.01
                velocity2 = (data_pca_org2_next[:, PCs] - data_pca_org2[0:show_velocity, PCs]) / 0.01
                norms1 = np.linalg.norm(velocity1, axis=1, keepdims=True)  # 每个向量长度
                norms1[norms1 == 0] = 1  # 防止除以零
                velocity1_norm = (velocity1 / norms1) * 3
                norms2 = np.linalg.norm(velocity2, axis=1, keepdims=True)
                norms2[norms2 == 0] = 1
                velocity2_norm = (velocity2 / norms2) * 3
                # 更新箭头（quiver）
                quiv1.set_offsets(data_pca_org1[:show_velocity, PCs])
                quiv1.set_UVC(velocity1_norm[:, 0], velocity1_norm[:, 1])

                quiv2.set_offsets(data_pca_org2[:show_velocity, PCs])
                quiv2.set_UVC(velocity2_norm[:, 0], velocity2_norm[:, 1])

            return sc1, sc2

        ani = FuncAnimation(
            fig,
            update,
            frames=num_inter,
            init_func=init,
            blit=True,
            interval=100
        )
        plt.tight_layout()
        ani.save(f'{path}/Org_compare.gif', writer='imagemagick')
        plt.close()


def Dynamic_path_plot(data_use, All_data, All_label, labels, True_t, Pseudo_t, inn_true, inn_pseudo, method='kde', num_inter=300, reg1=2e-2, reg2=2e-2, PCs=[], linewidth=5, arrow_size=1.5, path=''):
    with torch.no_grad():
        data_use = torch.from_numpy(data_use).to(device)
        z, _ = inn_true(data_use)
        z1_min = z.min(dim=0, keepdim=True)[0]
        z1_max = z.max(dim=0, keepdim=True)[0]
        z, _ = inn_pseudo(data_use)
        z2_min = z.min(dim=0, keepdim=True)[0]
        z2_max = z.max(dim=0, keepdim=True)[0]

        data_s1, data_trans1 = get_data_trans(data_use, True_t, inn_true, method=method, reg=reg1)
        data_s2, data_trans2 = get_data_trans(data_use, Pseudo_t, inn_pseudo, method=method, reg=reg2)

        data_trans1 = data_trans1 * (z1_max - z1_min + 1e-8) + z1_min
        data_s1 = data_s1 * (z1_max - z1_min + 1e-8) + z1_min
        data_trans2 = data_trans2 * (z2_max - z2_min + 1e-8) + z2_min
        data_s2 = data_s2 * (z2_max - z2_min + 1e-8) + z2_min

        pca = PCA(n_components=5)
        data_pca_all = pca.fit_transform(All_data)

        # 绘制散点图

        fig, ax = plt.subplots()  # 使用 plt.subplots 创建二维图
        cmap = plt.cm.plasma
        for i, color in enumerate(labels):
            # 找出当前颜色对应的标签
            current_label = labels[i]
            indices = [idx for idx, l in enumerate(All_label) if l == current_label]

            # 绘制二维散点图，去掉 z 轴
            ax.scatter(data_pca_all[indices, PCs[0]], data_pca_all[indices, PCs[1]], c=True_t[indices], s=5, alpha=0.1, cmap=cmap, vmin=True_t.min(), vmax=True_t.max())

        # 设置标题和标签
        ax.legend()
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('grey')  # 改边框颜色
            spine.set_linewidth(0)  # 线宽可选

        t_values = torch.linspace(0.0, 1.0, num_inter+1)
        mean_data1 = []
        mean_data2 = []
        for i, t in enumerate(t_values):
            with torch.no_grad():
                data_inter1 = (1 - t) * data_s1 + t * data_trans1
                data_inter2 = (1 - t) * data_s2 + t * data_trans2

                data_inter_org1 = inn_true(data_inter1.to(torch.float).to(device), rev=True)[0]
                mean_inter1 = data_inter_org1.mean(dim=0)
                data_inter_org2 = inn_pseudo(data_inter2.to(torch.float).to(device), rev=True)[0]
                mean_inter2 = data_inter_org2.mean(dim=0)

                mean_data1.append(mean_inter1)
                mean_data2.append(mean_inter2)

        mean_data1 = torch.stack(mean_data1)
        mean_data2 = torch.stack(mean_data2)

        # 2. 使用相同的 PCA 模型对 data_org 降维
        data_np1 = mean_data1.cpu().detach().numpy()
        data_pca1 = pca.transform(data_np1)
        data_np2 = mean_data2.cpu().detach().numpy()
        data_pca2 = pca.transform(data_np2)

        j = data_pca1.shape[0] - 2  # 倒数第二个点（作为箭头起点）
        plt.plot(data_pca1[:, PCs[0]], data_pca1[:, PCs[1]], linestyle='--', color='green', linewidth=linewidth)
        dx = data_pca1[-1, PCs[0]] - data_pca1[j, PCs[0]]
        dy = data_pca1[-1, PCs[1]] - data_pca1[j, PCs[1]]
        plt.arrow(data_pca1[j, PCs[0]], data_pca1[j, PCs[1]], dx, dy,
                  shape='full', head_width=arrow_size, color='green',
                  length_includes_head=True)
        plt.plot(data_pca2[:, PCs[0]], data_pca2[:, PCs[1]], linestyle='--', color='brown', linewidth=linewidth)
        dx = data_pca2[-1, PCs[0]] - data_pca2[j, PCs[0]]
        dy = data_pca2[-1, PCs[1]] - data_pca2[j, PCs[1]]
        plt.arrow(data_pca2[j, PCs[0]], data_pca2[j, PCs[1]], dx, dy,
                  shape='full', head_width=arrow_size, color='brown',
                  length_includes_head=True)
        fig.savefig(f"{path}/pseudo_true_path_compair.png", dpi=300, bbox_inches="tight")
        plt.close()




def OT_gene_Mean_Dynamic_compair(data_use, True_t, Pseudo_t, inn_true, inn_pseudo, genenames, method = 'kde', num_inter = 300, use = 'org', reg1=1e-2, reg2=2e-2, path = '', name1 = '', name2 = ''):
    with torch.no_grad():
        data_use = torch.from_numpy(data_use).to(device)
        data_s1, data_trans1 = get_data_trans(data_use, True_t, inn_true, method=method, reg=reg1)
        data_s2, data_trans2 = get_data_trans(data_use, Pseudo_t, inn_pseudo, method=method, reg=reg2)

        z, _ = inn_true(data_use)
        z_min = z.min(dim=0, keepdim=True)[0]
        z_max = z.max(dim=0, keepdim=True)[0]
        data_trans1 = data_trans1 * (z_max - z_min + 1e-8) + z_min
        data_s1 = data_s1 * (z_max - z_min + 1e-8) + z_min

        z, _ = inn_pseudo(data_use)
        z_min = z.min(dim=0, keepdim=True)[0]
        z_max = z.max(dim=0, keepdim=True)[0]
        data_trans2 = data_trans2 * (z_max - z_min + 1e-8) + z_min
        data_s2 = data_s2 * (z_max - z_min + 1e-8) + z_min

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
                    data_inter_org1 = inn_true(data_inter1.to(torch.float).to(device), rev=True)[0]
                    mean_inter1 = data_inter_org1.mean(dim=0)
                    std_inter1 = data_inter_org1.std(axis=0)
                    data_inter_org2 = inn_pseudo(data_inter2.to(torch.float).to(device), rev=True)[0]
                    mean_inter2 = data_inter_org2.mean(dim=0)
                    std_inter2 = data_inter_org2.std(axis=0)
                else:
                    mean_inter1 = data_inter1.mean(dim=0).unsqueeze(0)
                    mean_inter1 = inn_true(mean_inter1.to(torch.float).to(device), rev=True)[0][0]
                    mean_inter2 = data_inter2.mean(dim=0).unsqueeze(0)
                    mean_inter2 = inn_pseudo(mean_inter2.to(torch.float).to(device), rev=True)[0][0]
                mean_data1.append(mean_inter1)
                mean_data2.append(mean_inter2)
                std_data1.append(std_inter1)
                std_data2.append(std_inter2)

        mean_data1 = torch.stack(mean_data1)
        mean_data2 = torch.stack(mean_data2)
        std_data1 = torch.stack(std_data1).detach().cpu().numpy()
        std_data2 = torch.stack(std_data2).detach().cpu().numpy()
        t_np = t_values.cpu().numpy() * (num_inter/10)
        os.makedirs(path, exist_ok=True)
        # 将 tensor 转换为 NumPy 数组以便 matplotlib 使用
        filtered_data_np1 = mean_data1.detach().cpu().numpy()
        filtered_data_np2 = mean_data2.detach().cpu().numpy()
        # filtered_data_np1[filtered_data_np1 < 0] = 0
        pd.DataFrame(filtered_data_np1).to_csv(f"{path}/{name1}.csv")
        # filtered_data_np2[filtered_data_np2 < 0] = 0
        pd.DataFrame(filtered_data_np2).to_csv(f"{path}/{name2}.csv")
        matplotlib.use('Agg')
        # 绘制每个特征的散点图
        for i in range(filtered_data_np1.shape[1]):
            print(i)
            fig = plt.figure(figsize=(6, 4))
            ax = plt.gca()
            # 绘制 filtered_data_np1，标签为 "MET"
            plt.scatter(t_np, filtered_data_np1[:, i], alpha=0.6, label=name1, marker='o', s=20, color='green')
            ax.fill_between(t_np,
                            filtered_data_np1[:, i] - std_data1[:, i],
                            filtered_data_np1[:, i] + std_data1[:, i],
                            color='green', alpha=0.2)
            gene_name = genenames[i]
            sanitized_gene_name = gene_name.replace('/', '_').replace('\\', '_')
            # 绘制 filtered_data_np2，标签为 "EMT"
            plt.scatter(t_np, filtered_data_np2[:, i], alpha=0.6, label=name2, marker='o', s=20, color='orange')
            ax.fill_between(t_np,
                            filtered_data_np2[:, i] - std_data2[:, i],
                            filtered_data_np2[:, i] + std_data2[:, i],
                            color='orange', alpha=0.2)
            cor = np.corrcoef(filtered_data_np1[:, i], filtered_data_np2[:, i])[0, 1]
            plt.title(f'{sanitized_gene_name}  cor: {cor}', fontsize=20)  # 设置图标题为特征名
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

def gene_cor_distribution(path, name1, name2):
    True_data = pd.read_csv(f'{path}/pair_gene/{name1}.csv', header=0, index_col=0, engine='python')
    Pseudo_data = pd.read_csv(f'{path}/pair_gene/{name2}.csv', header=0, index_col=0, engine='python')

    # --- 计算每列的相关性 ---
    correlation_coefficients = []
    for col_name in True_data.columns:
        # 获取对应列的 Series
        true_col = True_data[col_name]
        pseudo_col = Pseudo_data[col_name]

        # 计算相关性系数
        corr_value = true_col.corr(pseudo_col)

        # 将相关性系数添加到列表中
        correlation_coefficients.append(corr_value)

    # --- 绘制概率密度图 (KDE) ---
    plt.figure(figsize=(10, 6))  # 设置图的大小

    # 使用 seaborn 绘制 KDE 图
    sns.kdeplot(data=correlation_coefficients, fill=True)

    # 添加图的标题和标签
    plt.title(f'Distribution of Column Correlation Coefficients\n({name1} vs {name2})')
    plt.xlabel('Pearson Correlation Coefficient')
    plt.ylabel('Density')

    # 设置 X 轴范围为 -1 到 1，因为相关性系数总是介于这两个值之间
    plt.xlim(-1, 1)

    # 添加网格线 (可选)
    plt.grid(True, linestyle='--', alpha=0.6)
    # 显示图
    plt.savefig(f"{path}/gene_cor_distribution.png", format='png')
    plt.close()  # 关闭当前图，释放内存





































































