import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
import ot
from downstream_analysis.Get_Probability_Measures import kde_gene_expression
from sklearn.decomposition import PCA
import seaborn as sns
from utils import scaled_output
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def new_plot_comparisions(
        df, points, true_data,
        palette='plasma',
        df_time_key='samples',
        x=0, y=1,
        groups=None
):
    if groups is None:
        groups = sorted(df[df_time_key].unique())
    cmap = plt.cm.plasma
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

        n = 0.3
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
        ax.scatter(
            true_data[x], true_data[y],
            c=true_data[df_time_key],
            s=size,
            alpha=1,
            marker='X',
            linewidths=0,
            edgecolors=None,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )
        # for trajectory in np.transpose(trajectories, axes=(1, 0, 2)):
        #     plt.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.3, color='Black');

        colors = points[df_time_key]
        n = 1
        o = '.'
        ax.scatter(
            points[x], points[y],
            c='black',
            s=bg_size,
            alpha=1 * n,
            marker=o,
            linewidths=0,
            edgecolors=None
        )
        ax.scatter(
            points[x], points[y],
            c='white',
            s=gap_size,
            alpha=1 * n,
            marker=o,
            linewidths=0,
            edgecolors=None
        )
        pnts = ax.scatter(
            points[x], points[y],
            c='grey',
            s=size,
            alpha=0.7 * n,
            marker=o,
            linewidths=0,
            edgecolors=None,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        plt.grid(False)
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


def inter_plot(data_use, t_use, inn, heldout=30, plot='org', reg=3e-2, use_kde=True, use_pca=True, output_dir=''):
    """
    绘制并比较插值数据与真实数据。
    根据 'inter' 参数（heldout时间点在l_unique中的1-based位置）
    动态确定源时间点和目标时间点进行插值。
    """
    l_unique, inverse_indices = torch.unique(t_use, return_inverse=True)
    inter = torch.where(l_unique == heldout)[0].item()
    n_unique = len(l_unique)

    # 根据规则确定 source_idx 和 target_idx
    if inter == 0:  # heldout 是第一个时间点
        source = inter + 1
        target = inter + 2
    elif inter == n_unique - 1:  # heldout 是最后一个时间点
        source = inter - 2
        target = inter - 1
    else:  # heldout 是中间的时间点
        source = inter - 1
        target = inter + 1

    z, _ = inn(data_use)
    z_min = z.min(dim=0, keepdim=True)[0]
    z_max = z.max(dim=0, keepdim=True)[0]
    z = scaled_output(z)
    data_t = z[inverse_indices == target]
    data_t = pd.DataFrame(data_t.cpu().detach().numpy())
    data_s = z[inverse_indices == source]
    data_s = pd.DataFrame(data_s.cpu().detach().numpy())
    C = cdist(data_s.values, data_t.values, metric='euclidean')

    mu = kde_gene_expression(data_s)
    nu = kde_gene_expression(data_t)

    P = ot.sinkhorn(mu, nu, C, reg=reg)
    P_normalized = P / P.sum(axis=1, keepdims=True)
    data_trans = np.dot(P_normalized, data_t.values)
    data_trans = torch.from_numpy(data_trans).to(device).to(torch.float32)
    data_s = torch.from_numpy(data_s.values).to(device)
    # 1. 对 All_data 进行 PCA 降维
    pca = PCA(n_components=50)
    if plot == 'org':
        All_data_np = data_use.cpu().detach().numpy()
    else:
        with torch.no_grad():
            z_all, _ = inn(data_use)
            z_all = (z_all - z_min) / (z_max - z_min + 1e-8)
        All_data_np = z_all.cpu().detach().numpy()

    if not use_pca:
        All_data_np = pca.fit_transform(All_data_np)
    df = pd.DataFrame(All_data_np[(t_use != l_unique[inter]).cpu().detach().numpy()])
    df.insert(0, 'samples', t_use.cpu().detach().numpy()[(t_use != l_unique[inter]).cpu().detach().numpy()])
    if plot == 'org':
        data_trans = data_trans * (z_max - z_min + 1e-8) + z_min
        data_s = data_s * (z_max - z_min + 1e-8) + z_min

    with torch.no_grad():
        t = (heldout - l_unique[source]) / (l_unique[target] - l_unique[source])
        data_inter = (1 - t) * data_s + t * data_trans
        if plot == 'org':
            data_inter = inn(data_inter, rev=True)[0].cpu().detach().numpy()
        data_true = All_data_np[(t_use == l_unique[inter]).cpu().detach().numpy()]
        if not use_pca:
            data_inter = pca.transform(data_inter)

        data_true_df = pd.DataFrame(data_inter)
        data_inter_df = pd.DataFrame(data_true)
        C = torch.cdist(torch.from_numpy(data_inter).to(device), torch.from_numpy(data_true).to(device), p=2).cpu().numpy()
        if use_kde:
            mu = kde_gene_expression(data_inter_df)
            nu = kde_gene_expression(data_true_df)
        else:
            ##使用均匀分布
            mu = np.ones((data_inter.shape[0],), dtype=np.float64) / data_inter.shape[0]
            nu = np.ones((data_true.shape[0],), dtype=np.float64) / data_true.shape[0]
        WD = ot.emd2(mu, nu, C, numItermax=1e7)

        data_inter_df.insert(0, 'samples', heldout)
        data_true_df.insert(0, 'samples', heldout)
        fig = new_plot_comparisions(df, data_inter_df, data_true_df)
        # 更新 scatter 的数据
        fig.suptitle(
            f'Heldout T{heldout} WD: {WD:.5f}',  # 这是你的标题文本
            fontsize=16,  # 可选：设置字体大小
            fontweight='bold',  # 可选：设置字体粗细 ('normal', 'bold', etc.)
            y=0.98  # 可选：微调标题的垂直位置 (1.0是顶部, 0.98稍微向下一点)
        )
        fig.savefig(f'{output_dir}/{heldout}_{plot}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)



