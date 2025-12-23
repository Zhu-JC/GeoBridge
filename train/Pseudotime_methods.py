import scanpy as sc
from sklearn.cluster import KMeans
import numpy as np
import omicverse as ov
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import anndata

def pseudotime_plot(All_data, pseudotime, PCs=[0, 1]):
    pca = PCA(n_components=5)
    All_data_np = All_data
    data_pca_all = pca.fit_transform(All_data_np)

    # 绘制散点图

    discrete_cmap = plt.cm.viridis  # You can choose other colormaps, e.g., 'plasma', 'inferno', 'Reds'

    fig, ax = plt.subplots(figsize=(8, 6))
    weights = pseudotime

    scatter = ax.scatter(
        data_pca_all[:, PCs[0]],
        data_pca_all[:, PCs[1]],
        c=weights,  # Use the weights array for coloring
        cmap=discrete_cmap,  # Specify the colormap
        s=5
    )
    ax.legend(loc='best')  # Adjust legend location as needed

    # Set title and labels
    ax.set_title(f'PCA Embedding (2D) Colored by Weight')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    cbar = fig.colorbar(scatter, ax=ax)
    plt.tight_layout()
    return fig


#基于流形学习的轨迹推断
def get_slingshot_pseudotime(data_path, true_t, cluster, path):
    adata = anndata.read_h5ad(data_path)
    sc.pp.pca(adata)
    All_data = adata.X
    if cluster:
        adata.obs['clusters'] = adata.obs[cluster]
    else:
        num_clusters = 4
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
        kmeans.fit(All_data)
        adata.obs['clusters'] = kmeans.labels_
    adata.obs['clusters'] = adata.obs['clusters'].astype(str).astype('category')
    Traj = ov.single.TrajInfer(adata, basis='X_pca', use_rep='X_pca', n_comps=50)
    Traj.set_origin_cells(adata.obs['clusters'][0])
    Traj.set_terminal_cells([adata.obs['clusters'][-1]])
    Traj.inference(method='slingshot', num_epochs=1)

    pseudotime = np.array(adata.obs['slingshot_pseudotime'])
    cor = np.corrcoef(pseudotime, true_t)[0, 1]
    fig = pseudotime_plot(All_data, pseudotime, PCs=[0, 1])
    plt.savefig(f"{path}/slingshot.png", format='png')
    plt.close(fig)
    return cor

# 基于PAGA的扩散模型
def get_PAGA_pseudotime(data_path, true_t, cluster, path):
    adata = anndata.read_h5ad(data_path)
    sc.pp.pca(adata)
    All_data = adata.X
    if cluster:
        adata.obs['clusters'] = adata.obs[cluster]
    else:
        num_clusters = 5
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
        kmeans.fit(All_data)
        adata.obs['clusters'] = kmeans.labels_
    Traj = ov.single.TrajInfer(adata, basis='X_pca', use_rep='X_pca', n_comps=50)
    Traj.set_origin_cells(adata.obs['clusters'][0])
    Traj.inference(method='diffusion_map')
    adata.obs['dpt_pseudotime']=adata.obs['dpt_pseudotime'].fillna(0)
    adata.obs['dpt_pseudotime'].replace([np.inf], 1, inplace=True)
    adata.obs['dpt_pseudotime'].replace([-np.inf], 0, inplace=True)
    pseudotime = np.array(adata.obs['dpt_pseudotime'])

    cor = np.corrcoef(pseudotime, true_t)[0, 1]
    fig = pseudotime_plot(All_data, pseudotime, PCs=[0, 1])
    plt.savefig(f"{path}/PAGA.png", format='png')
    plt.close(fig)
    return cor



#基于状态转移概率的轨迹推断
def get_palantir_pseudotime(data_path, true_t, cluster, path):
    adata = anndata.read_h5ad(data_path)
    sc.pp.pca(adata)
    All_data = adata.X
    if cluster:
        adata.obs['clusters'] = adata.obs[cluster]
    else:
        num_clusters = 5
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
        kmeans.fit(All_data)
        adata.obs['clusters'] = kmeans.labels_
    adata.obs['clusters'] = adata.obs['clusters'].astype(str).astype('category')
    Traj=ov.single.TrajInfer(adata, basis='X_pca', use_rep='X_pca', n_comps=50, groupby='clusters')
    Traj.set_origin_cells(adata.obs['clusters'][0])
    Traj.set_terminal_cells([adata.obs['clusters'][-1]])
    Traj.inference(method='palantir', num_waypoints=500)

    pseudotime = np.array(adata.obs['palantir_pseudotime'])

    cor = np.corrcoef(pseudotime, true_t)[0, 1]
    fig = pseudotime_plot(All_data, pseudotime, PCs=[0, 1])
    plt.savefig(f"{path}/palantir.png", format='png')
    plt.close(fig)
    return cor








