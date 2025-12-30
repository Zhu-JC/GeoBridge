# snapshot_trainer.py
import sys, os
current_path = os.getcwd()
sys.path.append(os.path.join(current_path, "train"))
import torch
import copy
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from data_loader import stratified_sampling
from losses import compute_cvl_loss, compute_iso_loss, compute_velocity_consistency_loss
from utils import scaled_output
from sklearn.cluster import KMeans
from downstream_analysis.Get_Probability_Measures import kde_gene_expression
from downstream_analysis.FISTA_OT import fista_ot2_fast
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


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


def get_All_t(All_data, cluster_labels, init_target):

    data_s = All_data[cluster_labels != init_target]
    data_t = All_data[cluster_labels == init_target]
    data_t1 = pd.DataFrame(data_t)
    data_s1 = pd.DataFrame(data_s)
    C = cdist(data_s1.values, data_t1.values, metric='euclidean')
    C2 = 1000000 * C / C.sum()
    mu = kde_gene_expression(data_s1)
    nu = kde_gene_expression(data_t1)
    KP = -fista_ot2_fast(C2, mu, nu, lambda_reg=1e-3, max_iter=1000, tol=1e-6)[1]
    KP_max = KP.max()
    KP_min = KP.min()
    KP_scaled = 1 - ((KP - KP_min) / (KP_max - KP_min))
    pseudotime = np.zeros_like(cluster_labels).astype(float)
    pseudotime[cluster_labels != init_target] = KP_scaled
    pseudotime[cluster_labels == init_target] = 1

    All_t = np.zeros_like(cluster_labels).astype(float)
    labels = np.unique(cluster_labels)

    # 遍历每一份索引
    for label in labels:
        # 取出这一份原始时间标签
        indices = cluster_labels == label
        t_segment = pseudotime[indices]

        # 计算这一份时间标签的平均值
        mean_time = np.mean(t_segment)

        # 将平均值赋值给这一份所有点在 All_t 中的位置
        All_t[indices] = mean_time
    All_t = All_t - All_t.min()
    return pseudotime, All_t

def get_clusters(All_data, num_clusters=5):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
    kmeans.fit(All_data)
    cluster_labels = kmeans.labels_
    return cluster_labels

class SnapShotTrainer:
    def __init__(self, model, config, train_data, true_t, interval, cluster_labels, num_clusters, save_dir):
        self.config = config
        self.device = config['device']
        self.model = model.to(self.device)
        self.interval = interval
        self.save_dir = save_dir

        self.train_data = train_data
        self.true_t = true_t
        self.train_data_np = train_data.cpu().numpy()
        if cluster_labels:
            self.cluster_labels = cluster_labels
        else:
            self.cluster_labels = get_clusters(self.train_data_np, num_clusters=num_clusters)
        self.init_target = self.cluster_labels[-1]
        self.pseudotime, self.train_t_np = get_All_t(self.train_data_np, self.cluster_labels, init_target=self.init_target)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['lr_scheduler_t_max'],
            eta_min=self.config['lr_scheduler_eta_min']
        )

        self.best_model_state = None
        self.min_val_loss = float('inf')
        self.cor_list = []
        self.pseudotime_list = []
        cor = np.corrcoef(self.pseudotime, self.true_t)[0, 1]
        self.cor_list.append(cor)

    def train(self):
        print("--- Starting Training ---")

        for epoch in range(1, self.config['epochs'] + 1):
            self.model.train()
            # 1. 数据采样
            batch_data, batch_t = stratified_sampling(
                self.train_data_np, self.train_t_np, self.config['batch_size']
            )
            batch_data, batch_t = batch_data.to(self.device), batch_t.to(self.device)

            # 2. 前向传播
            z, _ = self.model(batch_data)
            z_min = z.min(dim=0, keepdim=True)[0]
            z_max = z.max(dim=0, keepdim=True)[0]
            z_scaled = scaled_output(z)

            # 3. 计算损失
            loss_cvl = compute_cvl_loss(
                t=batch_t, z=z_scaled, input_data=batch_data, inn_model=self.model,
                z_min=z_min, z_max=z_max, ot_reg=self.config['OT_REGULARIZATION'], num_gap=self.config['num_gap']
            )
            loss_iso = compute_iso_loss(batch_data, z_scaled, batch_t)

            total_loss = self.config['lambda_cvl'] * loss_cvl + self.config['lambda_iso'] * loss_iso
            print(f'Epoch:{epoch}, cvl_loss:{loss_cvl}, iso_loss:{loss_iso}')
            # 4. 验证逻辑
            # 检查 val_epoch_start 是否为 -1 来决定是否进行验证
            if epoch > -1:
                self.validate()
            # 5. 更新伪时序训练标签
            if epoch % self.interval == 0:
                with torch.no_grad():
                    z2 = self.best_model_state(self.train_data)[0]
                    z2 = scaled_output(z2)
                    All_data_np = z2.cpu().numpy()
                    self.pseudotime, self.train_t_np = get_All_t(All_data_np, self.cluster_labels, init_target=self.init_target)
                    cor = np.corrcoef(self.pseudotime, self.true_t)[0, 1]

                    self.pseudotime_list.append(self.pseudotime)
                    self.cor_list.append(cor)
                    fig = pseudotime_plot(self.train_data_np, self.pseudotime, PCs=[0, 1])
                    plt.savefig(f"{self.save_dir}/GeoBridge_pseudotime_{epoch}.png", format='png')
                    plt.close(fig)
                    fig2 = pseudotime_plot(All_data_np, self.pseudotime, PCs=[0, 1])
                    plt.savefig(f"{self.save_dir}/GeoBridge_pseudotime_{epoch}_eu.png", format='png')
                    plt.close(fig2)


            # 6. 反向传播和优化
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config['grad_clip_max_norm'])
            self.optimizer.step()
            self.scheduler.step()


    def validate(self):
        self.model.eval()
        with torch.no_grad():
            # 使用全部训练数据进行验证
            full_z, _ = self.model(torch.from_numpy(self.train_data_np).float().to(self.device))
            full_z_scaled = scaled_output(full_z)
            val_loss = compute_velocity_consistency_loss(
                torch.from_numpy(self.train_t_np).float().to(self.device), full_z_scaled
            )
            if val_loss < self.min_val_loss:
                self.min_val_loss = val_loss
                self.best_model_state = copy.deepcopy(self.model)
                print(f"  New best model saved with validation loss: {self.min_val_loss:.4f}")