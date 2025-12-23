import sys, os
current_path = os.getcwd()
sys.path.append(os.path.join(current_path, "downstream_analysis"))
import anndata
import torch
import numpy as np
from train.config import TRAINING_CONFIGS, DEVICE
from train.model import load_trained_model
import GeoBridge_analysis
import Detect_driver

config_beta = TRAINING_CONFIGS['sc_beta']
config_ec = TRAINING_CONFIGS['sc_ec']
config_All = TRAINING_CONFIGS['Beta_Ec_All']
output_dir = os.path.join(current_path, "results", "Beta_Ec_analysis")
# 1.load data
adata = anndata.read_h5ad(config_All['data_path'])
All_data = torch.from_numpy(adata.X).to(DEVICE)
All_t = np.array(adata.obs['CellDay'])
All_label = np.array(adata.obs['Assigned_cluster'])
unique_labels = np.unique(All_label)
HVG = adata.var.gene_names

# 2.load model
n_dim = All_data.shape[1]
inn_beta = load_trained_model(
    config=config_beta,
    n_dim=n_dim
)
inn_beta.to(DEVICE)

inn_ec = load_trained_model(
    config=config_ec,
    n_dim=n_dim
)
inn_ec.to(DEVICE)

# 3.decide cell fate
source_list = ['prog_nkx61', 'neurog3_early', 'neurog3_mid', 'neurog3_late']
target_list = [unique_labels[4], unique_labels[5]]
target = (All_label == unique_labels[4]) | (All_label == unique_labels[5])

cell_fate_list, All_source_label, beta_weights_list, ec_weights_list = GeoBridge_analysis.get_fate_list(
    All_data, All_label, inn_beta, inn_ec, source_list, target_list, target, name1='sc_beta', name2='sc_ec')

# 4.Cell fate distribution in PCA
path = output_dir
All_source_labels = np.stack(All_source_label)
All_label_fate = np.concatenate(cell_fate_list)
colors = ['red', 'brown','green', 'orange']
for key_label in source_list:
    GeoBridge_analysis.fate_distribution_PCA(
        All_data, All_t, All_label, All_source_labels, All_label_fate, key_label, ['sc_beta', 'sc_ec'], colors, PCs=[0, 2], spot_size=3, path=path)

# 5.Cell fate weights in PCA
label_names = ['prog_nkx61', 'neurog3_early', 'neurog3_mid', 'neurog3_late', 'beta', 'ec']
solid_color_map = {
    'beta': 'brown',
    'ec': 'purple'
}
path = f'{output_dir}/Beta_fate_weights'
GeoBridge_analysis.fate_weights_plot(
    All_data, All_label, unique_labels, label_names, source_list, beta_weights_list, inn_beta, solid_color_map,
    PCs=[0, 2], spot_size=15, plot='org', path=path)
GeoBridge_analysis.fate_weights_plot(
    All_data, All_label, unique_labels, label_names, source_list, beta_weights_list, inn_beta, solid_color_map,
    PCs=[0, 2], spot_size=15, plot='eu', path=path)

path = f'{output_dir}/Ec_fate_weights'
GeoBridge_analysis.fate_weights_plot(
    All_data, All_label, unique_labels, label_names, source_list, ec_weights_list, inn_ec, solid_color_map,
    PCs=[0, 2], spot_size=15, plot='org', path=path)
GeoBridge_analysis.fate_weights_plot(
    All_data, All_label, unique_labels, label_names, source_list, ec_weights_list, inn_ec, solid_color_map,
    PCs=[0, 2], spot_size=15, plot='eu', path=path)

# 6.Single-cell navigetion
path = output_dir
for i in range(50):
    GeoBridge_analysis.pair_path_plot(All_data, All_t, inn_beta, 100, traj_width=5, cmap='plasma', PCs=[0, 2], path=path)

# 7.Plot PCA
path = output_dir
colors_order = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
label_names = ['prog_nkx61', 'neurog3_early', 'neurog3_mid', 'neurog3_late', 'beta', 'ec']
GeoBridge_analysis.plot_PCA(All_data, All_label, unique_labels, label_names, colors_order, inn_beta, plot='org', PCs=[0, 2], path=path)

# 8.Plot dynamic gif
source = (All_label == source_list[0])
path = f'{output_dir}/Dynamic_org'
GeoBridge_analysis.Dynamic_plot(All_data, All_label, unique_labels, All_t, inn_beta, inn_ec, cell_fate_list[0], method='kde', source=source, target=target, num_inter=300, reg1=2e-2, reg2=2e-2,
                             show_velocity=10, velocity_length=3, plot='org', path=path, name1='sc_beta', name2='sc_ec', PCs=[0,2])
path = f'{output_dir}/Beta_eu'
GeoBridge_analysis.Dynamic_plot(All_data, All_label, unique_labels, All_t, inn_beta, inn_ec, cell_fate_list[0], method='kde', source=source, target=target, num_inter=300, reg1=2e-2, reg2=2e-2,
                             show_velocity=10, velocity_length=0.5, plot='sc_beta_eu', path=path, name1='sc_beta', name2='sc_ec', PCs=[0,2])
path = f'{output_dir}/Ec_eu'
GeoBridge_analysis.Dynamic_plot(All_data, All_label, unique_labels, All_t, inn_beta, inn_ec, cell_fate_list[0], method='kde', source=source, target=target, num_inter=300, reg1=2e-2, reg2=2e-2,
                             show_velocity=10, velocity_length=0.5, plot='sc_ec_eu', path=path, name1='sc_beta', name2='sc_ec', PCs=[0,2])

# 9.Plot 30 single trajectories and average trajectory
source = (All_label == source_list[0])
path = output_dir
GeoBridge_analysis.inter_plot(All_data, All_t, inn_beta, inn_ec, cell_fate_list[0], traj_num=30, method='kde', source=source, target=target, num_inter=70, reg1=2e-2, reg2=2e-2, PCs=[0,2], name1='sc_beta', name2='sc_ec',
           plot='org', cmap='plasma', arrow_size=1.5, avg_width=5, path=path)
GeoBridge_analysis.inter_plot(All_data, All_t, inn_beta, inn_ec, cell_fate_list[0], traj_num=30, method='kde', source=source, target=target, num_inter=70, reg1=2e-2, reg2=2e-2, PCs=[0,2], name1='sc_beta', name2='sc_ec',
           plot='sc_beta_eu', cmap='plasma', arrow_size=0.2, avg_width=5, path=path)
GeoBridge_analysis.inter_plot(All_data, All_t, inn_beta, inn_ec, cell_fate_list[0], traj_num=30, method='kde', source=source, target=target, num_inter=70, reg1=2e-2, reg2=2e-2, PCs=[0,2], name1='sc_beta', name2='sc_ec',
           plot='sc_ec_eu', cmap='plasma', arrow_size=0.2, avg_width=5, path=path)


# 10.Plot Day18 to Day30 single trajectories of MET and EMT fates
path = f'{output_dir}/Neurog3_late_to_Beta_EC'
source = (All_label == source_list[3])
label_names = ['prog_nkx61', 'neurog3_early', 'neurog3_mid', 'neurog3_late', 'beta', 'ec']
solid_color_map = {
    'beta': 'brown',
    'ec': 'purple'
}
GeoBridge_analysis.Plot_fate_path(All_data, All_t, All_label, unique_labels, label_names, source_list, beta_weights_list, inn_beta, cell_fate_list[3], solid_color_map,
               method = 'kde', source=source, target=target, reg=2e-2, PCs=[0,2], name='sc_beta', path=path, plot='org', n_path=30, color='green', traj_alpha=1, traj_width=2, linewidth=2, arrow_size=0.5)
GeoBridge_analysis.Plot_fate_path(All_data, All_t, All_label, unique_labels, label_names, source_list, ec_weights_list, inn_ec, cell_fate_list[3], solid_color_map,
               method = 'kde', source=source, target=target, reg=2e-2, PCs=[0,2], name='sc_ec', path=path, plot='org', n_path=30, color='orange', traj_alpha=1, traj_width=2, linewidth=2, arrow_size=0.5)

# 11.Plot dynamic HVG gene expression
genenames = HVG.index.tolist()
path = f'{output_dir}/dynamic_gene_exp'
source = (All_label == source_list[0])
GeoBridge_analysis.OT_gene_Mean_Dynamic_compair(All_data, All_t, inn_beta, inn_ec, cell_fate_list[0], genenames,
                                                method='kde', source=source, target=target, num_inter=300, use='org', reg1=2e-2, reg2=2e-2, path=path, name1='sc_beta', name2='sc_ec')

# 12. Plot dynamic driver index
source = (All_label == source_list[3])
driver_index_beta, driver_genes_beta = Detect_driver.plot_top_Dynamic_driver(All_data, All_t, inn_beta, HVG, cell_fate_list[3], source, target,
                                            method='kde', reg=2e-2, name='sc_beta', n_top=100, figsize=(3, 3), fontsize=2, title_sz=5,  num_gene_clusters=2, path=output_dir)
driver_index_ec, driver_genes_ec = Detect_driver.plot_top_Dynamic_driver(All_data, All_t, inn_ec, HVG, cell_fate_list[3], source, target,
                                            method='kde', reg=2e-2, name='sc_ec', n_top=100, figsize=(3, 3), fontsize=2, title_sz=5,  num_gene_clusters=2, path=output_dir)
driver_index_list=[driver_index_beta, driver_index_ec]

path = f'{output_dir}/dynamic_gene_exp'
Detect_driver.plot_driver_exp_cor(driver_index_list, target_list, path=path)












































