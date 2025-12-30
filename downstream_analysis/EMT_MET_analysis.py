import sys, os
current_path = os.getcwd()
sys.path.append(os.path.join(current_path, "downstream_analysis"))
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)
import anndata
import torch
import numpy as np
from train.config import TRAINING_CONFIGS, DEVICE
from train.model import load_trained_model
import GeoBridge_analysis
import Detect_driver

config_MET = TRAINING_CONFIGS['MET']
config_EMT = TRAINING_CONFIGS['EMT']
config_All = TRAINING_CONFIGS['MET_EMT_All']
output_dir = os.path.join(current_path, "results", "MET_EMT_analysis")
os.makedirs(output_dir, exist_ok=True)
# 1.load data
adata = anndata.read_h5ad(config_All['data_path'])
All_data = torch.from_numpy(adata.X).to(DEVICE)
All_t = np.array(adata.obs['day'])
All_label = np.array(adata.obs['label'])
unique_labels = np.unique(All_label)
HVG = adata.var.highly_variable

# 2.load model
n_dim = All_data.shape[1]
inn_MET = load_trained_model(
    config=config_MET,
    n_dim=n_dim
)
inn_MET.to(DEVICE)

inn_EMT = load_trained_model(
    config=config_EMT,
    n_dim=n_dim
)
inn_EMT.to(DEVICE)

# 3.decide cell fate
source_list = [0, 2, 12, 18]
target_list = [unique_labels[5], unique_labels[4]]
target = (All_label == unique_labels[4]) | (All_label == unique_labels[5])

cell_fate_list, All_source_label, MET_weights_list, EMT_weights_list = GeoBridge_analysis.get_fate_list(
    All_data, All_label, inn_MET, inn_EMT, source_list, target_list, target, name1='MET', name2='EMT')

# 4.Cell fate distribution in PCA
path = output_dir
All_source_labels = np.stack(All_source_label)
All_label_fate = np.concatenate(cell_fate_list)
colors = ['green', 'red', 'brown', 'orange']
for key_label in source_list:
    GeoBridge_analysis.fate_distribution_PCA(
        All_data, All_t, All_label, All_source_labels, All_label_fate, key_label, ['MET', 'EMT'], colors, PCs=[0,1], spot_size=3, path=path)

# 5.Cell fate weights in PCA
label_names = ['0', '2', '12', '18', 'EMT', 'MET']
solid_color_map = {
    'EMT': 'purple',
    'MET': 'brown'
}
path = f'{output_dir}/MET_fate_weights'
GeoBridge_analysis.fate_weights_plot(
    All_data, All_label, unique_labels, label_names, source_list, MET_weights_list, inn_MET, solid_color_map,
    PCs=[0, 1], spot_size=15, plot='org', path=path)
GeoBridge_analysis.fate_weights_plot(
    All_data, All_label, unique_labels, label_names, source_list, MET_weights_list, inn_MET, solid_color_map,
    PCs=[0, 1], spot_size=15, plot='eu', path=path)

path = f'{output_dir}/EMT_fate_weights'
GeoBridge_analysis.fate_weights_plot(
    All_data, All_label, unique_labels, label_names, source_list, EMT_weights_list, inn_EMT, solid_color_map,
    PCs=[0, 1], spot_size=15, plot='org', path=path)
GeoBridge_analysis.fate_weights_plot(
    All_data, All_label, unique_labels, label_names, source_list, EMT_weights_list, inn_EMT, solid_color_map,
    PCs=[0, 1], spot_size=15, plot='eu', path=path)

# 6.Single-cell navigetion
path = output_dir
for i in range(50):
    GeoBridge_analysis.pair_path_plot(All_data, All_t, inn_MET, 100, traj_width=5, cmap='plasma', PCs=[0, 1], path=path)

# 7.Plot PCA
path = output_dir
colors_order = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
label_names = ['Day 0', 'Day 2', 'Day 12', 'Day 18', 'Day30-Mes', 'Day30-Epi']
GeoBridge_analysis.plot_PCA(All_data, All_label, unique_labels, label_names, colors_order, inn_MET, plot='org', PCs=[0, 1], path=path)

# 8.Plot dynamic gif
source = (All_label == source_list[0])
path = f'{output_dir}/Dynamic_org'
GeoBridge_analysis.Dynamic_plot(All_data, All_label, unique_labels, All_t, inn_EMT, inn_MET, cell_fate_list[0], method='kde', source=source, target=target, num_inter=300, reg1=2e-2, reg2=2e-2,
                             show_velocity=10, velocity_length=3, plot='org', path=path, name1='EMT', name2='MET', PCs=[0,1])
path = f'{output_dir}/EMT_eu'
GeoBridge_analysis.Dynamic_plot(All_data, All_label, unique_labels, All_t, inn_EMT, inn_MET, cell_fate_list[0], method='kde', source=source, target=target, num_inter=300, reg1=2e-2, reg2=2e-2,
                             show_velocity=10, velocity_length=0.5, plot='EMT_eu', path=path, name1='EMT', name2='MET', PCs=[0,1])
path = f'{output_dir}/MET_eu'
GeoBridge_analysis.Dynamic_plot(All_data, All_label, unique_labels, All_t, inn_EMT, inn_MET, cell_fate_list[0], method='kde', source=source, target=target, num_inter=300, reg1=2e-2, reg2=2e-2,
                             show_velocity=10, velocity_length=0.5, plot='MET_eu', path=path, name1='EMT', name2='MET', PCs=[0,1])

# 9.Plot 30 single trajectories and average trajectory
source = (All_label == source_list[0])
path = output_dir
GeoBridge_analysis.inter_plot(All_data, All_t, inn_MET, inn_EMT, cell_fate_list[0], traj_num=30, method='kde', source=source, target=target, num_inter=70, reg1=2e-2, reg2=2e-2, PCs=[0,1], name1='MET', name2='EMT',
           plot='org', cmap='plasma', arrow_size=1.5, avg_width=5, path=path)
GeoBridge_analysis.inter_plot(All_data, All_t, inn_MET, inn_EMT, cell_fate_list[0], traj_num=30, method='kde', source=source, target=target, num_inter=100, reg1=2e-2, reg2=2e-2, PCs=[0,1], name1='MET', name2='EMT',
           plot='MET_eu', cmap='plasma', arrow_size=0.35, avg_width=5, path=path)
GeoBridge_analysis.inter_plot(All_data, All_t, inn_MET, inn_EMT, cell_fate_list[0], traj_num=30, method='kde', source=source, target=target, num_inter=70, reg1=2e-2, reg2=2e-2, PCs=[0,1], name1='MET', name2='EMT',
           plot='EMT_eu', cmap='plasma', arrow_size=0.25, avg_width=5, path=path)


# 10.Plot Day18 to Day30 single trajectories of MET and EMT fates
path = f'{output_dir}/Day18_to_Day30'
source = (All_label == source_list[3])
label_names = ['0', '2', '12', '18', 'EMT', 'MET']
solid_color_map = {
    'EMT': 'purple',
    'MET': 'brown'
}
GeoBridge_analysis.Plot_fate_path(All_data, All_t, All_label, unique_labels, label_names, source_list, MET_weights_list, inn_MET, cell_fate_list[3], solid_color_map,
               method = 'kde', source=source, target=target, reg=2e-2, PCs=[0,1], name='MET', path=path, plot='org', n_path=30, color='orange', traj_alpha=1, traj_width=2, linewidth=3, arrow_size=1.2)
GeoBridge_analysis.Plot_fate_path(All_data, All_t, All_label, unique_labels, label_names, source_list, EMT_weights_list, inn_EMT, cell_fate_list[3], solid_color_map,
               method = 'kde', source=source, target=target, reg=2e-2, PCs=[0,1], name='EMT', path=path, plot='org', n_path=30, color='green', traj_alpha=1, traj_width=2, linewidth=3, arrow_size=1.2)

# 11.Plot dynamic HVG gene expression
genenames = HVG.index.tolist()
path = f'{output_dir}/dynamic_gene_exp'
source = (All_label == source_list[0])
GeoBridge_analysis.OT_gene_Mean_Dynamic_compair(All_data, All_t, inn_MET, inn_EMT, cell_fate_list[0], genenames,
                                                method='kde', source=source, target=target, num_inter=300, use='org', reg1=2e-2, reg2=2e-2, path=path, name1='MET', name2='EMT')

# 12. Plot dynamic driver index
source = (All_label == source_list[3])
driver_index_MET, driver_genes_MET = Detect_driver.plot_top_Dynamic_driver(All_data, All_t, inn_MET, genenames, cell_fate_list[3], source, target,
                                            method='kde', reg=2e-2, name='MET', n_top=100, figsize=(3, 3), fontsize=2, title_sz=5,  num_gene_clusters=3, path=output_dir)
driver_index_EMT, driver_genes_EMT = Detect_driver.plot_top_Dynamic_driver(All_data, All_t, inn_EMT, genenames, cell_fate_list[3], source, target,
                                            method='kde', reg=2e-2, name='EMT', n_top=100, figsize=(3, 3), fontsize=2, title_sz=5,  num_gene_clusters=2, path=output_dir)
driver_index_list=[driver_index_MET, driver_index_EMT]

path = output_dir
Detect_driver.plot_driver_exp_cor(driver_index_list, ['MET', 'EMT'], path=path)











































