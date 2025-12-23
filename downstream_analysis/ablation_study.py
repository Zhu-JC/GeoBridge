import sys, os
current_path = os.getcwd()
sys.path.append(os.path.join(current_path, "downstream_analysis"))
import anndata
import torch
import numpy as np
from train.config import TRAINING_CONFIGS, DEVICE
from train.model import load_trained_model
import GeoBridge_analysis

config = TRAINING_CONFIGS['MET_EMT_All']
config_noiso = TRAINING_CONFIGS['MET_EMT_noiso']
output_dir = os.path.join(current_path, "results", "ablation_study")
# 1.load data
adata = anndata.read_h5ad(config['data_path'])
All_data = torch.from_numpy(adata.X).to(DEVICE)
All_t = np.array(adata.obs['day'])
All_label = np.array(adata.obs['label'])
unique_labels = np.unique(All_label)
HVG = adata.var.highly_variable
# 2.load model
n_dim = All_data.shape[1]
inn = load_trained_model(
    config=config,
    n_dim=n_dim
)
inn.to(DEVICE)

inn_noiso = load_trained_model(
    config=config_noiso,
    n_dim=n_dim
)
inn_noiso.to(DEVICE)

# 3.Plot dynamic gif
source = (All_t == 0)
target = (All_t == 30)

# Stantard model
path = f'{output_dir}/dynamic_Org'
GeoBridge_analysis.Dynamic_plot_mt(All_data, All_data, All_label, unique_labels, All_t, inn, method='kde', source=source, target=target, target_list=[unique_labels[4], unique_labels[5]], color_list=['green', 'orange'], num_inter=300, reg=2e-2,
                                show_velocity=10, velocity_length=3.5, PCs=[0,1], path=path, plot='org')
path = f'{output_dir}/dynamic_Eu'
GeoBridge_analysis.Dynamic_plot_mt(All_data, All_data, All_label, unique_labels, All_t, inn, method='kde', source=source, target=target, target_list=[unique_labels[4], unique_labels[5]], color_list=['green', 'orange'], num_inter=300, reg=2e-2,
                                show_velocity=10, velocity_length=0.5, PCs=[0,1], path=path, plot='eu')

# No iso model
path = f'{output_dir}/dynamic_Org_noiso'
GeoBridge_analysis.Dynamic_plot_mt(All_data, All_data, All_label, unique_labels, All_t, inn_noiso, method='kde', source=source, target=target, target_list=[unique_labels[4], unique_labels[5]], color_list=['green', 'orange'], num_inter=300, reg=3e-2,
                                show_velocity=10, velocity_length=3.5, PCs=[0,1], path=path, plot='org')
path = f'{output_dir}/dynamic_Eu_noiso'
GeoBridge_analysis.Dynamic_plot_mt(All_data, All_data, All_label, unique_labels, All_t, inn_noiso, method='kde', source=source, target=target, target_list=[unique_labels[4], unique_labels[5]], color_list=['green', 'orange'], num_inter=300, reg=3e-2,
                                show_velocity=10, velocity_length=1.5, PCs=[0,1], path=path, plot='eu')
