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
import Pseudotime_analysis

config = TRAINING_CONFIGS['EMT']
output_dir = os.path.join(current_path, "results", f"EMT_pseudotime_results")
os.makedirs(output_dir, exist_ok=True)
# 1.load data
adata = anndata.read_h5ad(config['data_path'])
All_data = torch.from_numpy(adata.X).to(DEVICE)
True_t = np.array(adata.obs['day'])
Pseudo_t = np.load(f'{output_dir}/pseudo_t.npy')
HVG = adata.var.highly_variable

# 2.load model
n_dim = All_data.shape[1]
inn_true_t = load_trained_model(
    config=config,
    n_dim=n_dim
)
config['model_path'] = f'{output_dir}/EMT_pseudotime_model.pth'
inn_pseudo_t = load_trained_model(
    config=config,
    n_dim=n_dim
)
name1 = 'True-time'; name2 = 'Pseudo-time'
colors_order = ['blue', 'orange', 'green', 'red', 'purple']
labels = [0, 2, 12, 18, 30]

path=output_dir
# 3.Plot dynamic gif of true time model and pseudotime model
Pseudotime_analysis.Dynamic_plot(All_data, inn_true_t, inn_pseudo_t, True_t, Pseudo_t, All_data, True_t, labels,
                                 method='kde', num_inter=300, reg1=2e-2, reg2=3e-2, PCs=[0,1], path=path, name1=name1, name2=name2)
# 4.Plot average trajectory of true time model and pseudotime model
Pseudotime_analysis.Dynamic_path_plot(All_data, All_data, True_t, labels, True_t, Pseudo_t, inn_true_t, inn_pseudo_t,
                                      method='kde', num_inter=300, reg1=2e-2, reg2=3e-2, PCs=[0,1], linewidth=5, arrow_size=2.5, path=path)

# 5.Plot dynamic HVG gene expression of true time model and pseudotime model
path=f'{output_dir}/pair_gene'
genenames = HVG.index.tolist()
Pseudotime_analysis.OT_gene_Mean_Dynamic_compair(All_data, True_t, Pseudo_t, inn_true_t, inn_pseudo_t, genenames,
                                                 method='kde', num_inter=300, use = 'org', reg1=3e-2, reg2=3e-2, path=path, name1=name1, name2=name2)

# 6.Plot gene correlation between true time model and pseudotime model
path=output_dir
Pseudotime_analysis.gene_cor_distribution(path, name1, name2)








