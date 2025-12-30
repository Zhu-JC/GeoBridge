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

config = TRAINING_CONFIGS['HSCs']
output_dir = os.path.join(current_path, "results", "HSCs_analysis")
os.makedirs(output_dir, exist_ok=True)
# 1.load data
adata = anndata.read_h5ad(config['data_path'])
All_data = torch.from_numpy(adata.X.toarray()).to(DEVICE)
All_t = np.array(adata.obs['day'])
All_label = np.array(adata.obs['cell_type'])
unique_labels = np.unique(All_label)
HVG = adata.var['highly_variable'].index
# 2.load model
n_dim = All_data.shape[1]
inn = load_trained_model(
    config=config,
    n_dim=n_dim
)
inn.to(DEVICE)

###简单的PCA图
colors_order = ['yellow', 'brown', 'green', 'red', 'blue', 'purple', 'orange']
GeoBridge_analysis.plot_PCA(All_data, All_label, unique_labels, unique_labels, colors_order, inn, plot='org', PCs=[0,1], spot_size=10, path=output_dir)
GeoBridge_analysis.plot_PCA(All_data, All_label, unique_labels, unique_labels, colors_order, inn, plot='eu', PCs=[0,1], spot_size=5, path=output_dir)


###动态gif图
source = (All_t == 2)
target = (All_t == 7)

path = f'{output_dir}/dynamic_Org'
GeoBridge_analysis.Dynamic_plot_mt(All_data, All_data, All_label, unique_labels, All_t, inn, method='kde', source=source, target=target, target_list=['MkP','MasP','NeuP'], color_list=['blue', 'red', 'orange'], num_inter=300, reg=2e-2,
                                show_velocity=10, velocity_length=5, PCs=[0,1], path=path, plot='org')
path = f'{output_dir}/dynamic_Eu'
GeoBridge_analysis.Dynamic_plot_mt(All_data, All_data, All_label, unique_labels, All_t, inn, method='kde', source=source, target=target, target_list=['MkP','MasP','NeuP'], color_list=['blue', 'red', 'orange'], num_inter=300, reg=2e-2,
                                show_velocity=10, velocity_length=0.25, PCs=[0,1], path=path, plot='eu')

###30条路径变化路径
GeoBridge_analysis.inter_plot_mt(All_data, All_data, All_label, All_t, inn, traj_num=30, method='kde', source=source, target=target, target_list=['MkP','MasP','NeuP'],color_list=['black', 'black', 'black'], num_inter=70, reg=2e-2, PCs=[0,1],
                              plot='org', arrow_size=2.5, avg_width=5, path=output_dir)
GeoBridge_analysis.inter_plot_mt(All_data, All_data, All_label, All_t, inn, traj_num=30, method='kde', source=source, target=target, target_list=['MkP','MasP','NeuP'], color_list=['black', 'black', 'black'], num_inter=70, reg=2e-2, PCs=[0,1],
                              plot='eu', arrow_size=0.1, avg_width=5, path=output_dir)

###细胞命运plot
source_list = [2, 3, 4]
target = (All_t == 7)
cell_fate_list, All_source_label = GeoBridge_analysis.Decide_fate_mt(All_data, All_label, All_t, inn, source_list, method='kde', target=target, reg=2e-2)

All_source_labels = np.stack(All_source_label)
All_label_fate = np.concatenate(cell_fate_list)
colors_order = ['yellow', 'brown', 'green', 'red', 'blue', 'purple', 'orange']
for key_label in source_list:
    GeoBridge_analysis.fate_distribution_PCA(All_data, All_t, All_t, All_source_labels, All_label_fate, key_label, ['MkP', 'MasP', 'NeuP'], colors_order, PCs=[0,1], spot_size=3, path=output_dir)


###画路径的所有基因表达动态变化的对比，先做最优传输，构建中间分布，再求均值
target_list=['MkP','MasP','NeuP']
color_list=['blue', 'red', 'orange']
genenames = HVG
GeoBridge_analysis.OT_gene_Mean_Dynamic_compair_mt(All_data, All_t, inn, cell_fate_list[0], genenames, method='kde', source=source, target=target, num_inter=50, color_list=color_list, reg=2e-2, path=output_dir, fate_list=target_list)

###single-cell navigation
for i in range(50):
    GeoBridge_analysis.pair_path_plot(All_data, All_t, inn, 100, traj_width=5, cmap='plasma', PCs=[0, 1], path=output_dir)

###cell-type navigation
GeoBridge_analysis.Cell_Navigation(All_data, All_t, inn, cell_fate_list[0],
                navi_org='NeuP', navi_target='MasP', HVG=HVG,
                source=source, target=target, PCs=[0, 1], method='kde', reg=2e-2, num_inter=200, path=output_dir)

###计算动态driver index
driver_index_MkP, driver_genes_MkP = Detect_driver.plot_top_Dynamic_driver(All_data, All_t, inn, HVG, cell_fate_list[0], source, target,
                                            method='kde', reg=2e-2, name='MkP', n_top=100, figsize=(3, 3), fontsize=2, title_sz=5,  num_gene_clusters=2, path=output_dir)
driver_index_NeuP, driver_genes_NeuP = Detect_driver.plot_top_Dynamic_driver(All_data, All_t, inn, HVG, cell_fate_list[0], source, target,
                                            method='kde', reg=2e-2, name='NeuP', n_top=100, figsize=(3, 3), fontsize=2, title_sz=5,  num_gene_clusters=2, path=output_dir)
driver_index_MasP, driver_genes_MasP = Detect_driver.plot_top_Dynamic_driver(All_data, All_t, inn, HVG, cell_fate_list[0], source, target,
                                            method='kde', reg=2e-2, name='MasP', n_top=100, figsize=(3, 3), fontsize=2, title_sz=5,  num_gene_clusters=2, path=output_dir)
driver_index_list=[driver_index_MkP, driver_index_MasP, driver_index_NeuP]

path = f'{output_dir}/dynamic_gene_exp'
Detect_driver.plot_driver_exp_cor(driver_index_list, target_list, path=path)
