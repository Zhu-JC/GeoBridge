# config.py
import torch
from pathlib import Path

# --- Global Settings ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_SAVE_DIR = Path("./results")

# --- Multi-Model Training Configurations ---
TRAINING_CONFIGS = {
    "EMT": {
        "data_path": Path("./data/adata_EMT.h5ad"),
        "save_path": BASE_SAVE_DIR / 'model',
        "time_column": "day",
        "is_sparse": False,
        "seed": 42,
        "hidden_dim": 1024,
        "num_blocks": 6,
        "batch_size": 2400,
        "epochs": 500,
        "learning_rate": 0.001,
        "lr_scheduler_t_max": 100,
        "lr_scheduler_eta_min": 1e-6,
        "grad_clip_max_norm": 1.0,
        'OT_REGULARIZATION': 1e-2,
        'num_gap': 1,
        "val_epoch_start": 400,
        "lambda_cvl": 1.0,
        "lambda_iso": 100.0,
        "use_pca": False,
        "use_kde": True
    },
    "sc_beta": {
        "data_path": Path("./data/adata_beta.h5ad"),
        "save_path": BASE_SAVE_DIR / 'model',
        "time_column": "CellDay",
        "is_sparse": False,
        "seed": 42,
        "hidden_dim": 1024,
        "num_blocks": 6,
        "batch_size": 4200,
        "epochs": 100,
        "learning_rate": 0.001,
        "lr_scheduler_t_max": 100,
        "lr_scheduler_eta_min": 1e-6,
        "grad_clip_max_norm": 1.0,
        'OT_REGULARIZATION': 1e-2,
        'num_gap': 1,
        "val_epoch_start": -1,
        "lambda_cvl": 1.0,
        "lambda_iso": 100.0,
        "use_pca": False,
        "use_kde": False
    },
    "eb": {
        "data_path": Path("./data/adata_eb.h5ad"),
        "save_path": BASE_SAVE_DIR / 'model',
        "time_column": "time_labels",
        "is_sparse": False,
        "seed": 42,
        "hidden_dim": 1024,
        "num_blocks": 6,
        "batch_size": 3000,
        "epochs": 500,
        "learning_rate": 0.001,
        "lr_scheduler_t_max": 100,
        "lr_scheduler_eta_min": 1e-6,
        "grad_clip_max_norm": 1.0,
        'OT_REGULARIZATION': 1e-2,
        'num_gap': 0,
        "val_epoch_start": 400,
        "lambda_cvl": 1.0,
        "lambda_iso": 100.0,
        "use_pca": True,
        "use_kde": True
    }
}