# utils.py
import torch
import random
import numpy as np

def set_seed(seed_value: int = 42):
    """设置随机种子以保证实验可复现性。"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed_value}")

def scaled_output(z: torch.Tensor) -> torch.Tensor:
    """对 z 进行特征维度的最小-最大归一化处理。"""
    z_min = z.min(dim=0, keepdim=True)[0]
    z_max = z.max(dim=0, keepdim=True)[0]
    # 避免分母为零
    z_norm = (z - z_min) / (z_max - z_min + 1e-8)
    return z_norm