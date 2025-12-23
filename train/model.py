# model.py
import torch
import torch.nn as nn
import torch.nn.init as init
import FrEIA.framework as Ff
import FrEIA.modules as Fm

hidden_dim=1024

def subnet_fc(dims_in, dims_out):
    """定义用于仿射耦合块的子网络。"""
    subnet = nn.Sequential(
        nn.Linear(dims_in, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, dims_out)
    )
    # 参数初始化
    for layer in subnet:
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                init.zeros_(layer.bias)
    return subnet

def build_inn_model(n_dim: int, num_blocks: int) -> Ff.SequenceINN:
    """构建 INN 模型。"""
    inn = Ff.SequenceINN(n_dim)
    for _ in range(num_blocks):
        inn.append(
            Fm.AllInOneBlock,
            subnet_constructor=subnet_fc,
            permute_soft=False
        )
    return inn


# for name, param in inn_model.named_parameters():
#     print(name)
#     print(param.data)        # .data 可以查看具体权重


def load_trained_model(config: dict, n_dim: int):
    """构建模型结构并加载预训练的权重。"""
    model = build_inn_model(
        n_dim=n_dim,
        num_blocks=config['num_blocks']
    )
    model.load_state_dict(torch.load(config['model_path']))
    model.eval()
    print("Model loaded successfully.")
    return model