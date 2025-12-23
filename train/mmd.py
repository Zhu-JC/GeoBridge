import torch

def rbf_kernel(x, y, gamma):
    pairwise_distance = torch.cdist(x, y, p=2) ** 2
    return torch.exp(-gamma * pairwise_distance)

def linear_kernel(x, y):
    return torch.matmul(x, y.T)

def mmd_distance(x, y, gamma):
    xx = rbf_kernel(x, x, gamma)
    xy = rbf_kernel(x, y, gamma)
    yy = rbf_kernel(y, y, gamma)

    return xx.mean() + yy.mean() - 2 * xy.mean()

def compute_scalar_mmd(target, transport, gammas=None):
    if gammas is None:
        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]

    def safe_mmd(target, transport, gamma):
        return mmd_distance(target, transport, gamma)

    mmd_values = torch.stack([safe_mmd(target, transport, gamma) for gamma in gammas])
    return mmd_values.mean()

def compute_linear_mmd(target, transport):

    xx = linear_kernel(target, target)
    xy = linear_kernel(target, transport)
    yy = linear_kernel(transport, transport)
    mmd_sq = xx.mean() + yy.mean() - 2 * xy.mean()

    return mmd_sq

def rbf_genes_kernel(X, Y, gamma):
    """
    Calculates the RBF kernel matrix between corresponding columns of X and Y.

    Args:
        X: A PyTorch tensor of shape (n_samples_x, n_features).
        Y: A PyTorch tensor of shape (n_samples_y, n_features).
        gamma: The bandwidth parameter of the RBF kernel.

    Returns:
        A PyTorch tensor of shape (n_features,) containing the RBF kernel sums for each column pair.
    """
    # (n_samples_x, 1, n_features)
    X_col = X.unsqueeze(1)
    # (1, n_samples_y, n_features)
    Y_col = Y.unsqueeze(0)
    # (n_features,)
    kernel_sum = torch.exp(-gamma * (X_col - Y_col).pow(2)).sum(dim=(0,1))

    return kernel_sum


def mmd_genes_distance(X, Y, gamma):
    """
    Calculates the MMD distance between corresponding columns of X and Y using the RBF kernel.

    Args:
        X: A PyTorch tensor of shape (n_samples_x, n_features).
        Y: A PyTorch tensor of shape (n_samples_y, n_features).
        gamma: The bandwidth parameter of the RBF kernel.

    Returns:
        A PyTorch tensor of shape (n_features,) containing the MMD distance between corresponding columns.
    """
    n_samples_x = X.shape[0]
    n_samples_y = Y.shape[0]

    XX_sum = rbf_genes_kernel(X, X, gamma)
    XY_sum = rbf_genes_kernel(X, Y, gamma)
    YY_sum = rbf_genes_kernel(Y, Y, gamma)

    # Calculate MMD^2 for each column
    mmd_squared = (XX_sum / (n_samples_x * n_samples_x)
                   - 2 * XY_sum / (n_samples_x * n_samples_y)
                   + YY_sum / (n_samples_y * n_samples_y))


    return mmd_squared


def compute_vector_mmd(X, Y, gammas=None):
    """
    Calculates the mean MMD distance between corresponding columns of X and Y for multiple gamma values.

    Args:
        X: A PyTorch tensor of shape (n_samples_x, n_features).
        Y: A PyTorch tensor of shape (n_samples_y, n_features).
        gammas: A list of gamma values for the RBF kernel.

    Returns:
        A PyTorch tensor of shape (n_features,) containing the mean MMD distance for each column.
    """
    if gammas is None:
        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]

    mmd_values = torch.stack([mmd_genes_distance(X, Y, gamma) for gamma in gammas], dim=1)
    return mmd_values.mean(dim=1)

