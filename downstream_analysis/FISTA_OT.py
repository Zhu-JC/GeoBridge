import numpy as np
import Get_Probability_Measures
from tqdm import tqdm
def fista_ot(C, data_s, data_t, method ='neighbor', lambda_reg=0.1, max_iter=1000, tol=1e-6):
    """
    使用FISTA算法求解最优传输问题。

    参数：
    C : numpy.ndarray
        成本矩阵，大小为 (m, n)。
    mu : numpy.ndarray
        源分布，大小为 (m,)。
    nu : numpy.ndarray
        目标分布，大小为 (n,)。
    lambda_reg : float
        正则化参数。
    max_iter : int
        最大迭代次数。
    tol : float
        收敛容忍度。

    返回：
    P : numpy.ndarray
        最优传输计划矩阵，大小为 (m, n)。
    """
    if method == 'neighbor':
        mu = Get_Probability_Measures.Neighbor_Measures(data_s, 10, epsilon=1e-5)
        mu = mu.to_numpy()
        nu = Get_Probability_Measures.Neighbor_Measures(data_t, 10, epsilon=1e-5)
        nu = nu.to_numpy()
    else:
        mu = Get_Probability_Measures.kde_gene_expression(data_s)
        nu = Get_Probability_Measures.kde_gene_expression(data_t)

    m, n = C.shape
    psi = np.zeros(n)  # 初始化对偶变量psi
    z = np.zeros(n)
    t = 1

    for iteration in range(max_iter):
        # 更新 z
        z_old = z.copy()
        grad = np.zeros(n)
        for i in range(m):
            Vi = np.exp((psi - C[i]) / lambda_reg)
            grad += (Vi / Vi.sum()) * mu[i]

        grad -= nu
        z = psi - lambda_reg * grad

        # 执行近端步骤
        z -= np.mean(z)

        # 更新 psi
        t_old = t
        t = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
        psi = z + ((t_old - 1) / t) * (z - z_old)

        # 检查收敛
        if np.linalg.norm(z - z_old) < tol:
            break

    # 计算最优传输计划
    P = np.zeros((m, n))
    for i in range(m):
        P[i] = np.exp((psi - C[i]) / lambda_reg)
        P[i] /= (P[i].sum())
        P[i] = P[i] * mu[i]

    return P, psi, mu, nu


def fista_ot2(C, mu, nu, lambda_reg=0.1, max_iter=1000, tol=1e-6):
    """
    使用FISTA算法求解最优传输问题。

    参数：
    C : numpy.ndarray
        成本矩阵，大小为 (m, n)。
    mu : numpy.ndarray
        源分布，大小为 (m,)。
    nu : numpy.ndarray
        目标分布，大小为 (n,)。
    lambda_reg : float
        正则化参数。
    max_iter : int
        最大迭代次数。
    tol : float
        收敛容忍度。

    返回：
    P : numpy.ndarray
        最优传输计划矩阵，大小为 (m, n)。
    """
    m, n = C.shape
    C = C.T
    phi = np.zeros(m)  # 初始化对偶变量phi
    z = np.zeros(m)
    t = 1

    for iteration in range(max_iter):
        # 更新 z
        z_old = z.copy()
        grad = np.zeros(m)
        for i in range(n):
            Vi = np.exp((-phi - C[i]) / lambda_reg)
            grad += (Vi / Vi.sum()) * nu[i]

        grad = mu - grad
        z = phi - lambda_reg * grad

        # 执行近端步骤
        z -= np.mean(z)

        # 更新 psi
        t_old = t
        t = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
        phi = z + ((t_old - 1) / t) * (z - z_old)

        # 检查收敛
        if np.linalg.norm(z - z_old) < tol:
            break

    # 计算最优传输计划
    P = np.zeros((n, m))
    for i in range(n):
        P[i] = np.exp((-phi - C[i]) / lambda_reg)
        P[i] /= (P[i].sum())
        P[i] = P[i] * nu[i]

    return P.T, phi


def fista_ot2_fast(C, mu, nu, lambda_reg=0.1, max_iter=1000, tol=1e-6):
    m, n = C.shape
    phi = np.zeros(m)
    z = np.zeros(m)
    t = 1
    for iteration in tqdm(range(max_iter)):
        z_old = z.copy()
        # 向量化梯度计算
        exp_term = np.exp((-phi[:, None] - C) / lambda_reg)  # m×n
        exp_norm = exp_term / exp_term.sum(axis=0, keepdims=True)
        grad = mu - exp_norm @ nu  # m-dim
        # FISTA更新
        z = phi - lambda_reg * grad
        z -= np.mean(z)
        t_old = t
        t = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
        phi = z + ((t_old - 1) / t) * (z - z_old)
        if np.linalg.norm(z - z_old) < tol:
            break
    exp_term = np.exp((-phi[:, None] - C) / lambda_reg)
    exp_norm = exp_term / exp_term.sum(axis=0, keepdims=True)
    P = exp_norm * nu
    return P, phi


def fista_ot3(C, data_s, data_t, method ='neighbor', lambda_reg=0.1, max_iter=1000, tol=1e-6):
    """
    使用FISTA算法求解最优传输问题。

    参数：
    C : numpy.ndarray
        成本矩阵，大小为 (m, n)。
    mu : numpy.ndarray
        源分布，大小为 (m,)。
    nu : numpy.ndarray
        目标分布，大小为 (n,)。
    lambda_reg : float
        正则化参数。
    max_iter : int
        最大迭代次数。
    tol : float
        收敛容忍度。

    返回：
    P : numpy.ndarray
        最优传输计划矩阵，大小为 (m, n)。
    """
    if method == 'neighbor':
        mu = Get_Probability_Measures.Neighbor_Measures(data_s, 10, epsilon=1e-5)
        mu = mu.to_numpy()
        nu = Get_Probability_Measures.Neighbor_Measures(data_t, 10, epsilon=1e-5)
        nu = nu.to_numpy()
    else:
        mu = Get_Probability_Measures.kde_gene_expression(data_s)
        nu = Get_Probability_Measures.kde_gene_expression(data_t)
    m, n = C.shape
    C = C.T
    phi = np.zeros(m)  # 初始化对偶变量phi
    z = np.zeros(m)
    t = 1

    for iteration in range(max_iter):
        # 更新 z
        z_old = z.copy()
        Vi = np.exp((-phi - C) / lambda_reg)  # `phi` 广播到 C 的每一行
        Vi_sum = Vi.sum(axis=1, keepdims=True)  # 对每一行求和
        grad = np.sum((Vi / Vi_sum) * nu[:, None], axis=0)

        grad = mu - grad
        z = phi - lambda_reg * grad

        # 执行近端步骤
        z -= np.mean(z)

        # 更新 psi
        t_old = t
        t = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
        phi = z + ((t_old - 1) / t) * (z - z_old)

        # 检查收敛
        if np.linalg.norm(z - z_old) < tol:
            break

    # 计算最优传输计划
    print(iteration)
    P = np.zeros((n, m))
    for i in range(n):
        P[i] = np.exp((-phi - C[i]) / lambda_reg)
        P[i] /= (P[i].sum())
        P[i] = P[i] * nu[i]

    return P.T, phi, mu, nu







