# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Import Packages
import numpy as np

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Unconstrained Gradient Descent Matrix Form
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def un_mf(r, etup, k, alpha=0.005, tol=0.001,
          iter_max=100000, seed=42):
    n, m = r.shape
    np.random.seed(seed)
    u = np.zeros((n, k))
    u_0 = np.random.randint(low=0,
                            high=2,
                            size=(n, k))
    v = np.zeros((m, k))
    v_0 = np.random.randint(low=0,
                            high=2,
                            size=(m, k))
    e = np.zeros((n, m))
    norm = 1
    ind = 1
    while (norm > tol) & (ind < iter_max):
        e_0 = e.copy()
        u = u_0 + alpha * e @ v_0
        v = v_0 + alpha * e.T @ u_0
        e = r - u @ v.T

        for i, j in etup:
            e[i, j] = 0

        u_0 = u.copy()
        v_0 = v.copy()
        l_0 = np.linalg.norm(e_0, ord="fro") ** 2
        l_1 = np.linalg.norm(e, ord="fro") ** 2
        norm = np.abs(l_1 - l_0)
        ind += 1
        print("The norm: {:6.5f} || iter: {:d}".format(
            norm, ind
        ))
    return u, v

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Unconstrained Reg GD Matrix Form
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def un_reg_mf(r, etup, k, alpha=0.005, tol=0.001,
              reg=10, iter_max=100000, seed=42):
    n, m = r.shape
    np.random.seed(seed)
    u = np.zeros((n, k))
    u_0 = np.random.randint(low=0,
                            high=2,
                            size=(n, k))
    v = np.zeros((m, k))
    v_0 = np.random.randint(low=0,
                            high=2,
                            size=(m, k))
    e = np.zeros((n, m))
    norm = 1
    ind = 1
    coef = 1 - alpha * reg
    while (norm > tol) & (ind < iter_max):
        e_0 = e.copy()
        u = u_0 * coef + alpha * e @ v_0
        v = v_0 * coef + alpha * e.T @ u_0
        e = r - u @ v.T

        for i, j in etup:
            e[i, j] = 0

        norm_e_0 = np.linalg.norm(e_0, ord="fro") ** 2
        norm_u_0 = np.linalg.norm(u_0, ord="fro") ** 2
        norm_u = np.linalg.norm(u, ord="fro") ** 2
        norm_v_0 = np.linalg.norm(v_0, ord="fro") ** 2
        norm_v = np.linalg.norm(v, ord="fro") ** 2
        norm_e = np.linalg.norm(e, ord="fro") ** 2
        l_0 = norm_e_0 + reg * norm_u_0 + reg * norm_v_0
        l_1 = norm_e + reg * norm_u + reg * norm_v
        norm = np.abs(l_1 - l_0)
        ind += 1
        u_0 = u.copy()
        v_0 = v.copy()
        print("The norm: {:6.5f} || iter: {:d}".format(
            norm, ind
        ))
    return u, v

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Unconstrained Reg Biased GD Matrix Form
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def un_reg_bi_mf(r, etup, k, alpha=0.005, tol=0.001,
                 reg=10, iter_max=100000, seed=42):
    n, m = r.shape
    k = k - 2
    np.random.seed(seed)
    mean = np.sum(r) / (n*m)
    r = r - mean
    u = np.zeros((n, k+2))
    u_0 = np.random.randint(low=0,
                            high=2,
                            size=(n, k+2))
    for i in range(n):
        u_0[i, k+1] = 1

    v = np.zeros((m, k+2))
    v_0 = np.random.randint(low=0,
                            high=2,
                            size=(m, k+2))

    for j in range(m):
        v_0[j, k] = 1

    e = np.zeros((n, m))
    norm = 1
    ind = 1
    coef = 1 - alpha * reg
    while (norm > tol) & (ind < iter_max):
        e_0 = e.copy()
        u = u_0 * coef + alpha * e @ v_0
        v = v_0 * coef + alpha * e.T @ u_0

        for i in range(n):
            u[i, k+1] = 1

        for j in range(m):
            v[j, k] = 1

        e = r - u @ v.T
        for i, j in etup:
            e[i, j] = 0

        norm_e_0 = np.linalg.norm(e_0, ord="fro") ** 2
        norm_u_0 = np.linalg.norm(u_0, ord="fro") ** 2
        norm_u = np.linalg.norm(u, ord="fro") ** 2
        norm_v_0 = np.linalg.norm(v_0, ord="fro") ** 2
        norm_v = np.linalg.norm(v, ord="fro") ** 2
        norm_e = np.linalg.norm(e, ord="fro") ** 2
        l_0 = norm_e_0 + reg * norm_u_0 + reg * norm_v_0
        l_1 = norm_e + reg * norm_u + reg * norm_v
        norm = np.abs(l_1 - l_0)
        ind += 1
        u_0 = u.copy()
        v_0 = v.copy()
        print("The norm: {:6.5f} || iter: {:d}".format(
            norm, ind
        ))
    return u + mean, v + mean

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
