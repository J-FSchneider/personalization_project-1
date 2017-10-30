# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Import Packages
import numpy as np
import pandas as pd
from model_based import simple_latent_factor_model
from matrix_factorization import un_reg_mf

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Lose entries function
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def lose_entries(a, percentage, seed=42, lost_ind=-700):
    """
    This function takes a given matrix and uniformly at
    random substitutes a defined percentage of its
    entries for the user specified value lost_ind
    :param a: np.array | a given matrix
    :param percentage: float | % of altered entries
    :param seed: int | replicate results
    :param lost_ind: int | value for altered entries
    :return r: np.array | the matrix modified
    """
    n, m = a.shape
    nm = n * m
    np.random.seed(seed)
    # Elongate the matrix
    aux = a.reshape((nm, 1)).copy()
    lost_number = int(np.floor(nm * percentage))
    lost = np.random.randint(low=0,
                             high=nm,
                             size=lost_number)
    aux[lost, 0] = lost_ind
    # Restore the shape of the matrix
    r = aux.reshape(n, m)
    return r

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Empty tuple scan
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def empty(r, lost_ind=-700):
    """
    This function searches throughout a given matrix
    for the values that are equal to lost_ind. Then
    it outputs the (i, j) tuples where the value
    was found.
    This function is used in the matrix factorization
    algorithm to control which entries need to be
    ignored
    :param r: np.array | the given matrix to search
    :param lost_ind: int | the value to look for
    :return empty_set: list | the list of tuples
                        where the value was found
    """
    n, m = r.shape
    empty_set = []
    for i in range(n):
        for j in range(m):
            if r[i, j] == lost_ind:
                empty_set.append((i, j))

    return empty_set

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Test Cases
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Test Case 01 := Identity Blocks


def testcase01(m, n):
    nm = n * m
    identity = np.eye(m, m)
    r = np.zeros((nm, m))

    for i in range(n):
        r[i * m: (i + 1) * m, :] = identity

    return r

# Test Case 02 := Upper Triangular Blocks


def testcase02(m, n):
    nm = n * m
    ones = np.ones((m, m))
    triangular = np.triu(ones)
    r = np.zeros((nm, m))

    for i in range(n):
        r[i * m: (i + 1) * m, :] = triangular

    return r

# Test Case 03 := Randint {0, 1} blocks


def testcase03(m, n, seed=212):
    np.random.seed(seed)
    nm = n * m
    block = np.random.randint(low=0, high=2, size=(m, m))
    r = np.zeros((nm, m))

    for i in range(n):
        r[i * m: (i + 1) * m, :] = block

    return r

# Test Case 04 := all entries belong to Randint {0, 1}


def testcase04(m, n, seed=42):
    nm = n * m
    np.random.seed(seed)
    r = np.random.randint(low=0, high=2, size=(nm, m))
    return r

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# SVD solution
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def svd_approx(a, k):
    """
    This function generates the k-rank approximation
    of a given matrix a
    :param a: np.array | high-dimensional matrix to
                approximate
    :param k: int | the rank of the approximation
    :return r_approx: np.array | the low-rank matrix
    """
    n, m = a.shape
    u, s, v = np.linalg.svd(a)
    s_trun = np.zeros((n, m))
    s_trun[:k, :k] = np.diag(s[:k])
    r_approx = np.dot(u, np.dot(s_trun, v))
    return r_approx

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Test script
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


m = 6
n = 3
per = 0.25
k = 6
lost_ind = -700

np.set_printoptions(
    formatter={'float': lambda x: "{:0.2}".format(x)})

test1 = testcase01(m, n)
test2 = testcase02(m, n)
test3 = testcase03(m, n)
r1 = lose_entries(test1, percentage=per, lost_ind=lost_ind)
r2 = lose_entries(test2, percentage=per, lost_ind=lost_ind)
r3 = lose_entries(test3, percentage=per, lost_ind=lost_ind)
emp_tup1 = empty(r1, lost_ind=lost_ind)
emp_tup2 = empty(r2, lost_ind=lost_ind)
emp_tup3 = empty(r3, lost_ind=lost_ind)
svd1 = svd_approx(test1, k)
svd2 = svd_approx(test2, k)
svd3 = svd_approx(test3, k)
u1, v1 = un_reg_mf(r1, emp_tup1, k=k, reg=0, alpha=0.01, tol=0.000001)
gd1 = u1 @ v1.T
u2, v2 = un_reg_mf(r2, emp_tup2, k=k, reg=0, alpha=0.01, tol=0.000001)
gd2 = u2 @ v2.T
u3, v3 = un_reg_mf(r3, emp_tup3, k=k, reg=0, alpha=0.01, tol=0.000001)
gd3 = u3 @ v3.T
# ub1, vb1 = un_reg_bi_mf(r1, emp_tup1, k=k, reg=0, alpha=0.01, tol=0.001)
# gbd1 = ub1 @ vb1.T
# gdb2 = un_reg_bi_mf(r2, emp_tup2, k=k, reg=0, alpha=0.01, tol=0.001)
# gdb3 = un_reg_bi_mf(r3, emp_tup3, k=k, reg=0, alpha=0.01, tol=0.001)

m1 = r1.copy()
m1 = pd.DataFrame(m1)
for i, j in emp_tup1:
    m1.iloc[i, j] = np.nan
j1, ju1, jv1 = simple_latent_factor_model(m1, latent_factors=k,
                                          learning_rate=0.01)
m2 = r2.copy()
m2 = pd.DataFrame(m2)
for i, j in emp_tup1:
    m2.iloc[i, j] = np.nan
j2, ju2, jv2 = simple_latent_factor_model(m2, latent_factors=k,
                                          learning_rate=0.01)
m3 = r3.copy()
m3 = pd.DataFrame(m3)
for i, j in emp_tup1:
    m3.iloc[i, j] = np.nan
j3, ju3, jv3 = simple_latent_factor_model(m3, latent_factors=k,
                                          learning_rate=0.01)

model_comp1 = np.linalg.norm(j1 - gd1, 2)
print("\nModel Diff Test01: {:6.2f}".format(model_comp1))
model_comp2 = np.linalg.norm(j2 - gd2, 2)
print("\nModel Diff Test02: {:6.2f}".format(model_comp2))
model_comp3 = np.linalg.norm(j3 - gd3, 2)
print("\nModel Diff Test03: {:6.2f}".format(model_comp3))

# bias_comp1 = np.linalg.norm(jb1 - gd1, 2)
# print("\nBias Diff Test01: {:6.2f}".format(bias_comp1))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
