# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Import Packages
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import pandas as pd
from matrix_factorization import un_reg_mf
# from matrix_factorization_model import latent_factors_with_bias
from surprise import SVD
from surprise import Reader
from surprise import Dataset

# =========================================================================

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Lose entries function
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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

# =========================================================================

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Empty tuple scan
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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

# =========================================================================

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Test Cases
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

# =========================================================================

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# SVD solution
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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

# =========================================================================

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# TEST SCRIPT
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Parameters
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


M = 6
N = 4
per = 0
K = 6
lost_int = -700
alph = 0.01
tole = 1e-10
conv_rate = 0.5

np.set_printoptions(
    formatter={'float': lambda x: "{:0.2}".format(x)})

# =========================================================================

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Construct tests and SVD approximations
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

test1 = testcase01(M, N)
test2 = testcase02(M, N)
test3 = testcase03(M, N)
r1 = lose_entries(test1, percentage=per, lost_ind=lost_int)
r2 = lose_entries(test2, percentage=per, lost_ind=lost_int)
r3 = lose_entries(test3, percentage=per, lost_ind=lost_int)
emp_tup1 = empty(r1, lost_ind=lost_int)
emp_tup2 = empty(r2, lost_ind=lost_int)
emp_tup3 = empty(r3, lost_ind=lost_int)
svd1 = svd_approx(test1, K)
svd2 = svd_approx(test2, K)
svd3 = svd_approx(test3, K)

# =========================================================================

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Run AP Latent Factor Model
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

u1, v1 = un_reg_mf(r1, emp_tup1, k=K, reg=0, alpha=0.01, tol=tole)
gd1 = u1 @ v1.T
u2, v2 = un_reg_mf(r2, emp_tup2, k=K, reg=0, alpha=0.01, tol=tole)
gd2 = u2 @ v2.T
u3, v3 = un_reg_mf(r3, emp_tup3, k=K, reg=0, alpha=0.01, tol=tole)
gd3 = u3 @ v3.T

# =========================================================================

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Tests | Convert Numpy Array to Pandas DataFrame
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

m1 = r1.copy()
m1 = pd.DataFrame(m1)
for q, rr in emp_tup1:
    m1.iloc[q, rr] = np.nan
mlong1 = m1.stack()
mlong1 = mlong1.reset_index()
mlong1.columns = ["userID", "itemID", "ratings"]

m2 = r2.copy()
m2 = pd.DataFrame(m2)
for q, rr in emp_tup1:
    m2.iloc[q, rr] = np.nan
mlong2 = m2.stack()
mlong2 = mlong2.reset_index()
mlong2.columns = ["userID", "itemID", "ratings"]

m3 = r3.copy()
m3 = pd.DataFrame(m3)
for q, rr in emp_tup1:
    m3.iloc[q, rr] = np.nan
mlong3 = m3.stack()
mlong3 = mlong3.reset_index()
mlong3.columns = ["userID", "itemID", "ratings"]

# =========================================================================

# #########################################################################
# Extract predictions from Scikit-Surprise
# #########################################################################


def predictions_df(pred):
    users = []
    items = []
    ratings = []
    dataframe = pd.DataFrame()
    for uid, iid, r_ui, _, _ in pred:
        users.append(uid)
        items.append(iid)
        ratings.append(r_ui)

    dataframe["itemID"] = items
    dataframe["userID"] = users
    dataframe["ratings"] = ratings
    return dataframe

# =========================================================================

# #########################################################################
# Tests against Scikit-Surprise
# #########################################################################


reader = Reader(rating_scale=(0, 1))
algo = SVD(n_factors=K,
           n_epochs=100,
           biased=False,
           reg_all=0,
           lr_all=alph,
           verbose=False)

data = Dataset.load_from_df(mlong1, reader)
trainset = data.build_full_trainset()
algo.train(trainset)
testset = trainset.build_anti_testset()
predictions = algo.test(testset)
dfpred1 = predictions_df(predictions)
df1 = pd.concat([mlong1, dfpred1])
df1 = pd.DataFrame(df1)
df1 = df1.pivot(index="userID", columns="itemID", values="ratings")
num1 = np.array(df1)

data = Dataset.load_from_df(mlong2, reader)
trainset = data.build_full_trainset()
algo.train(trainset)
testset = trainset.build_anti_testset()
predictions = algo.test(testset)
dfpred2 = predictions_df(predictions)
df2 = pd.concat([mlong2, dfpred2])
df2 = pd.DataFrame(df2)
df2 = df2.pivot(index="userID", columns="itemID", values="ratings")
num2 = np.array(df2)

data = Dataset.load_from_df(mlong3, reader)
trainset = data.build_full_trainset()
algo.train(trainset)
testset = trainset.build_anti_testset()
predictions = algo.test(testset)
dfpred3 = predictions_df(predictions)
df3 = pd.concat([mlong3, dfpred3])
df3 = pd.DataFrame(df3)
df3 = df3.pivot(index="userID", columns="itemID", values="ratings")
num3 = np.array(df3)

# =========================================================================

# #########################################################################
# Norm Comparison AP vs Scikit-Surprise
# #########################################################################

model_comp1 = np.linalg.norm(num1 - gd1, 2)
print("\nModel Diff Test01: {:6.2f}".format(model_comp1))
model_comp2 = np.linalg.norm(num2 - gd2, 2)
print("\nModel Diff Test02: {:6.2f}".format(model_comp2))
model_comp3 = np.linalg.norm(num3 - gd3, 2)
print("\nModel Diff Test03: {:6.2f}".format(model_comp3))

model_comp1 = np.linalg.norm(svd1 - gd1, 2)
print("\nAP Diff SVD01: {:6.2f}".format(model_comp1))
model_comp2 = np.linalg.norm(svd2 - gd2, 2)
print("\nAP Diff SVD02: {:6.2f}".format(model_comp2))
model_comp3 = np.linalg.norm(svd3 - gd3, 2)
print("\nAP Diff SVD03: {:6.2f}".format(model_comp3))

model_comp1 = np.linalg.norm(svd1 - num1, 2)
print("\nSur Diff SVD01: {:6.2f}".format(model_comp1))
model_comp2 = np.linalg.norm(svd2 - num2, 2)
print("\nSur Diff SVD02: {:6.2f}".format(model_comp2))
model_comp3 = np.linalg.norm(svd3 - num3, 2)
print("\nSur Diff SVD03: {:6.2f}".format(model_comp3))

# =========================================================================

# #########################################################################
# Tests against Jan's models
# #########################################################################
# j1, ju1, jv1 = latent_factors_with_bias(m1,
#                                         latent_factors=K,
#                                         bias=None,
#                                         bias_weights=None,
#                                         regularization=0,
#                                         learning_rate=alph,
#                                         convergence_rate=conv_rate)
#
# model_comp1 = np.linalg.norm(j1 - num1, 2)
# print("\nJan Diff Sur01: {:6.2f}".format(model_comp1))
# =========================================================================
