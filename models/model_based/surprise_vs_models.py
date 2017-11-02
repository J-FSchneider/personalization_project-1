# Import Packages
import numpy as np
import pandas as pd
from tests import testcase01
from tests import lose_entries
from tests import empty
from matrix_factorization_model import simple_latent_factor_model
from surprise import SVDpp
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import evaluate

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Parameters
# ===============================================================
m = 3
n = 2
u = m * n
rango = u * m
per = 0.25
lost = -700

test1 = testcase01(m, n)

t_1 = lose_entries(test1, per, lost_ind=lost)
empt_1 = empty(t_1, lost_ind=lost)
ret1 = t_1.reshape(rango, )

array_1 = np.zeros((rango, 3))

array_1[:, 0] = ret1

for i in range(u):
    for j in range(m):
        array_1[i * j, 1] = i
        array_1[i * j, 2] = j

df_1 = pd.DataFrame()
