# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Import Packages
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import time
import numpy as np
import pandas as pd
from models.neighborhood_based.ItemBasedCF import ItemBasedCF
from models.model_based.matrix_creation import hit_rate_matrix_popular_items
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load Data
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
filename = "/home/ap/personalization_project/models/model_based/train.csv"
data_org = pd.read_csv(filename, nrows=1000)
# data_org = pd.read_csv(filename)
np.set_printoptions(precision=2, suppress=True)
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Prep Data
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
matrix = hit_rate_matrix_popular_items(data_org)
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Fit the Neighborhood Model
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ibcf = ItemBasedCF(k=40)
ibcf.fit(matrix)
pred_matrix = matrix.copy()
start = time.time()
n, m = matrix.shape
for i in matrix.columns:
    for u in matrix.index:
        pred_matrix.loc[u, i] = ibcf.predict(u, i)
end = time.time()
print(end - start)


predictions = pred_matrix.stack().reset_index()
predictions.columns = ["userID", "itemID", "pred"]
predictions["ratings_pred"] = 0
predictions.loc[predictions["pred"] > 0.5, "ratings_pred"] = 1

original = matrix.stack().reset_index()
original.columns = ["userID", "itemID", "hit_rate"]
original["ratings"] = 0
original.loc[original["hit_rate"] > 0.5, "ratings"] = 1
results = pd.merge(original, predictions,
                   how="inner", on=["userID", "itemID"])
results["diff"] = 1 - np.abs(results["ratings"] - results["ratings_pred"])
df = results.groupby(by="userID")["diff"].mean()
df = df.reset_index()
df.columns = ["userID","pred_hr"]

print("\nBelow are the results")
print(results.head())
print("\nBelow is a summary of the results")
print(df.head())