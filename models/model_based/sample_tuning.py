# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Import Packages
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import pandas as pd
from models.model_based.data_prep import hit_rate
from models.model_based.data_prep import relevant_elements
from models.model_based.data_prep import prep_test
from models.model_based.data_prep import prep_for_model
from models.model_based.matrix_factorization_model \
    import latent_factors_with_bias
# from models.model_based.SVDpp import SurSVDpp
from models.model_based.SVD import SurSVD

# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load Data
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
filename = "/home/ap/personalization_project/models/model_based/train.csv"
# data = pd.read_csv(filename, nrows=100000)
data = pd.read_csv(filename)
filename = "/home/ap/personalization_project/models/model_based/test.csv"
test = pd.read_csv(filename)
test = prep_test(test)
np.set_printoptions(precision=2, suppress=True)
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Data Prep
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Add the hit rate to the data
data = hit_rate(data)
data_filtered = data[data["listen_type"] == 1]

# Define Thresholds
thres = 0.25
user_threshold = 5000

# Filter the data for relevant users
data_users = data[data["user_id"] <= user_threshold]

# Acquire the most relevant songs
songs_list = relevant_elements(data_users, "media_id", threshold=thres)
print("\nThis is the number of relevant songs")
print(len(songs_list))

# Filter the data to contain the most relevant songs
data_users = data_users[data_users["media_id"].isin(songs_list)]

# Trim data based to only columns needed
df = prep_for_model(data_users)
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Run Jan's Model against this data set
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
k = 5
ratings_matrix = df.pivot(index="userID",
                          columns="itemID",
                          values="ratings")
matrix = ratings_matrix.copy()
pred, _, _ = latent_factors_with_bias(ratings_matrix,
                                      latent_factors=k,
                                      bias=0.5,
                                      bias_weights=0.25,
                                      regularization=0.6,
                                      learning_rate=0.0001,
                                      convergence_rate=0.99999)
n, m = ratings_matrix.shape
for i in range(n):
    for j in range(m):
        matrix.iloc[i, j] = pred[i, j]

predictions = matrix.stack().reset_index()
predictions.columns = ["userID", "itemID", "ratings"]

# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compare results with test set provided by Dreezer
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
result = pd.merge(test, predictions,
                  how="inner",
                  on=["userID", "itemID"])
result["predictions"] = 0
result.loc[result["ratings"] > 0.5, "predictions"] = 1
print("\nBelow are the merge of the results")
print(result.head())
print("\nThis is the number of (users, songs) that matched for Jan")
print(len(result))
print("\nBelow is the number of songs that were missclasified for Jan")
print(np.linalg.norm(result["ratings_org"] - result["predictions"], ord=1))
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Run SVD model for Dreezer's comparison
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
k = 5
Sur = SurSVD(k=k)
Sur.fit_directly(df)
predictions = Sur.predictions
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compare results with test set provided by Dreezer
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
result = pd.merge(test, predictions,
                  how="inner",
                  on=["userID", "itemID"])
result["predictions"] = 0
result.loc[result["ratings"] > 0.5, "predictions"] = 1
print("\nBelow are the merge of the results")
print(result.head())
print("\nThis is the number of (users, songs) that matched")
print(len(result))
print("\nBelow is the number of songs that were missclasified")
print(np.linalg.norm(result["ratings_org"] - result["predictions"], ord=1))
# =========================================================================
