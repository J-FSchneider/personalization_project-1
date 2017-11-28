# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Import Packages
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import pandas as pd
from utils.loss_functions import mean_squared_error
from models.cross_validation.ModelTester import ModelTester
from models.cross_validation.parameter_test import parameter_test
from models.cross_validation.parameter_test import ready_to_plot
from models.model_based.data_prep import hit_rate
from models.model_based.data_prep import relevant_elements
from models.model_based.data_prep import prep_for_model
from models.model_based.matrix_creation import hit_rate_matrix_popular_items
# from models.model_based.matrix_factorization_model \
#     import latent_factors_with_bias
from models.model_based.SVD import SurSVD
# from models.model_based.SVDpp import SurSVDpp
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load Data
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
filename = "/home/ap/personalization_project/models/model_based/train.csv"
# data_org = pd.read_csv(filename, nrows=10000)
data_org = pd.read_csv(filename)
np.set_printoptions(precision=2, suppress=True)
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Data Prep
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Add the hit rate to the data
data = hit_rate(data_org)
data_filtered = data[data["listen_type"] == 1]

# Define Thresholds
thres = 0.1
user_threshold = 5000
print("\nRunning for {:d} users".format(user_threshold))

# Filter the data for relevant users
data_users = data[data["user_id"] <= user_threshold]

# Acquire the most relevant songs
songs_list = relevant_elements(data_users, "media_id", threshold=thres)
print("\nThis is the number of relevant songs\n")
print(len(songs_list))

# Filter the data to contain the most relevant songs
data_users = data_users[data_users["media_id"].isin(songs_list)]

# Trim data based to only columns needed
df = prep_for_model(data_users)
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Set parameters for tests
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ratings_matrix = df.pivot(index="userID",
                          columns="itemID",
                          values="ratings")
k = 5
# k_values = [1, 2, 5, 7, 10, 50, 75, 100, 200, 250, 500]
# k_values = [5, 50, 100]
# k_values = [2, 5, 10, 100]
# k_values = [2, 5]
k_values = [2]
cv_times = 1
ratios = (0.7, 0.1, 0.2)
model_tester = ModelTester(ratios=ratios, model_based=True, seed=42)
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Run Jan's Model against this data set
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# matrix = ratings_matrix.copy()
# pred, _, _ = latent_factors_with_bias(ratings_matrix,
#                                       latent_factors=k,
#                                       bias=0.5,
#                                       bias_weights=0.25,
#                                       regularization=0.6,
#                                       learning_rate=0.0001,
#                                       convergence_rate=0.99999)
# n, m = ratings_matrix.shape
# for i in range(n):
#     for j in range(m):
#         matrix.iloc[i, j] = pred[i, j]
#
# predictions = matrix.stack().reset_index()
# predictions.columns = ["userID", "itemID", "ratings"]
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Perform Tests with SVD with most relevant songs for the most relevant
# users
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d_test, d_train = parameter_test(k_val=k_values,
                                 cv_times=cv_times,
                                 model=SurSVD,
                                 loss_function=mean_squared_error,
                                 model_tester=model_tester,
                                 data=ratings_matrix,
                                 verbose=False)
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Print Results SVD
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dftest = ready_to_plot(d_test)
dftrain = ready_to_plot(d_train)
print("\nBelow is the result for the test set most relevant")
print(dftest)
print("\nBelow is the result for the train set most relevant")
print(dftrain)
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Perform Tests with SVD for 100 top songs
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ratings_matrix = hit_rate_matrix_popular_items(data_org)
d_test, d_train = parameter_test(k_val=k_values,
                                 cv_times=cv_times,
                                 model=SurSVD,
                                 loss_function=mean_squared_error,
                                 model_tester=model_tester,
                                 data=ratings_matrix,
                                 verbose=False)
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Print Results SVD for 100 top songs
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dftest = ready_to_plot(d_test)
dftrain = ready_to_plot(d_train)
print("\nBelow is the result for the test set 100 top songs")
print(dftest)
print("\nBelow is the result for the train set 100 top songs")
print(dftrain)
# =========================================================================
