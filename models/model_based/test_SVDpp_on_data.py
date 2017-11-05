# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Import Packages
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# import numpy as np
import pandas as pd
from utils.loss_functions import mean_squared_error
from models.model_based.SVDpp import SurSVDpp
from models.model_based.SVD import SurSVD
from models.model_based.matrix_creation \
    import hit_rate_matrix_popular_items
from models.cross_validation.ModelTester import ModelTester
from models.cross_validation.parameter_test import parameter_test
from models.cross_validation.parameter_test import ready_to_plot
# =========================================================================

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load Data
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# filename = "train.csv"
filename = "/home/ap/personalization_project/models/model_based/train.csv"
data = pd.read_csv(filename, nrows=100000)
# data = pd.read_csv(filename)
print(data.head())

# =========================================================================

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Sample Data
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Do the hit rate sampling schema
# TODO: align if it is necessary to take out the users as Jan opted

ratings_matrix = hit_rate_matrix_popular_items(data=data)
print("\nBelow is a snapshot of the ratings_matrix")
size = 5
print(ratings_matrix.iloc[:size, :size])

# =========================================================================

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Set parameters for tests
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
k_values = [1, 2, 5, 7, 10, 50, 75, 100, 200, 250, 500]
# k_values = [2, 5, 10]
# k_values = [5, 50, 100]
cv_times = 3
ratios = (0.6, 0.1, 0.3)
model_tester = ModelTester(ratios=ratios, model_based=True, seed=42)


# =========================================================================

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Perform Tests with SVD++ model
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d_test, d_train = parameter_test(k_val=k_values,
                                 cv_times=cv_times,
                                 model=SurSVDpp,
                                 loss_function=mean_squared_error,
                                 model_tester=model_tester,
                                 data=ratings_matrix,
                                 verbose=False)

# =========================================================================

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Print Results SVD++
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dftest = ready_to_plot(d_test)
dftrain = ready_to_plot(d_train)

print("\nBelow is the result for the test set")
print(dftest)
print("\nBelow is the result for the train set")
print(dftrain)

# =========================================================================

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Perform Tests with SVD
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

print("\nBelow is the result for the test set")
print(dftest)
print("\nBelow is the result for the train set")
print(dftrain)

# =========================================================================