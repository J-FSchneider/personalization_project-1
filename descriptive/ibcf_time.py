# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Imports
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import pandas as pd
# from utils.Pipeline import Pipeline
from models.model_based.matrix_creation import hit_rate_matrix_popular_items
from models.neighborhood_based.ItemBasedCF import ItemBasedCF
from models.cross_validation.ModelTester import ModelTester
from models.cross_validation.parameter_test import parameter_test
import utils.loss_functions as lf
from utils.preprocessing import get_moment_of_day2
# from utils.similarity import ochiai
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load data
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
a = pd.DataFrame()  # to avoid "error message" of not using pandas
# file = "./descriptive/db_nrows.csv"
# file = "./descriptive/train_sample.csv"
# file = "./descriptive/train_sample02.csv"
file = "./descriptive/train.csv"

# data_file = "./descriptive/db_nrows.csv"
# data_file = "./descriptive/train.csv"
# data_file = "./descriptive/train_sample02.csv"
# data_file = "./descriptive/train_sample.csv"
# spotify_file = "./descriptive/SpotifyAudioFeatures_clean.csv"
# sample_path = "./descriptive/train_sample.csv"
# pipe = Pipeline(deezer_path=data_file,
#                 spotify_path=spotify_file,
#                 sample_path=sample_path,
#                 use_sample=False)
# df = pipe.make_selected_few()
# print(df.shape)
df = pd.read_csv(file)
df["moment_of_day"] = df["ts_listen"].apply(get_moment_of_day2)
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Do one ran per time of day approach
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
user = "user_id"
songs = "media_id"
y = "is_listened"
# k_vals = [10, 20]
k_vals = [25]
cv_times = 1
print("\nRunning the model wo tod segmentation")
matrix = hit_rate_matrix_popular_items(data=df)
# matrix = data.pivot_table(index=user,
#                           columns=songs,
#                           values=y,
#                           aggfunc="mean")
result_test, result_train = parameter_test(k_vals,
                                           ItemBasedCF,
                                           ModelTester(),
                                           lf.mean_squared_error,
                                           matrix,
                                           cv_times,
                                           verbose=True)

tod_list = ['morning', 'afternoon_evening', 'late_night']
for tod in tod_list:
    print("\nCurrently on: "+tod)
    data = df[df["moment_of_day"] == tod].copy()
    data = data[[user, songs, y]]
    print(data.head())
    matrix = hit_rate_matrix_popular_items(data=data)
    # Below is the code in case that we want to use Pipeline
    # matrix = data.pivot_table(index=user,
    #                           columns=songs,
    #                           values=y,
    #                           aggfunc="mean")
    _, _ = parameter_test(k_vals,
                          ItemBasedCF,
                          ModelTester(),
                          lf.mean_squared_error,
                          matrix,
                          cv_times,
                          verbose=True)
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Do gigant matrix approach
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# df.is_copy = False
# df = df.reset_index()
# df["user_time"] = df["user_id"].astype(str)+"_&_"+df["moment_of_day"]
# df["user_time_id"] = df.index
# print(df.head())
# user = "user_time_id"
# songs = "media_id"
# y = "is_listened"
# data = df[[user, songs, y]]
# print(data.head())
# matrix = data.pivot_table(index=user,
#                           columns=songs,
#                           values=y,
#                           aggfunc="mean")
# print(matrix.head())
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Evaluate the model
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# k_vals = [25]
# cv_times = 1
# result_test, result_train = parameter_test(k_vals,
#                                            ItemBasedCF,
#                                            ModelTester(),
#                                            lf.mean_squared_error,
#                                            matrix,
#                                            cv_times,
#                                            verbose=True)
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Run IBCF model
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ibcf = ItemBasedCF(k=15, sim_measure=ochiai)
# ibcf.fit(matrix)
# print("\nThe prediction for user 44 in the morning is")
# print(ibcf.predict(user=0, item=876498))
# =========================================================================
