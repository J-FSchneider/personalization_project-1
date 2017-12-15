# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Imports
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import pandas as pd
from utils.Pipeline import Pipeline
from models.neighborhood_based.ItemBasedCF import ItemBasedCF
# from utils.similarity import ochiai
from models.cross_validation.ModelTester import ModelTester
from models.cross_validation.parameter_test import parameter_test
import utils.loss_functions as lf
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load data
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# data_file = "./descriptive/db_nrows.csv"
# data_file = "./descriptive/train.csv"
spotify_file = "./descriptive/SpotifyAudioFeatures_clean.csv"
sample_path = "./descriptive/train_sample.csv"
data_file = "./descriptive/train_sample02.csv"
pipe = Pipeline(deezer_path=data_file,
                spotify_path=spotify_file,
                sample_path=sample_path,
                use_sample=False)
a = pd.DataFrame()  # to avoid "error message" of not using pandas
df = pipe.make_selected_few()
print(df.shape)
df.is_copy = False
df = df.reset_index()
df["user_time"] = df["user_id"].astype(str)+"_&_"+df["moment_of_day"]
df["user_time_id"] = df.index
print(df.head())
user = "user_time_id"
songs = "media_id"
y = "is_listened"
data = df[[user, songs, y]]
print(data.head())
matrix = data.pivot_table(index=user,
                          columns=songs,
                          values=y,
                          aggfunc="mean")
print(matrix.head())
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Run IBCF model
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ibcf = ItemBasedCF(k=15, sim_measure=ochiai)
# ibcf.fit(matrix)
# print("\nThe prediction for user 44 in the morning is")
# print(ibcf.predict(user=0, item=876498))
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Evaluate the model
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
k_vals = [5, 10]
cv_times = 1
result_test, result_train = parameter_test(k_vals,
                                           ItemBasedCF,
                                           ModelTester(),
                                           lf.mean_squared_error,
                                           matrix,
                                           cv_times,
                                           verbose=True)
# =========================================================================
