# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Imports
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import pandas as pd
import numpy as np
from models.tars.context_lf import ContextLF
from utils.preprocessing import get_moment_of_day2
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load data
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
a = pd.DataFrame()  # to avoid "error message" of not using pandas
file = "./descriptive/db_nrows.csv"
# file = "./descriptive/train_sample.csv"
# file = "./descriptive/train.csv"
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
print(df.head())
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Run TARS model
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Note:: the model is simply creating TimeOfDay slices internally
context_lf = ContextLF(data=df,
                       latent_factors=5,
                       context_id="moment_of_day",
                       ratings="is_listened")
# Note:: the fit method does not currently return the matrices
context_lf.fit()
# Note:: also the matrix U is created as a list, when it ends
# Note:: when it ends up being a np.array after the model is ran
U = context_lf.U
print("\nThe type of U is")
print(type(U))
V = context_lf.V
print("\nThe type of V is")
print(type(V))
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Evaluate the performance of the model
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ...
# =========================================================================
