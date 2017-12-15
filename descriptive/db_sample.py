# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Imports
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import pandas as pd
from models.model_based.matrix_creation import hit_rate_matrix_popular_items
# =========================================================================
# data_file = "db_nrows.csv"
# data_file = "./descriptive/db_nrows.csv"
data_file = "train.csv"
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Create sample
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data = pd.read_csv(data_file)
data = data[data["is_listened"] == 1]
sample = hit_rate_matrix_popular_items(data)
print(sample.head())
songs = list(sample.columns)
print(len(songs))
print("The type of the songs")
print(type(songs))
users = sample.reset_index()
users = list(users["user_id"].unique())
print("The type of the users")
print(type(users))
print(len(users))
# =========================================================================
mask = (data["user_id"].isin(users)) & (data["media_id"].isin(songs))
db = data[mask]
print(db.shape)
db.to_csv("train_sample.csv")
print(len(data["user_id"].unique()))
print(len(db["user_id"].unique()))
print(len(data["media_id"].unique()))
print(len(db["media_id"].unique()))
