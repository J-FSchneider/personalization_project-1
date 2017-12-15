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
# data = data[data["is_listened"] == 1]  # Needed not to filter
# sample = hit_rate_matrix_popular_items(data)
sample = hit_rate_matrix_popular_items(data=data,
                                       n_users=4000,
                                       n_items=500,
                                       min_rating=30)
print(sample.head())
songs = list(sample.columns)
print("The type of the songs")
print(type(songs))
print("The number of songs included {:2d}".format(len(songs)))
users = sample.reset_index()
users = list(users["user_id"].unique())
print("The type of the users")
print(type(users))
print(len(users))
# =========================================================================
mask = (data["user_id"].isin(users)) & (data["media_id"].isin(songs))
db = data[mask]
print(db.shape)
db.to_csv("train_sample02.csv")
print(len(data["user_id"].unique()))
print(len(db["user_id"].unique()))
print("\nThe songs in the orginal data set")
print(len(data["media_id"].unique()))
print("\nThe songs in the sample")
print(len(db["media_id"].unique()))
