# import numpy as np
import pandas as pd
from matrix_creation import sample_data_by_freq
from surprise import SVDpp
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import evaluate

filename = "train.csv"
# Read part of the file
df = pd.read_csv(filename, nrows=100000)

# Read all file completely
# df = pd.read_csv(filename)

print(df.head())

user, media = sample_data_by_freq(data=df)

users = df[df["user_id"].isin(user)]

items = users[users["media_id"].isin(media)]

# TODO: need to verify that this alternative makes sense
matrix = items.drop_duplicates(
    subset=['media_id', 'user_id'], keep='first')

matrix = matrix.rename(columns={"user_id": "userID",
                                "media_id": "itemID",
                                "is_listened": "ratings"})

df = matrix[["userID", "itemID", "ratings"]]

reader = Reader(rating_scale=(0, 1))

data = Dataset.load_from_df(df, reader)

algo = SVDpp()
# trainset = data.build_full_trainset()
# algo.train(trainset)
# testset = trainset.build_anti_testset()
# predictions = algo.test(testset)
#
#
# def predictions_df(pred):
#     users = []
#     items = []
#     ratings = []
#     dataframe = pd.DataFrame()
#     for uid, iid, r_ui, _, _ in pred:
#         users.append(uid)
#         items.append(iid)
#         ratings.append(r_ui)
#
#     dataframe["itemID"] = items
#     dataframe["rating"] = ratings
#     dataframe["userID"] = users
#     return dataframe
#
# dfpred = predictions_df(predictions)

perf = evaluate(algo, data, measures=["RMSE", "MAE"])

print(perf)

algo = SVD()
perf = evaluate(algo, data, measures=["RMSE", "MAE"])
print(perf)
