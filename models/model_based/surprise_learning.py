# Surprise Module checks
import pandas as pd
from surprise import SVDpp
from surprise import Dataset
# from surprise import evaluate
from surprise import Reader

# data = Dataset.load_builtin("ml-100k")
# data.split(n_folds=3)
# algo = SVD()
# perf = evaluate(algo, data, measures=["RMSE", "MAE"])
#
# print(perf)

ratings_dict = {"itemID": [1, 1, 1, 2, 2],
                "userID": [9, 32, 2, 45, 20],
                "rating": [3, 2, 4, 3, 1]}

df = pd.DataFrame(ratings_dict)

reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(df[["userID",
                               "itemID",
                                "rating"]],
                            reader)

trainset = data.build_full_trainset()
# algo = SVD()
algo = SVDpp()
algo.train(trainset)
testset = trainset.build_anti_testset()
predictions = algo.test(testset)


def predictions_df(pred):
    users = []
    items = []
    ratings = []
    dataframe = pd.DataFrame()
    for uid, iid, r_ui, _, _ in pred:
        users.append(uid)
        items.append(iid)
        ratings.append(r_ui)

    dataframe["itemID"] = items
    dataframe["rating"] = ratings
    dataframe["userID"] = users
    return dataframe

dfpred = predictions_df(predictions)
