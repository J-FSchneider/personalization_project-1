# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Import Packages
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# import numpy as np
import pandas as pd
# from data_prep import hit_rate
# from data_prep import song_rank
from surprise import SVDpp
from surprise import Dataset
from surprise import Reader

# =========================================================================

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Define Class
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class SurSVDpp:
    def __init__(self, k=5):
        if not isinstance(k, int) or k <= 0:
            raise IOError("Parameter k should be a positive integer.")
        self.algo = SVDpp()
        self.data = None
        self.predict = pd.DataFrame()
        self.k = k

    def fit(self, rating_matrix):
        """
        Fits the instance to the rating matrix. The index must be the users and
        the columns the items.
        :param rating_matrix: pd.DataFrame | rating matrix
        :return: void
        """
        data_long = rating_matrix.stack().reset_index()
        data_long.columns = ["user_id", "item_id", "ratings"]

        # Run SVD++
        algo = SVDpp(n_factors=self.k)
        reader = Reader(rating_scale=(0, 1))
        data = Dataset.load_from_df(data_long, reader)
        trainset = data.build_full_trainset()
        algo.train(trainset)
        testset = trainset.build_anti_testset()
        predictions = algo.test(testset)

        # Reconstruct predictions
        users = []
        items = []
        ratings = []
        dataframe = pd.DataFrame()
        for uid, iid, r_ui, _, _ in predictions:
            users.append(uid)
            items.append(iid)
            ratings.append(r_ui)

        dataframe["itemID"] = items
        dataframe["rating"] = ratings
        dataframe["userID"] = users
        self.predict = dataframe

    def predict(self, user, item):
        """
        Predict the probability that input user will like input item
        :param user: int | user ID
        :param item: int | item ID
        :return: float | probability that user likes item
        """
        proba = self.predict.iloc[user, item]
        return proba
