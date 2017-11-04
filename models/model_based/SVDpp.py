# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Import Packages
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import pandas as pd
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
        self.data = None
        self.k = k
        self.algo = SVDpp(n_factors=self.k)
        self.predictions = pd.DataFrame()

    def fit(self, rating_matrix):
        """
        Fits the instance to the rating matrix. The index must be
        the users and the columns the items.
        :param rating_matrix: pd.DataFrame | rating matrix
        :return: void
        """
        data_long = rating_matrix.stack().reset_index()
        data_long.columns = ["user_id", "item_id", "ratings"]

        # Run SVD++
        reader = Reader(rating_scale=(0, 1))
        data = Dataset.load_from_df(data_long, reader)
        trainset = data.build_full_trainset()
        self.algo.train(trainset)
        testset = trainset.build_anti_testset()
        predictions = self.algo.test(testset)

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
        dataframe["ratings"] = ratings
        dataframe["userID"] = users
        self.predictions = dataframe

    def predict(self, user, item):
        """
        Predict the probability that input user will like input item
        :param user: int | user ID
        :param item: int | item ID
        :return: float | probability that user likes item
        """
        cond1 = self.predictions["userID"] == user
        cond2 = self.predictions["itemID"] == item
        mask = cond1 & cond2
        temp = np.array(self.predictions.loc[mask, "ratings"])
        proba = np.sum(temp)
        return proba

# =========================================================================
