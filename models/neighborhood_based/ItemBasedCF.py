import numpy as np
from utils.similarity import ochiai


class ItemBasedCF():
    def __init__(self, k=5, sim_measure=ochiai):
        if not isinstance(k, int) or k <= 0:
            raise IOError("Parameter k should be a positive integer.")
        self.users = []
        self.items = []
        self.memory = {}
        self.data = None
        self.k = k
        self.sim_measure = sim_measure

    def fit(self, rating_matrix):
        """
        Fits the instance to the rating matrix. The index must be the users and
        the columns the items.
        :param rating_matrix: pd.DataFrame | rating matrix
        :return: void
        """
        self.users = rating_matrix.index.values
        self.items = rating_matrix.columns.values
        self.data = rating_matrix

    def get_similar_items(self, item):
        """
        Get the similar items of a given input item.
        The similarity used is the self.sim_measure function
        :param item: int | item ID
        :return: list of tuples | list of tuples being defined as
                                  (similarity coefficient, item ID)
        """
        if item not in self.items:
            raise IOError("Item not available in data.")

        if item not in self.memory:
            neighbors = [(self.sim_measure(self.data[item], self.data[other]), other)
                         for other in self.items if other != item]
            self.memory[item] = neighbors
        else:
            neighbors = self.memory[item]

        top_k_neighbors = sorted(neighbors, key=lambda x: x[0])[-self.k:]

        return top_k_neighbors

    def predict(self, user, item):
        """
        Predict the probability that input user will like input item
        :param user: int | user ID
        :param item: int | item ID
        :return: float | probability that user likes item
        """
        if user not in self.users:
            raise IOError("User not available in data.")

        # Get neighbors of item
        item_neighbors = self.get_similar_items(item)
        # Get the ratings given by the user to the neighbors of item
        # May contain NaN
        user_ratings_neighbors = [self.data.loc[user][neighb[1]] for neighb in
                                  item_neighbors]

        item_rating = list(
            zip([x[0] for x in item_neighbors], user_ratings_neighbors))

        # Get rating for item "item"
        weighted_ratings = 0
        sum_abs_coeff = 0
        size = 0

        for coeff, rating in item_rating:
            if np.isnan(rating):
                size -= 1
                continue
            else:
                weighted_ratings += coeff * rating
                sum_abs_coeff += abs(coeff)

        # Checks if all ratings are NaN, then input the user mean rating
        # in the original rating matrix
        # TODO: print warning
        if size == 0:
            return np.nanmean(self.data.loc[user])

        return weighted_ratings / sum_abs_coeff

    def recommendations(self, user):
        """top k/m items for te user"""
        # TODO: implement
        pass
