import numpy as np


class ModelTester():
    def __init__(self, ratios=(0.7, 0.2, 0.1)):
        """
        Constructor of the class
        :param ratios: 3-tuple | ratios of train, validation and test sets
        """
        # TODO: add some sanity checks
        self.ratios = ratios
        self.valid_set = {}
        self.test_set = {}
        self.non_null_indices = []
        self.train = None

    def fit(self, data):
        """
        Fits the instance to the data
        :param data: pd.DataFrame | rating dataframe
        :return: void
        """
        self.train = data
        # get the indices of the non null (not NaN) values
        self.non_null_indices = list(
            self.train[self.train.notnull()].stack().index)

    def transform(self):
        """
        Fills up the attributes of the instance and transforms in-place
        the rating matrix
        :return: void
        """

        train_ratio, valid_ratio, test_ratio = self.ratios

        # Shuffle the non_null_indices
        shuffled = np.random.shuffle(self.non_null_indices)
        # Get the indices for validation and test sets
        valid_indices = shuffled[int(len(self.non_null_indices) * train_ratio):
                                 int(len(self.non_null_indices) * (train_ratio + valid_ratio))]
        test_indices = shuffled[int(
            len(self.non_null_indices) * (train_ratio + valid_ratio)) + 1:]

        # Fills the attribute dictionaries
        self.valid_set = {(u, i): self.train.loc[u][i] for (u, i) in
                          valid_indices}
        self.test_set = {(u, i): self.train.loc[u][i] for (u, i) in test_indices}

        # Replace original values in DataFrame data by np.nan values
        # TODO: add sanity checks (the right number of values has been replaced)
        for u, i in self.valid_set:
            self.train.loc[u][i] = np.nan
        for u, i in self.test_set:
            self.train.loc[u][i] = np.nan

    def fit_transform(self, data):
        """
        Runs fit and then transform functions
        :param data:
        :return:
        """
        print(">>> The cross-validation framework is being built ...\n"
              "    Please wait, you'll be able to use all of its features soon!")

        self.fit(data)
        self.transform()

        print(">>> DONE")

    def evaluate_test(self, predictions, loss_func):
        """
        Loss function applied to predictions and to hidden test ratings
        :param predictions: dict | dictionary containing the predictions
                                   of your model
        :param loss_func: func | loss function to use
        :return: float | loss value on the test set
        """
        # Check that the predictions are here for all the (u, i) in test_set
        for key in self.test_set:
            if key not in predictions:
                raise IOError("The predictions do not contain all "
                              "the (user, item) combinations of the test set")

        # TODO: to implement
        pass

    def evaluate_valid(self, predictions, loss_func):
        """
        Loss function applied to predictions and to hidden validation ratings
        :param predictions: dict | dictionary containing the predictions
                                   of your model
        :param loss_func: func | loss function to use
        :return: float | loss value on the validation set
        """
        # Check that the predictions are here for all the (u, i) in test_set
        for key in self.valid_set:
            if key not in predictions:
                raise IOError("The predictions do not contain all "
                              "the (user, item) combinations of the "
                              "validation set")

        # TODO: to implement
        pass
