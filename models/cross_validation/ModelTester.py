import numpy as np
from loss_functions import mean_squared_error, absolute_mean_error




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
        self.train_set = {}
        self.non_null_indices = []
        self.data = None

    def fit(self, data):
        """
        Fits the instance to the data
        :param data: pd.DataFrame | rating dataframe
        :return: void
        """
        self.data = data

        # get the indices of the non null (not NaN) values
        self.non_null_indices = list(
            self.data[self.data.notnull()].stack().index)

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
        train_indices = shuffled[:int(len(self.non_null_indices) * train_ratio)]
        valid_indices = shuffled[int(len(self.non_null_indices) * train_ratio):
                                 int(len(self.non_null_indices) * (train_ratio + valid_ratio))]
        test_indices = shuffled[int(
            len(self.non_null_indices) * (train_ratio + valid_ratio)) + 1:]

        # Fills the attribute dictionaries
        self.train_set = {(u, i): self.data.loc[u][i] for (u, i) in train_indices}
        self.valid_set = {(u, i): self.data.loc[u][i] for (u, i) in valid_indices}
        self.test_set = {(u, i): self.data.loc[u][i] for (u, i) in test_indices}


        # Replace original values in DataFrame data by np.nan values
        for u, i in self.valid_set:
            self.data.loc[u][i] = np.nan
        for u, i in self.test_set:
            self.data.loc[u][i] = np.nan

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
        :param predictions: dict, df | dictionary, dataframe containing the predictions
                                   of your model
        :param loss_func: func | loss function to use
        :return: float | loss value on the test set
        """
        # transform prediction df into dictionary
        if type(data) == pd.core.frame.DataFrame:
                non_null_indices = list(predictions[predictions.notnull()].stack().index)
                pred = {(u, i): predictions.loc[u][i] for (u, i) in
                          non_null_indices}
                predictions = pred

        # Check that the predictions are here for all the (u, i) in test_set
        for key in self.test_set:
            if key not in predictions:
                raise IOError("The predictions do not contain all "
                              "the (user, item) combinations of the test set")

        # Calculate Loss
        loss = loss_func(predictions, self.test_set)

        return loss 

    def evaluate_valid(self, predictions, loss_func):
        """
        Loss function applied to predictions and to hidden test ratings
        :param predictions: dict, df | dictionary, dataframe containing the predictions
                                   of your model
        :param loss_func: func | loss function to use
        :return: float | loss value on the test set
        """
        # transform prediction df into dictionary
        if type(data) == pd.core.frame.DataFrame:
                non_null_indices = list(predictions[predictions.notnull()].stack().index)
                pred = {(u, i): predictions.loc[u][i] for (u, i) in
                          non_null_indices}
                predictions = pred

        # Check that the predictions are here for all the (u, i) in test_set
        for key in self.valid_set:
            if key not in predictions:
                raise IOError("The predictions do not contain all "
                              "the (user, item) combinations of the "
                              "validation set")
        # Calculate loss 
        loss = loss_func(predictions, self.valid_set)

        return loss 

    def evaluate_train(self, predictions, loss_func):
        """
        Loss function applied to predictions and to hidden test ratings
        :param predictions: dict, df | dictionary, dataframe containing the predictions
                                   of your model
        :param loss_func: func | loss function to use
        :return: float | loss value on the test set
        """
        # transform prediction df into dictionary
        if type(data) == pd.core.frame.DataFrame:
                non_null_indices = list(predictions[predictions.notnull()].stack().index)
                pred = {(u, i): predictions.loc[u][i] for (u, i) in
                          non_null_indices}
                predictions = pred

        # Check that the predictions are here for all the (u, i) in test_set    
        for key in self.valid_set:
            if key not in predictions:
                raise IOError("The predictions do not contain all "
                              "the (user, item) combinations of the "
                              "validation set")
        # Calculate loss 
        loss = loss_func(predictions, self.train_set)

        return loss 
