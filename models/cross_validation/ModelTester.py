import numpy as np
import pandas as pd


class ModelTester:
    def __init__(self, ratios=(0.7, 0.2, 0.1), model_based=True, seed=42):
        """
        Constructor of the class
        :param ratios: 3-tuple | ratios of train, validation and test sets
        :param model_based: boolean | boolean to state if model based framework.
                                      If False, it assume neighborhood base
                                      model.
        """
        # TODO: add some sanity checks on sum of ratios
        self.MODEL_BASED = model_based
        self.ratios = ratios
        self.valid_set = {}
        self.test_set = {}
        self.train_set = {}
        self.non_null_indices = []
        self.data = None
        self.seed = seed
        # Set the seed for random generators
        np.random.seed(self.seed)

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
        shuffled = np.random.permutation(self.non_null_indices)

        # Get the indices for validation and test sets
        valid_indices = shuffled[int(len(self.non_null_indices) * train_ratio):
                                 int(len(self.non_null_indices) * (train_ratio + valid_ratio))]
        test_indices = shuffled[int(
            len(self.non_null_indices) * (train_ratio + valid_ratio)) + 1:]

        # Fills the attribute dictionaries
        self.valid_set = {(u, i): self.data.loc[u][i]
                          for (u, i) in valid_indices}
        self.test_set = {(u, i): self.data.loc[u][i]
                         for (u, i) in test_indices}

        if self.MODEL_BASED:
            # Unnecessary work for neighborhood based models as it does modify
            # train set, only needs to evaluate test sets.
            train_indices = shuffled[
                            :int(len(self.non_null_indices) * train_ratio)]
            self.train_set = {(u, i): self.data.loc[u][i]
                              for (u, i) in train_indices}

        # Replace original values in DataFrame data by np.nan values
        for u, i in self.valid_set:
            self.data.loc[u][i] = np.nan
        for u, i in self.test_set:
            self.data.loc[u][i] = np.nan

    def fit_transform(self, data, verbose=True):
        """
        Runs fit and then transform functions
        :param data:
        :return:
        """
        if verbose:
            print(">>> The cross-validation framework is being built ...\n"
              "    Please wait, you'll be able to use all of its features soon!")

        self.fit(data)
        self.transform()

        if verbose:
            print(">>> DONE")

    def evaluate_test(self, predictions, loss_func, verbose=True):
        """
        Loss function applied to predictions and to hidden test ratings
        :param predictions: dict or pd.DataFrame | dictionary or pd.DataFrame
                                                   containing the predictions
                                                   of your model
        :param loss_func: func | loss function to use
        :return: float | loss value on the test set
        """
        # Transform prediction df into dictionary
        if isinstance(predictions, pd.DataFrame):
            pred = {(u, i): predictions.loc[u][i]
                    for (u, i) in self.test_set}
            predictions = pred

        # Check that the predictions are here for all the (u, i) in test_set
        for key in self.test_set:
            if key not in predictions:
                raise IOError("The predictions do not contain all "
                              "the (user, item) combinations of the test set")

        # Calculate Loss
        loss = loss_func(predictions, self.test_set, verbose=verbose)

        return loss 

    def evaluate_valid(self, predictions, loss_func, verbose=True):
        """
        Loss function applied to predictions and hidden validation set ratings
        :param predictions: dict or pd.DataFrame | dictionary or pd.DataFrame
                                                   containing the predictions
                                                   of your model
        :param loss_func: func | loss function to use
        :return: float | loss value on the validation set
        """
        # Transform prediction df into dictionary
        if isinstance(predictions, pd.DataFrame):
            pred = {(u, i): predictions.loc[u][i]
                    for (u, i) in self.valid_set}
            predictions = pred

        # Check that the predictions are here for all the (u, i) in test_set
        for key in self.valid_set:
            if key not in predictions:
                raise IOError("The predictions do not contain all "
                              "the (user, item) combinations of the "
                              "validation set")
        # Calculate loss 
        loss = loss_func(predictions, self.valid_set, verbose=verbose)

        return loss 

    def evaluate_train(self, predictions, loss_func, verbose=True):
        """
        Loss function applied to predictions and train ratings.
        This function will be used by matrix factorization models only.
        :param predictions: dict or pd.DataFrame | dictionary or pd.DataFrame
                                                   containing the predictions
                                                   of your model
        :param loss_func: func | loss function to use
        :return: float | loss value on the train set
        """
        if not self.MODEL_BASED:
            # TODO: improve error message
            raise ValueError("You cannot use this function as you need this "
                             "method. Only for model based.")
        # Transform prediction df into dictionary
        if isinstance(predictions, pd.DataFrame):
            non_null_indices_pred = list(predictions[predictions.notnull()]
                                         .stack().index)
            pred = {(u, i): predictions.loc[u][i]
                    for (u, i) in non_null_indices_pred}
            predictions = pred

        # Check that the predictions are here for all the (u, i) in test_set    
        for key in self.train_set:
            if key not in predictions:
                raise IOError("The predictions do not contain all "
                              "the (user, item) combinations of the "
                              "train set")
        # Calculate loss 
        loss = loss_func(predictions, self.train_set, verbose=verbose)

        return loss

    def shuffle_cv(self):
        """
        Shuffles the indices between valid and train set without
        changing the test set
        :return: void
        """
        merged = self.train_set.copy()
        merged.update(self.valid_set)
        tmp = list(merged.keys())
        random_keys = np.random.randn(len(merged))
        merged_shuffled_keys = [tmp[i] for i in np.argsort(random_keys)]
        size = len(self.train_set)
        new_train_indices = merged_shuffled_keys[:size]
        new_valid_indices = merged_shuffled_keys[size:]
        self.train_set = {t: merged[t] for t in new_train_indices}
        self.valid_set = {t: merged[t] for t in new_valid_indices}
