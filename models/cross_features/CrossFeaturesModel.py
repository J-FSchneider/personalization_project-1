import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class CrossFeaturesModel:
    def __init__(self, data=None, target=None,
                 estimator='logistic', cv=3, verbose=False):
        if data is None or target is None:
            raise IOError("You should indicate both the train set "
                          "and the target column.")
        self.data = data
        self.target = target
        self.cv = cv
        self.estimator = estimator
        self.clf = None
        self.best_estimator = None
        self.verbose = verbose

    def set_clf(self):
        """
        Set the attribute according to the model used
        :return: None
        """
        if self.estimator == "logistic":
            self.clf = LogisticRegression()
        elif self.estimator == "random_forest":
            self.clf = RandomForestClassifier()
        else:
            raise IOError("This estimator is not supported by the model.")

    def _check_trained(self):
        """
        Sanity checker: verify that the model has been trained
        :return: None
        """
        if self.best_estimator is None:
            raise IOError("You need to train your model first. "
                          "Call method 'train'.")

    def grid_search_params(self):
        """
        Get the grid search parameters for the given model
        :return: dict | dictionary of parameters to try in grid search
        """
        if self.estimator == "logistic":
            params = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'penalty': ['l1', 'l2'],
                'class_weight': [None, 'balanced']
            }
            return params

        elif self.estimator == "random_forest":
            # TODO: complete
            return {}

    def train(self):
        """
        This function performs the training of the model.
        It actually uses other methods to do so.
        :return: None
        """
        print(">>> Model is training")
        t0 = time()
        self.set_clf()
        self.grid_search_cv()
        self.fit()
        print(">>> Model is trained and ready to use!")
        print("Run time: {} seconds".format(time() - t0))

    def fit(self):
        """
        Fits the best estimator obtained to the data
        :return: None
        """
        self.best_estimator.fit(self.data, self.target)

    def grid_search_cv(self):
        """
        Performs the grid search and saves the best model
        in the attribute self.best_estimator.
        :return: None
        """
        params = self.grid_search_params()
        grid = GridSearchCV(self.clf, params, cv=self.cv, verbose=self.verbose)
        grid.fit(self.data, self.target)
        self.best_estimator = grid.best_estimator_

    def cross_val_accuracy(self, verbose=False):
        """
        Computes the accuracy over the folds of the cross validation
        :param verbose: bool | verbose
        :return: list | list of accuracies
        """
        scores = cross_val_score(self.best_estimator,
                                 self.data, self.target, cv=self.cv)
        if verbose:
            print(">>> Accuracy over each fold of the CV :")
            print(list(scores))

        print(">>> Mean of the accuracy of the model over all folds:")
        print(np.mean(scores))
        return scores

    def predict(self, x):
        """
        Returns the predict class {0, 1} for the data x
        :param x: pd.DataFrame | data for which to predict
                                 the classes of its samples
        :return: list, np.ndarray | list of class predicitons {0, 1}
        """
        self._check_trained()
        return self.best_estimator.predict(x)

    def predict_proba(self, x):
        """
        Returns the probability to belong to class 1 for the data x
        :param x: pd.DataFrame | data for which to obtain the proba
        :return: list, np.ndarray | list of probabilities
        """
        self._check_trained()
        return self.best_estimator.predict_proba(x)

    def get_most_important_features(self, top=10):
        """
        Get the most important features from the model
        :param top: int | number of most important features to keep (keep top)
        :return: list[(str, float)] | list of (feature name, weight)
        """
        self._check_trained()
        if self.best_estimator is None:
            raise IOError("You must first do the grid search "
                          "to obtain the best model.")

        if self.estimator == "logistic":
            zipped = list(zip(self.data.columns, self.best_estimator.coef_[0]))
            return sorted(zipped, key=lambda x: abs(x[1]))[::-1][:top]

        elif self.estimator == "random_forest":
            indx = np.argsort(self.best_estimator.feature_importances_)[::-1]
            return self.data.columns[indx][:top]

    def plot_important_features(self, top=10):
        """
        Plots the top-most important features
        :param top: int | number of most important features to plot
        :return: plt.figure()
        """
        self._check_trained()
        if top > 20:
            raise IOError("Too much features to plot. "
                          "Please select parameter top < 20")

        features = self.get_most_important_features(top)
        name, weights = list(zip(*features))
        x_pos = np.arange(len(name))

        plt.bar(x_pos, weights, align='center')
        plt.xticks(x_pos, name, rotation='vertical')
        plt.ylabel('Top %i features weights' % top)
        plt.show()
