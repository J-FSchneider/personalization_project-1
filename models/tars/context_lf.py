import numpy as np
import pandas as pd


def squared_error_gradient(M, U, V):
    """
    calculates the gradient and error of the simple MSE Loss function (unregularized)
    :param M: np.array | Numpy array (user_id,media_id,rating) that contains the rating matrix in long-format
    :param U: np.array | Numpy array that contains the user latent factors
    :param V: np.array | Numpy array that contains the item latent factors
    :return:
    """
    UV = pd.DataFrame(np.dot(U, V))
    E = np.subtract(M, UV)
    J = 0.5 * np.nansum(np.square(E))

    return E, J



class ContextLF:

    def __init__(self, data, latent_factors=5, context_id = 'moment_of_day', ratings='is_listened'):
        if data is None:
            raise IOError("Please specify the data set.")
        self.data = data
        self.n = data['user_id'].nunique()
        self.m = data['media_id'].nunique()
        self.context_id = context_id
        self.context = np.append(data[context_id].unique(),'no_context')
        self.d = len(self.context)
        self.U = []
        self.V = []
        self.lf = latent_factors
        self.ratings = ratings




    def fit(self, converg=0.1, learning_rate=0.00001, regularization=1, verbose=False):
        """
        :param converg: float | convergence criteria
        :param learning_rate: float |
        :param regularization: float |
        :param verbose: bol |
        :return: latent factor matrices for each context
        """
        self.data = pd.DataFrame(
            self.data[self.data['listen_type'] == 1].groupby(['user_id', 'media_id', self.context_id])[self.ratings].mean())
        self.data = self.data.reset_index()

        for i in range(self.d):

            if verbose:
                print('Model for context: '+ str(self.context[i]))
            if i == self.d -1:
                data_sample = self.data
            else:
                data_sample = self.data[self.data[self.context_id] == self.context[i]]

            R = pd.pivot_table(data_sample, columns='media_id', index='user_id', values='is_listened', aggfunc='mean')
            U = pd.DataFrame(np.random.rand(R.shape[0], 5), index=R.index)
            V = pd.DataFrame(np.random.rand(5, R.shape[1]), columns=R.columns)

            convergence = False
            t = 0
            J1 = 0
            while not convergence:
                # compute Gradient and Loss
                E, J = squared_error_gradient(R, U, V)
                # fill Errors with 0 to ignore the values for prediction in the update
                E_upt = E.fillna(0)
                # update the matrices U and V by gradient descent
                U_update = np.dot(E_upt, np.transpose(V))
                V_update = np.transpose(np.dot(np.transpose(E_upt), U))
                U_new = U * (1 - learning_rate * regularization) + learning_rate * U_update
                V_new = V * (1 - learning_rate * regularization) + learning_rate * V_update
                U = U_new
                V = V_new
                if verbose and t%100 == 0:
                    print('Loss on iteration ' + str(t) + ': ' + str(J))

                # Convergence criteria
                if ((J1 - J) < converg) and (t > 0):
                    convergence = True

                else:
                    J1 = J
                t += 1

            self.U.append(U)
            self.V.append(V)



    def predict_topk(self, user_id, context='no_context', k=20):
        """

        :param user_id: int | specify the user id, for which you would like to make the prediction
        :param context: str | specify the name of the context for which you would like to make the prediction,
                                if None, no context is chosen
        :param k: int   | number of top k items to be returned
        :return: np.array | list of top k items, descending
        """

        # select index of context
        try:
            t = self.context.tolist().index(context)
        except:
            print('Context is not a valid context! Try: ' + str(self.context.tolist()))
            return None


        # select U and V
        U = self.U[t]
        if user_id in U.index:
            V = self.V[t]

        else:
            U = self.U[-1]
            V = self.V[-1]

        # make top-k predictions

        pred = np.transpose(pd.DataFrame(U.loc[user_id,:]))

        # predict row

        items = pd.DataFrame(np.dot(pred, V), columns=V.columns)
        has_data = self.data[(self.data['user_id'] == user_id) & (self.data[self.context_id] == context)]['media_id'].unique()
        top_items = np.transpose(items.drop(has_data, axis=1))
        top_k = np.array(top_items.nlargest(k, 0).index)

        return top_k

    def song_similarity(self, songs, data):

        """
        The function takes in a list of recommended songs and outputs a list
        of serendipitous songs by genre. If the list of serendipitous songs is
        too short(<= 5), we output the original recommendations.


        :param songs: list of recommended songs
        :return: list of serendipitous songs
        """
        data = data[data.media_id.isin(songs)]
        data = data.drop_duplicates(['media_id'], keep='first')
        data = data[['media_id', 'genre_id']]
        data = data.drop_duplicates(['genre_id'], keep='first')
        if len(data['media_id']) >= 5:
            return list(data['media_id'])
        else:
            return songs
