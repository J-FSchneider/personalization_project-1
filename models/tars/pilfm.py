import numpy as np

class PILFM:

    def __init__(self, data, latent_factors=5):
        """

        :param data: pandas df | long-format ratings data with labels (user_id, media_id,rating,context_id),
                            rating should be between 0 and 1, context should be categories
        """
        if data == None:
            raise IOError("Please specify the data set.")
        self.data = data
        self.n = data['user_id'].nunique()
        self.m = data['media_id'].nunique()
        self.d = data['context_id'].nunique()
        self.U = np.random.rand(self.n,latent_factors)
        self.V = np.random.rand(self.m,latent_factors)
        self.W = np.random.rand(self.d,latent_factors)
        self.lf = latent_factors

    def fit(self):

        R = np.array()


        convergence = False
        J =

        while not(convergence):
            # calculate prediction error




            # update model




    def predict(self):
        pass