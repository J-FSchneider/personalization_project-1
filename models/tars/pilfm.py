import numpy as np

class PILFM:

    def __init__(self, data):
        """

        :param data: pandas df | long-format ratings data with labels (user_id, media_id,rating,context_id),
                            rating should be between 0 and 1, context should be categories
        """
        self.data = data
        self.n = data['user_id'].nunique()
        self.m = data['media_id'].nunique()
        self.d = data['context_id'].nunique()
        self.U = np.random.rand(self.n,self.m)
        self.V = np.random.rand(self.n,self.d)
        self.W = np.random.rand(self.m,self.d)

    def fit(self):
        pass




    def predict(self):
        pass