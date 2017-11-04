# Import Packages
# import numpy as np
import pandas as pd
from utils.loss_functions import mean_squared_error
from SVDpp import SurSVDpp
from models.cross_validation.latent_factor_test import latent_factor_test
from models.cross_validation.ModelTester import ModelTester

ratings_dict = {"itemID": [1, 1, 1, 2, 2],
                "userID": [9, 32, 2, 45, 20],
                "ratings": [0, 0, 1, 0, 1]}

df = pd.DataFrame(ratings_dict)
df = df[["userID", "itemID", "ratings"]]

print(df.head())

ratings_matrix = df.pivot(index="userID",
                          columns="itemID",
                          values="ratings")

print(ratings_matrix.head())

a = SurSVDpp()
a.fit(ratings_matrix)

print("\nOutput")
print(a.predict(user=2, item=2))

k_values = [1, 2, 3]
cv_times = 3


d_test, d_train = latent_factor_test(k_val=k_values,
                                     cv_times=cv_times,
                                     model=SurSVDpp,
                                     loss_function=mean_squared_error,
                                     model_tester=ModelTester,
                                     data=ratings_matrix)
