# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Import Packages
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from models.model_based.data_prep import hit_rate
from models.model_based.data_prep import hr_evolution
# from models.model_based.data_prep import relevant_elements
from models.model_based.data_prep import plot_relevant_elements
# from models.model_based.data_prep import song_rank
# from models.model_based.SVDpp import SurSVDpp
# from models.model_based.SVDpp import SurSVD

# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load Data
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
filename = "/home/ap/personalization_project/models/model_based/train.csv"
# data = pd.read_csv(filename, nrows=10000)
data = pd.read_csv(filename)
np.set_printoptions(precision=2, suppress=True)
# data = pd.read_csv(filename)

# =========================================================================

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Add Hit Rate and Song Ranking
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Add the hit rate to the data
data = hit_rate(data)

# Build the user ranges for the plots
first = [x for x in range(10, 100, 10)]
second = [y for y in range(100, 1000, 100)]
third = [z for z in range(1000, 8000, 1000)]
# user_range = first
# user_range = first + second
user_range = first + second + third

# Construct the report for the Hit Rate evolution
data_filtered = data[data["listen_type"] == 1]
report = hr_evolution(data_filtered, user_range)
print(report.head(10))

# Construct the report for the number of songs that account for
# the threshold of the total
threshold = 0.5
report2 = plot_relevant_elements(data, user_range, "media_id",
                                 threshold=threshold)
print("\n")
print(report2)
report3 = plot_relevant_elements(data, user_range, "artist_id",
                                 threshold=threshold)
print("\n")
print(report3)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot Results
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# plt.figure()
# plt.plot(x[:, 0], x[:, 1])
#
# plt.figure()
# plt.plot(x[:, 0], x[:, 3])
# plt.show()
#
# =========================================================================
