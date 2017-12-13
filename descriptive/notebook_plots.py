# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Imports
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from descriptive.tod_analysis import tod_pivot
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load data
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load data
# data = pd.read_csv("user_selection.csv", nrows=100)  # Constrained
data = pd.read_csv("user_selection.csv")  # Only selected users
# data = pd.read_csv("db.csv", nrows=100)  # Constrained
# data = pd.read_csv("db.csv")

# Note:: only using the songs that were actually listened to
# Only use the songs that where listened to
data = data[data["is_listened"] == 1]
# data = data[data["listen_type"] == 0]

# Select columns
user = "user_id"
song = "media_id"
time = "moment_of_day"
srp = "song_rel_per"
column_selection = [user, song, time]
data = data[column_selection]
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Bring audio features
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
audio_df = pd.read_csv("audio_features.csv")

# Select the audio features to include
audio_features = ['media_id',
                  'spotify_name',
                  'spotify_artist',
                  'duration_ms',
                  'energy',
                  'tempo',
                  'danceability',
                  'valence']
audio_df = audio_df[audio_features]
print("\nHere are the audio features")
print(audio_df.head(5))
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Run Pivot
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
user_final, final = tod_pivot(data=data,
                              audio_df=audio_df,
                              time_threshold=0.05)
print("\nBelow is the final table")
print(final.head())
print("\nThe final results at a user level")
print(user_final.head(20))
tmp = np.sum(user_final["final"] * user_final[srp])
print("\nThe results at an weighted aggregate "
      "level: {:2.2f}".format(tmp))
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Output Excels
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Note:: for a large data set the second excel will not output
pivot = pd.ExcelWriter("pivot_analysis.xlsx")
# user_final.to_excel(pivot, "User_Analysis")
final.to_excel(pivot, "User_Song Analysis")
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot time threshold impact
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# thresholds = np.linspace(0, 0.21, num=10)
# n = len(thresholds)
# x = np.zeros((n, 1))
# i = 0
# for thres in thresholds:
#     tmp, _ = tod_pivot(data=data,
#                        audio_df=audio_df,
#                        time_threshold=thres)
#     x[i] = np.sum(tmp["final"] * tmp[srp])
#     i += 1
#
# plt.plot(thresholds, x)
# plt.show()
# =========================================================================
