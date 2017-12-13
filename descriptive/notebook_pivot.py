# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Imports
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import pandas as pd
from descriptive.tod_analysis import time_table
from descriptive.tod_analysis import u_pivot
from descriptive.tod_analysis import tod_pivot
# import matplotlib.pyplot as plt
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load data
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load data
data = pd.read_csv("user_selection.csv", nrows=100)  # Constrained
# data = pd.read_csv("user_selection.csv")  # Only selected users
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
tr = "time_rel"
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
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Do time agg analysis
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
time_agg = time_table(data=data,
                      user_id=user,
                      song_id=song,
                      time_id=time,
                      time_threshold=0.1)
print("\nBelow is the user / time analysis table")
print(time_agg.head(10))

# User with non-significant times of day
non_sig = time_agg[time_agg[tr] == 0].sort_values(by=user)
print("\nBelow are the users that had time taken out")
print(non_sig.head(10))
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot selected user time distributions
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# user_selection = [1, 2, 13]
# barplot_data = time_agg[time_agg[user].isin(user_selection)]
# barplot_data = barplot_data[[user, time, "time_per"]]
# print(barplot_data.head())
# i = 1
# for u in user_selection:
#     plt.subplot(1, 3, i)
#     tmp = barplot_data[barplot_data[user] == u]
#     plt.bar(tmp[time], tmp["time_per"], tick_label=tmp[time])
#     objects = ("morning", "afternoon_evening", "late_night")
#     plt.xticks(tmp[time], objects)
#     plt.ylabel("% of listening time")
#     plt.title("User "+str(u)+" time distribution")
#     i += 1
#
# plt.show()
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Do step by step user pivots
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
time_rel = time_agg[[user, time, tr]]
time_vars = time_agg[time].unique()  # List to undo pivot
u_piv, u_before, w_nans = u_pivot(data=data,
                                  user_id=user,
                                  song_id=song,
                                  time_id=time,
                                  time_rel=time_rel,
                                  time_vars=time_vars)
print("\nBelow are the step by step pivots")
print(w_nans.head(5))
print(u_piv.head())
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Show final user summary
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
s = "song_sum"
tot = "total"
m = "mult"
w = "weights"
u_fin, fin = tod_pivot(data=data,
                       audio_df=audio_df,
                       time_threshold=0.05)
print("\nBelow is the certain table for a user")
column_selection = [user,
                    song,
                    "spotify_name",
                    "spotify_artist",
                    m,
                    s,
                    tot,
                    w]
user_2 = fin[column_selection]
user_2 = user_2[user_2[user] == 2]
user_2 = user_2.sort_values(by=w, ascending=False)
print(user_2.head())
print("\nBelow is the user final table")
print(u_fin.head())
aux = np.sum(u_fin["final"] * u_fin["song_rel_per"])
print("The result is {:2.2f}".format(aux))
# =========================================================================
