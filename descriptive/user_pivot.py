# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Imports
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import pandas as pd
import numpy as np
from descriptive.tod_analysis import tod_pivot
from utils.df_trans import df_summ
from utils.df_trans import df_tot
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Variable naming
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Name of the identifiers
user = "user_id"
song = "media_id"
time = "moment_of_day"

# Set new variables
st = "st_count"
sr = "song_rel"
srp = "song_rel_per"
t = "time_sum"
tp = "time_per"
tr = "time_rel"
s = "song_sum"
sp = "song_per"
tot = "total"
w = "weights"
indi = "indicator"
m = "mult"
wi = "weight_ind"
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load data
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load data
# data = pd.read_csv("user_selection.csv", nrows=100)  # Constrained
# data = pd.read_csv("user_selection.csv")  # Only selected users
# data = pd.read_csv("db.csv", nrows=100)  # Constrained
data = pd.read_csv("db.csv")

# Note:: only using the songs that were actually listened to
# Only use the songs that where listened to
data = data[data["is_listened"] == 1]
data = data[data["listen_type"] == 0]

# Select columns
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
# Construct the individual tables
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Create granular user/song/time table
master = df_summ(df=data,
                 index=[user, song, time],
                 rename=st,
                 target=song,
                 criteria="count")
print("\nBelow is the user/song/time table")
print(master.head(5))
print("\nAlong with its first shape")
print(master.shape)

# Do the time table
time_agg = df_summ(df=master,
                   index=[user, time],
                   rename=t,
                   target=st,
                   criteria="sum")
time_agg = df_tot(df=time_agg,
                  index=user,
                  rename=tot,
                  target=t,
                  criteria="sum")
time_agg[tp] = time_agg[t] / time_agg[tot]

# Note:: here is the relevant time threshold
time_threshold = 0.1
time_agg[tr] = time_agg[tp].apply(lambda x: 1 if x > time_threshold else 0)
time_rel = time_agg[[user, time, tr]]
time_vars = time_agg[time].unique()  # List to undo pivot
print("\nBelow is the user / time aggregation")
print(time_agg.head(5))

# Do the song table
song_agg = df_summ(df=master,
                   index=[user, song],
                   rename=s,
                   target=st,
                   criteria="sum")
song_agg = df_tot(df=song_agg,
                  index=user,
                  rename=tot,
                  target=s,
                  criteria="sum")
song_agg[w] = song_agg[s] / song_agg[tot]
print("\nBelow is the user/song aggregation")
print(song_agg.head(5))
user_song_rel = df_summ(df=song_agg,
                        index=user,
                        rename=sr,
                        target=s,
                        criteria="sum")
print("\nBelow is the user/song relevance computations")
user_song_rel[tot] = np.sum(user_song_rel[sr])
user_song_rel[srp] = user_song_rel[sr] / user_song_rel[tot]
print(user_song_rel.head())
user_song_rel = user_song_rel[[user, srp]]
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Generate user pivot table
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
user_pivot = pd.pivot_table(master,
                            values=st,
                            index=[user, song],
                            columns=time)
print("\nBelow is the user pivot table")
print(user_pivot.head())
user_pivot = user_pivot.fillna(0)
user_pivot = user_pivot.reset_index()
user_melt = pd.melt(user_pivot,
                    id_vars=[user, song],
                    value_vars=time_vars)
print("\nHere is the undo of the pivot")
print(user_melt.head(20))
print("\nAlong with the new shape")
print(user_melt.shape)
user_melt = pd.merge(user_melt, time_rel,
                     on=[user, time],
                     how="left")
user_melt[indi] = user_melt["value"][user_melt[tr] == 1]
user_melt = user_melt.fillna(1)
user_melt[indi] = user_melt[indi].apply(lambda x: 1 if x > 0 else 0)
print("\nThe new user melt")
print(user_melt.head(20))
user_melt = user_melt[[user, song, time, indi]]
user_pivot = pd.pivot_table(user_melt,
                            values=indi,
                            index=[user, song],
                            columns=time)
print("\nBelow is the second user pivot table")
print(user_pivot.head(20))

user_pivot[m] = 1
for column in user_pivot.columns:
    user_pivot[m] = user_pivot[m] * user_pivot[column]

user_pivot = user_pivot.reset_index()
print("\nThird version of the pivot table")
print(user_pivot.head())
user_mult = user_pivot[[user, song, m]]
print("\nBelow is the user mult table")
print(user_mult.head())
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Join the tables
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
final = pd.merge(song_agg, user_mult,
                 on=[user, song],
                 how="left")
final[wi] = final[m] * final[w]
final = pd.merge(final, audio_df,
                 on=song,
                 how="left")
print("\nBelow is the final table")
print(final.head())
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Aggregate results
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
user_final = df_summ(df=final,
                     index=user,
                     rename="final",
                     target=wi,
                     criteria="sum")
user_final = user_final.sort_values(by=user)
user_final = pd.merge(user_final, user_song_rel,
                      on=user,
                      how="left")
print("\nThe final results at a user level")
print(user_final.head(20))
tmp = np.sum(user_final["final"] * user_final[srp])
aux = np.sum(user_final["final"] / len(user_final["final"]))
print("\nThe results at an weighted aggregate level: {:2.2f}".format(tmp))
print("\nThe results at a aggregate level: {:2.2f}".format(aux))
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Check that function is working properly
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
uf, f = tod_pivot(data=data,
                  audio_df=audio_df,
                  time_threshold=time_threshold)
print("\nBelow is the final table")
print(f.head())
print("\n[Check] The final results at a user level")
print(uf.head(20))
tmp = np.sum(uf["final"] * uf[srp])
print("\n[Check] The results at an weighted aggregate "
      "level: {:2.2f}".format(tmp))
# =========================================================================
