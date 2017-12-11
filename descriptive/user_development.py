# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Imports
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import pandas as pd
import numpy as np
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
t = "time_sum"
tp = "time_per"
s = "song_sum"
sp = "song_per"
tot = "total"
w = "weights"
wtot = "wtotal"
diff = "diff"
wdiff = "wdiff"
usi = "user_song_ind"
ui = "user_ind"
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load data
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load data
# data = pd.read_csv("user_selection.csv", nrows=100)  # Contrained
data = pd.read_csv("user_selection.csv")  # Only selected users
# data = pd.read_csv("db.csv", nrows=100)  # Constrained
# data = pd.read_csv("db.csv")  # Constrained

# Select columns
column_selection = [user, song, time]
data = data[column_selection]
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

# Do the time / song aggregations
time_agg = df_summ(df=master,
                   index=[user, time],
                   rename=t,
                   target=st,
                   criteria="sum")
print("\nBelow is the user/time aggregation")
print(time_agg.head(5))
song_agg = df_summ(df=master,
                   index=[user, song],
                   rename=s,
                   target=st,
                   criteria="sum")
print("\nBelow is the user/song aggregation")
print(song_agg.head(5))

# Get the user/media relevance weights
weights = df_tot(df=song_agg,
                 index=user,
                 rename=wtot,
                 target=s,
                 criteria="sum")
weights[w] = weights[s] / weights[wtot]
weights = weights[[user, song, w]]
print("\nBelow are the weights per songs")
print(weights.head(5))
check = df_summ(df=weights,
                index=user,
                rename="check",
                target=w,
                criteria="sum")
print("\nBelow is the consistency check of the weights")
print(check.head(3))
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Join the individual tables
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
master = df_tot(df=master,
                index=user,
                rename=tot,
                target=st,
                criteria="sum")
aux = pd.merge(master, time_agg, on=[user, time], how="left")
print("\nBelow is the master with the time part")
print(aux.head(3))
result = pd.merge(aux, song_agg, on=[user, song], how="left")
print("\nBelow is the master with all parts")
print(result.head(3))
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Add % to columns
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Create the % columns
result[sp] = result[st] / result[s]
result[tp] = result[t] / result[tot]
result[diff] = np.abs(result[sp] / result[tp] - 1)
print("\nBelow is the result table with the diff colum")
print(result.head(5))

# Add the user/song weights
result = result.merge(weights, on=[user, song], how="left")
result[wdiff] = result[w] * result[diff]
print("\nWith weights")
print(result.head(3))
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Yield user/song and user "weighted difference" results
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Bring audio features
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
print(audio_df.head(10))

# Do the differences aggregation
user_song = df_summ(df=result,
                    index=[user, song],
                    rename=usi,
                    target=wdiff,
                    criteria="sum")
user_song = user_song.merge(weights, on=[user, song], how="left")
user_song = user_song.sort_values(by=[user, usi], ascending=[1, 0])
user_song = user_song.merge(audio_df, on=song, how="left")

# Also merge into master
final_master = result.merge(audio_df, on=song, how="left")

# Print final results
print("\nFinal user/song differences")
print(user_song.head(4))
user_ind = df_summ(df=result,
                   index=user,
                   rename=ui,
                   target=wdiff,
                   criteria="sum")
user_ind = user_ind.sort_values(by=ui, ascending=False)
print("\nFinal user differences")
print(user_ind.head(30))
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Output Excel file for QC
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Output the master table
master_writer = pd.ExcelWriter("master.xlsx")
final_master.to_excel(master_writer, "Sheet1")

# Output individual tables
time_writer = pd.ExcelWriter("time.xlsx")
time_agg.to_excel(time_writer, "Sheet1")
song_writer = pd.ExcelWriter("song.xlsx")
song_agg.to_excel(song_writer, "Sheet1")

# Output the audio features
audio_writer = pd.ExcelWriter("audio.xlsx")
audio_df.to_excel(audio_writer, "Sheet1")

# Output the user/song difference table
usi_writer = pd.ExcelWriter("usi.xlsx")
user_song.to_excel(usi_writer, "Sheeet1")

# Output the user difference table
# =========================================================================
