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
m = "mult"
mp = "always_per"
mpu = "always_per_user"
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load data
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load data
# data = pd.read_csv("user_selection.csv", nrows=100)  # Contrained
# data = pd.read_csv("user_selection.csv")  # Only selected users
# data = pd.read_csv("db.csv", nrows=100)  # Constrained
data = pd.read_csv("db.csv")

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
print(audio_df.head(10))
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
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Filter out noise
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# result = result[result[tp] > 0.1]
result = result[result[s] > 1]
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Create user pivot
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# df_piv = result[result[sp] > 0.1]
df_piv = result[[user, song, time, st]]
print(df_piv.head())
user_pivot = pd.pivot_table(df_piv,
                            values=st,
                            index=[user, song],
                            columns=time)
print("\nBelow if the user pivot table")
print(user_pivot.head())
user_pivot = user_pivot.fillna(0)
user_pivot = user_pivot.applymap(lambda x: 1 if x > 1 else 0)
user_pivot[m] = 1
for column in user_pivot.columns:
    user_pivot[m] = user_pivot[m] * user_pivot[column]

user_pivot = user_pivot.reset_index()
print("\nSecond version of the pivot table")
print(user_pivot.head())
user_pivot = user_pivot.merge(audio_df, on=song, how="left")
user_mult = user_pivot[[user, song, m]]
print("\nBelow is the user mult table")
print(user_mult.head())

# Add to song_agg
song_agg = pd.merge(song_agg, user_mult,
                    on=[user, song],
                    how="left")
song_agg = song_agg.merge(audio_df, on=song, how="left")
song_agg = song_agg.merge(weights, on=[user, song], how="left")
song_agg = song_agg.fillna(0)
song_agg[mp] = song_agg[m] * song_agg[w]
user_always = df_summ(df=song_agg,
                      index=user,
                      rename=mpu,
                      target=mp,
                      criteria="sum")
aux = df_summ(df=song_agg,
              index=user,
              rename="song_total",
              target=s,
              criteria="sum")
user_always = user_always.merge(aux, on=user, how="left")
user_always["total"] = np.sum(user_always["song_total"])
user_always["per"] = user_always["song_total"] / user_always["total"]
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute weights
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Add the user/song mult
result = result.merge(user_mult, on=[user, song], how="left")
# Add the user/song weights
result = result.merge(weights, on=[user, song], how="left")
result[wdiff] = result[w] * result[diff]
result[mp] = result[w] * result[m]
print("\nWith weights")
print(result.head(3))
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Construct final tables
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Do the differences aggregation
user_song = df_summ(df=result,
                    index=[user, song],
                    rename=usi,
                    target=wdiff,
                    criteria="sum")
user_song = user_song.merge(weights, on=[user, song], how="left")
user_song = user_song.sort_values(by=[user, usi], ascending=[1, 0])
user_song = user_song.merge(audio_df, on=song, how="left")
user_song = user_song.merge(user_mult, on=[user, song], how="left")
user_song[mp] = user_song[m] * user_song[w]

# Also merge into master
result = result.merge(user_mult, on=[user, song], how="left")
final_master = result.merge(audio_df, on=song, how="left")

# Print final results
print("\nFinal user/song differences")
print(user_song.head(4))

result = result[result[s] > 1]
user_ind = df_summ(df=result,
                   index=user,
                   rename=ui,
                   target=wdiff,
                   criteria="sum")
aux = df_summ(df=result,
              index=user,
              rename=mpu,
              target=mp,
              criteria="mean")
user_ind = user_ind.merge(aux, on=user, how="left")
user_ind = user_ind.sort_values(by=ui, ascending=False)
print("\nFinal user differences")
print(user_ind.head(30))
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Output Excel file for QC
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Note:: no excel files are output right now
# # Output the master table
# master_writer = pd.ExcelWriter("master.xlsx")
# final_master.to_excel(master_writer, "Sheet1")
#
# # Output individual tables
# time_writer = pd.ExcelWriter("time.xlsx")
# time_agg.to_excel(time_writer, "Sheet1")
song_writer = pd.ExcelWriter("song.xlsx")
# song_agg.to_excel(song_writer, "Sheet1")
user_always.to_excel(song_writer, "Sheet2")
#
# # Output the audio features
# audio_writer = pd.ExcelWriter("audio.xlsx")
# audio_df.to_excel(audio_writer, "Sheet1")
#
# # Output the user/song difference table
# usi_writer = pd.ExcelWriter("usi.xlsx")
# user_song.to_excel(usi_writer, "Sheeet1")
#
# # Output the user pivot
# piv_writer = pd.ExcelWriter("pivot.xlsx")
# user_pivot.to_excel(piv_writer, "Sheet1")
# =========================================================================
