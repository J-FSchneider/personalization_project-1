# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Imports
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from utils.Pipeline import Pipeline
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load data
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
spotify_file = "SpotifyAudioFeatures_clean.csv"
# data_file = "db_nrows.csv"
data_file = "train.csv"
pipe = Pipeline(deezer_path=data_file, spotify_path=spotify_file)
df = pipe.make_selected()
print(df.head())
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Get first user/song table
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
songs = [135010092, 130700214, 70079770]  # for user 3466
# user_song = pipe.user_song(user_id=821, songs=[876500])
user_song = pipe.user_song(user_id=3466, songs=songs)
print("\nBelow is the user / song hour analysis")
print(user_song.head())
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Get user / day table
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
days = [21, 23, 25, 26]  # for user 7
# user_day = pipe.user_day(user_id=612, days=[22])
user_day = pipe.user_day(user_id=7, song_id=18190270, days=days)
print("\nBelow is the user / day analysis")
print(user_day.head())
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Get user pivot table example
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
w_nans = pipe.get_user_pivot()
print("\nBelow is the pivot with NaNs")
print(w_nans.head())
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Get final analysis
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# u = pipe.run_user_analysis(user_id=573)
u = pipe.run_user_analysis(user_id=1)  # for user 1
print("\nBelow is the summary of the analysis for a certain user")
print(u.head())
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Run all the analysis
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
u_fin = pipe.run_analysis()
print("\nBelow is the summarizing table")
print(u_fin.head())
# =========================================================================
