# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Imports
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import pandas as pd
from utils.Pipeline import Pipeline
from utils.df_trans import df_summ
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load data & add features
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Run pipeline
spotify_file = "./descriptive/SpotifyAudioFeatures_clean.csv"
data_file = "./descriptive/train.csv"
data = pd.read_csv(data_file, nrows=100)  # to check headers
pipe = Pipeline(deezer_path=data_file, spotify_path=spotify_file)
df = pipe.make()

# Create output files
df.to_csv("./descriptive/db.csv")

# Select columns
column_selection = ['genre_id',
                    'user_id',
                    'spotify_name',
                    'spotify_artist',
                    'spotify_album_name',
                    'hour_listen',
                    'day_listen',
                    'moment_of_day',
                    'energy',
                    'tempo',
                    'danceability',
                    'time_signature',
                    'duration_ms',
                    'ts_listen',
                    'media_id',
                    'album_id',
                    'context_type',
                    'release_date',
                    'platform_name',
                    'platform_family',
                    'media_duration',
                    'listen_type',
                    'user_gender',
                    'artist_id',
                    'user_age',
                    'is_listened',
                    'acousticness',
                    'deezer_artist',
                    'deezer_bpm',
                    'deezer_name',
                    'id',
                    'instrumentalness',
                    'key',
                    'liveness',
                    'loudness',
                    'mode',
                    'speechiness',
                    'type',
                    'valence',
                    'month_listen',
                    'year_listen',
                    'converted_ts',
                    'year_release',
                    'month_release',
                    'day_release',
                    'track_tempo_bucket',
                    'track_age_bucket',
                    'track_duration_bucket',
                    'user_age_bucket']
df = df[column_selection]
media_selection = [7, 8, 43, 51, 5943, 5188]
user_selection = df[df["user_id"].isin(media_selection)]
user_selection.to_csv("./descriptive/user_selection.csv")
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Generate audio features
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
audio_features = ['media_id',
                  'spotify_name',
                  'spotify_artist',
                  'spotify_album_name',
                  'genre_id',
                  'duration_ms',
                  'energy',
                  'tempo',
                  'danceability',
                  'valence',
                  'time_signature',
                  'acousticness',
                  'mode',
                  'key',
                  'speechiness',
                  'loudness',
                  'liveness',
                  'instrumentalness']
audio_df = df_summ(df=df,
                   index=audio_features,
                   rename="count",
                   target="media_id",
                   criteria="count")
audio_df = audio_df.sort_values(by="count", ascending=False)
audio_df.to_csv("./descriptive/audio_features.csv")
# =========================================================================
