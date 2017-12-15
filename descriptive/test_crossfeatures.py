# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Imports
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import pandas as pd
from utils.Pipeline import Pipeline
from utils.Vectorizer import Vectorizer
from models.cross_features.CrossFeaturesModel import CrossFeaturesModel
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load data
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# data_file = "./descriptive/db_nrows.csv"
data_file = "./descriptive/train.csv"
spotify_file = "./descriptive/SpotifyAudioFeatures_clean.csv"
sample_path = "./descriptive/train_sample.csv"
pipe = Pipeline(deezer_path=data_file,
                spotify_path=spotify_file,
                sample_path=sample_path,
                use_sample=True)
df = pipe.make()
print(df.shape)
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Select columns
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y = df["is_listened"]
column_selection = ['genre_id',
                    'user_gender',
                    "moment_of_week",
                    'user_age_bucket',
                    'track_age_bucket',
                    "track_valence_bucket",
                    "track_danceability_bucket",
                    "track_energy_bucket"]
df = df[column_selection]
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Create Cross Features
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
transforms = [("genre_id", "user_gender", "&"),
              ("user_age_bucket", "track_age_bucket", "&"),
              ("user_age_bucket", "track_valence_bucket", "&"),
              ("track_danceability_bucket", "moment_of_week", "&"),
              ("track_energy_bucket", "moment_of_week", "&"),
              ]
vect = Vectorizer(transforms)
vect.fit_transform(df, transforms)
print("\nThe new dataframe")
print(df.head())
# =========================================================================
# Place dummy columns
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
df_dummies = pd.get_dummies(data=df)
print("\nThe dataframe with dummies")
print(df_dummies.shape)
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Run Model
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model = CrossFeaturesModel(data=df_dummies,
                           target=y,
                           estimator="logistic")
model.train()
model.plot_important_features(top=10)
# =========================================================================
