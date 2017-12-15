# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Imports
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import matplotlib.pyplot as plt
from utils.Pipeline import Pipeline

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
                use_sample=False)
df = pipe.make()
print(df.shape)
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Make moment of day / week frequency plot
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plt.figure()
ax = plt.subplot(111)
plt.title("Moment of Day Split")
plt.hist(df["moment_of_day"])
order = ['morning',
         'afternoon_evening',
         'late_night']
ax.set_xticklabels(order)

plt.figure()
ax = plt.subplot(111)
plt.title("Hour of Day Histogram")
plt.hist(df["hour_listen"], bins=24, rwidth=0.75)

plt.figure()
ax = plt.subplot(111)
plt.title("Weekday vs Weekend Split")
plt.hist(df["moment_of_week"])
order = ["weekday_morning",
         "weekday_afternoon_to_evening",
         "weekday_late_night",
         "weekend_morning",
         "weekend_afternoon_evening",
         "weekend_late_night"]
ax.set_xticklabels(order, rotation=45)
plt.show()
# =========================================================================
