import pandas as pd
import numpy as np
from collections import Counter
from descriptive.tod_analysis import df_summ, time_table, u_pivot, tod_pivot
from time import time
from utils.preprocessing import parse_release_date, parse_ts_listen, \
                                parse_user_age, parse_moment_of_day, \
                                parse_media_duration_bucket, \
                                parse_track_age_bucket, parse_track_tempo_bucket, \
                                parse_track_energy_bucket, \
                                parse_track_danceability_bucket, \
                                parse_track_speechiness_bucket, \
                                parse_track_valence_bucket, parse_moment_of_week


class Pipeline:
    def __init__(self, deezer_path=None, spotify_path=None,
                 user_thres=None, item_thres=None,
                 verbose=True):
        if deezer_path is None or spotify_path is None:
            raise IOError("You must indicate the paths to both data sets.")
        self.dz_path = deezer_path
        self.sp_path = spotify_path
        self.user_thres = user_thres
        self.item_thres = item_thres
        self.keep_media = []
        self.keep_users = []
        self.sp_data = None
        self.dz_data = None
        self.dz_data_selected = None
        self.sp_data_selected = None
        self.verbose = verbose

    def load_data(self):
        """
        Loads data sets from the indicated paths
        :return: None
        """
        self.sp_data = pd.read_csv(self.sp_path)
        self.dz_data = pd.read_csv(self.dz_path)

    def get_keep_media(self):
        """
        Gets the media id to keep from the Spotify data set.
        Can apply a threshold if wanted to keep the top "item_thres"
        songs in the data set.
        :return: None
        """
        self.keep_media = list(self.sp_data.media_id.map(int))
        if self.item_thres is not None:
            cnt = Counter(self.dz_data.media_id).most_common(self.item_thres)
            cnt_id = [media_id for media_id, _ in cnt]
            self.keep_media = [media_id for media_id in self.keep_media if
                               media_id in cnt_id]

    def get_keep_users(self):
        """
        Gets list of users ID. Can apply a threshold if wanted to keep the
        users that appear at least a given number of time in the data set.
        :return: None
        """
        cnt = Counter(self.dz_data["user_id"])
        if self.user_thres is not None:
            tmp = {uid: val for uid, val in cnt.items()
                   if val > self.user_thres}
        else:
            tmp = {uid: val for uid, val in cnt.items()}
        self.keep_users = list(tmp.keys())

    def filter_data(self):
        """
        Filters the data given the elements to keep self.keep_media and
        self.keep_users
        :return: None
        """
        self.dz_data = self.dz_data[
            self.dz_data["media_id"].isin(self.keep_media)]
        self.dz_data = self.dz_data[
            self.dz_data["user_id"].isin(self.keep_users)]

    def add_features(self):
        """
        Merges the Deezer and Spotify data frames on "media_id"
        :return: None
        """
        self.dz_data = pd.merge(self.dz_data, self.sp_data, on=['media_id'])
        parse_ts_listen(self.dz_data, drop_tmp=True)
        parse_release_date(self.dz_data)
        parse_moment_of_week(self.dz_data)
        parse_track_tempo_bucket(self.dz_data)
        parse_track_age_bucket(self.dz_data)
        parse_media_duration_bucket(self.dz_data)
        parse_user_age(self.dz_data)
        parse_track_energy_bucket(self.dz_data)
        parse_track_valence_bucket(self.dz_data)
        parse_track_speechiness_bucket(self.dz_data)
        parse_track_danceability_bucket(self.dz_data)

    def describe(self):
        """
        Function to describe the data generated
        :return: None
        """
        print("========== DATASET DESCRIPTION ==========")
        print(">>> Shape: {}".format(self.dz_data.shape))
        print(">>> Unique users: {}".format(len(self.keep_users)))
        print(">>> Unique songs: {}".format(len(self.keep_media)))
        print("=========================================")

    def free_memory(self):
        """
        Function to use if you want to free the memory from the temporary
        data sets created.
        :return:  None
        """
        del self.dz_data
        self.dz_data = None
        del self.sp_data
        self.sp_data = None
        print("Memory freed! Yay!!!")

    def make(self):
        """
        Pipeline in itself. Runs all the methods above to generate the final
        data frame to be used.
        :return:  pd.DataFrame | Data frame to use by the model
        """
        t0 = time()
        self.load_data()
        self.get_keep_media()
        self.get_keep_users()
        self.filter_data()
        self.add_features()
        if self.verbose:
            self.describe()
        self.dz_data = self.dz_data.sample(frac=1).reset_index(drop=True)
        print("Running time: {} seconds".format(int(time() - t0)))
        return self.dz_data

    def make_selected(self, columns=None):
        """
        Pipeline in itself. Runs all the methods above to generate the final
        data frame to be used. It also selects specific columns for the analysis
        :return:  pd.DataFrame | Data frame to use by the model with selected
                            columns
        """
        self.load_data()
        self.get_keep_media()
        self.get_keep_users()
        self.filter_data()
        self.add_features()
        if self.verbose:
            self.describe()
        self.dz_data = self.dz_data.sample(frac=1).reset_index(drop=True)

        if columns is None:
            column_selection = ['user_id',
                                'media_id',
                                'moment_of_day',
                                'day_listen',
                                'hour_listen',
                                'spotify_name',
                                'spotify_artist',
                                'energy',
                                'danceability',
                                'tempo',
                                'valence']

        else:
            if not isinstance(columns, (list, np.ndarray)):
                raise IOError("The parameter 'columns' should be a list or"
                              "an np.array.")
            column_selection = columns

        data = self.dz_data[self.dz_data["is_listened"] == 1]
        data = data[column_selection]
        self.dz_data_selected = data
        return data

    def audio_selected(self):
        audio_features = ['media_id',
                          'spotify_name',
                          'spotify_artist',
                          'energy',
                          'tempo',
                          'danceability',
                          'valence']
        self.sp_data_selected = self.sp_data[audio_features]

    def user_song(self, user_id, songs):
        mask = (self.dz_data_selected["user_id"] == user_id) & \
               (self.dz_data_selected["media_id"].isin(songs))
        tmp = self.dz_data_selected[mask]
        tmp = df_summ(df=tmp,
                      index=["user_id",
                             "media_id",
                             "moment_of_day",
                             "hour_listen",
                             'spotify_name',
                             'spotify_artist',
                             'energy',
                             'danceability',
                             'tempo',
                             'valence'],
                      rename="times_listened_in_month",
                      target="media_id",
                      criteria="count")
        return tmp

    def user_day(self, user_id, song_id, days):
        mask = (self.dz_data["user_id"] == user_id) & \
               (self.dz_data["day_listen"].isin(days)) & \
               (self.dz_data["media_id"] == song_id)
        tmp = self.dz_data[mask]
        column_selection = ['user_id',
                            'media_id',
                            'moment_of_day',
                            'day_listen',
                            'hour_listen',
                            'spotify_name',
                            'spotify_artist',
                            'energy',
                            'danceability',
                            'tempo',
                            'valence']
        tmp = tmp[column_selection]
        tmp = tmp.sort_values(by="day_listen")
        return tmp

    def get_user_pivot(self):
        time_agg = time_table(data=self.dz_data_selected,
                              user_id="user_id",
                              song_id="media_id",
                              time_id="moment_of_day",
                              time_threshold=0.05)
        time_rel = time_agg[["user_id", "moment_of_day", "time_rel"]]
        time_vars = time_agg["moment_of_day"].unique()
        _, _, w_nans = u_pivot(data=self.dz_data_selected,
                               user_id="user_id",
                               song_id="media_id",
                               time_id="moment_of_day",
                               time_rel=time_rel,
                               time_vars=time_vars)
        return w_nans

    def run_user_analysis(self, user_id):
        self.audio_selected()
        _, fin = tod_pivot(data=self.dz_data_selected,
                           audio_df=self.sp_data_selected,
                           time_threshold=0.05)
        u = fin[["user_id",
                 "media_id",
                 "spotify_name",
                 "spotify_artist",
                 "mult",
                 "song_sum",
                 "total",
                 "weights"]]
        u = u[u["user_id"] == user_id]
        u = u.sort_values(by="weights", ascending=False)
        return u

    def run_analysis(self):
        self.audio_selected()
        u_fin, _ = tod_pivot(data=self.dz_data_selected,
                             audio_df=self.sp_data_selected,
                             time_threshold=0.05)
        return u_fin


if __name__ == "__main__":
    # Here's an example of how to use this class to generate the data
    # Get the paths to your data
    path_deezer = "my_path_to_deezer_csv"
    path_spotify = "my_path_to_spotify_csv"
    # Create an instance of the Pipeline class
    pipe = Pipeline(deezer_path=path_deezer, spotify_path=path_spotify)
    # Or if you want to give some thresholds
    pipe = Pipeline(deezer_path=path_deezer, spotify_path=path_spotify,
                    user_thres=30, item_thres=250)
    # Now just call "make" to create the data you need !
    my_final_data = pipe.make()
    # You're done! :D
