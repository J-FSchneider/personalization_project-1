import pandas as pd
from collections import Counter
from time import time


class Pipeline():
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
