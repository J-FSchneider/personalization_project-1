import copy
import pandas as pd
from utils.preprocessing import parse_release_date, parse_ts_listen


class CleanTransformer():
    """
    This class allows you to transform the date columns of your dataframe
    and clean their values (taking out the outliers and strange values)
    """
    def __init__(self):
        self.data = None
        self.columns = None
        self.fitted = False
        self.transformed = False

    def fit(self, data):
        """
        Fits the instance to the dataframe data
        :param data: pd.Dataframe | dataframe to clean and transform
        :return: None
        """
        self.fitted = True
        self.data = copy.copy(data)
        self.columns = self.data.columns

    def transform(self):
        """
        Transform the attribute data by adding new columns.
        The method used are defined in the 'utils' module.
        :return: None
        """
        self.transformed = True
        parse_ts_listen(self.data, drop_tmp=True)
        parse_release_date(self.data)

    def fit_transform(self, data):
        """
        Apply the functions fit and transform one after the other
        :param data: pd.Dataframe | dataframe to clean and transform
        :return: self
        """
        self.fit(data)
        self.transform()
        return self

    # TODO: add other features to clean if needed
    def clean_ts_listen(self):
        self.data = self.data[(2011 <= self.data["year_listen"]) &
                              (self.data["year_listen"] <= 2017)]
        self.data = self.data[(1 <= self.data["month_listen"]) &
                              (self.data["month_listen"] <= 12)]
        self.data = self.data[(1 <= self.data["day_listen"]) &
                              (self.data["day_listen"] <= 31)]
        self.data = self.data[(0 <= self.data["hour_listen"]) &
                              (self.data["hour_listen"] <= 23)]

    def clean_release_date(self):
        # TODO: verify min and max year release allowed
        self.data = self.data[(1950 <= self.data["year_release"]) &
                              (self.data["year_release"] <= 2017)]
        self.data = self.data[(1 <= self.data["month_release"]) &
                              (self.data["month_release"] <= 12)]
        self.data = self.data[(1 <= self.data["day_release"]) &
                              (self.data["day_release"] <= 31)]

    def clean_media_duration(self):
        # TODO: change threshold if needed
        self.data = self.data[self.data["media_duration"] <= 700]

    def remove_outliers(self):
        # TODO: to implement
        pass

    def clean(self):
        if not self.transformed:
            raise IOError("You need first to transform your dataframe.")
        self.clean_ts_listen()
        self.clean_release_date()
        return self.data

    def dump(self, path=None):
        if path is None:
            self.data.to_csv("cleaned_data.csv")
        else:
            self.data.to_csv(path)