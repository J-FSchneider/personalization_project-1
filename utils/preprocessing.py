"""
This file contains some functions one can use to preprocess the dataset.
If functions copied outside this file, do not forget imports.
"""

import pandas as pd
import datetime


def convert_ts(ts):
    """
    Convert timestamp to datetime
    :param ts: int | timestamp in seconds since 1st of January 1970
    :return:
    """
    formatted_date = datetime.datetime.fromtimestamp(ts)\
        .strftime('%Y-%m-%d %H:%M:%S')
    return formatted_date


def parse_ts_listen(data, drop_tmp=False):
    """
    Parse the data "ts_listened" column and creates new columns
    Modifies the dataframe in-place
    :param data: pd.Dataframe | dataframe containing column "ts_listen"
    :param drop_tmp: boolean | set to True if want to drop the temporary column
    :return: pd.Dataframe | same dataframe with new columns added
    """
    if "ts_listen" not in data:
        raise IOError("The input dataframe does not contain "
                      "the column 'ts_listen'")
    # Map function
    data["converted_ts"] = data["ts_listen"].map(convert_ts)
    # Add new columns
    data["year_listen"] = pd.DatetimeIndex(data['converted_ts']).year
    data["month_listen"] = pd.DatetimeIndex(data['converted_ts']).month
    data["day_listen"] = pd.DatetimeIndex(data['converted_ts']).day
    data["hour_listen"] = pd.DatetimeIndex(data['converted_ts']).hour
    # Drop temporary column
    if drop_tmp:
        data.drop("converted_ts", axis=1)


def parse_release_date(data):
    """
    Parse "release_date" time format in the dataset
    Format of time: YYYYMMDD
    Modifies the dataframe in-place
    :param data: pd.Dataframe | dataframe containing column "release_date"
    :return: pd.Dataframe | same dataframe with new columns added
    """
    if "release_date" not in data:
        raise IOError("The input dataframe does not contain "
                      "the column 'release_date'")
    # Add new columns
    data["year_release"] = data["release_date"] // 10000
    data["month_release"] = (data["release_date"] // 100) % 100
    data["day_release"] = data["release_date"] % 100

if __name__ == "__main__":
    path = "my/path/to/dataset"
    # Load data
    data = pd.read_csv(path)
    # Transform the time columns
    parse_listen_type(data)
    parse_release_date(data)
