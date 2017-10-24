"""
This file contains some functions one can use to preprocess the dataset.
If functions copied outside this file, do not forget imports.
"""

import pandas as pd
import datetime

##############################################################################
#                           Time related functions                           #
##############################################################################


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


def get_moment_of_day(ts_listen):
    """
    Gets the moment of the day
    :param ts_listen: int | timestamp of listening
    :return: string | element of MOMENTS.keys()
    """

    # Define the moments of the day
    MOMENTS = {
        "early_morning": (6, 8),
        "morning": (9, 12),
        "day": (12, 17),
        "evening": (17, 23),
        "late_night": (0, 5)
    }

    # Get the hour of listening
    hour = datetime.datetime.strptime(convert_ts(ts_listen),
                                      "%Y-%m-%d %H:%M:%S").hour

    # Get the corresponding moment
    moment = [k for (k, v) in MOMENTS.items() if v[0] <= hour <= v[1]][0]

    return moment


def parse_moment_of_day(data):
    """
    Creates a new column in the dataframe containing a string corresponding
    to the moment of the day
    :param data: pd.Dataframe | dataframe containing column "ts_listen"
    :return: pd.Dataframe | same dataframe with new columns added
    """
    if "ts_listen" not in data:
        raise IOError("The input dataframe does not contain "
                      "the column 'ts_listen'")

    data["moment_of_day"] = data["ts_listen"].map(get_moment_of_day)


##############################################################################
#                           User related functions                           #
##############################################################################

def get_user_age_bucket(user_age):
    """
    Bucketize the age of the user
    :param user_age: int | age of the user
    :return: string | bucket of age
    """

    # Define age buckets: ages between 18 and 30
    # TODO: Change keys if want to simplify
    BUCKETS = {
        "[18-21]": (18, 21),
        "[22-25]": (22, 25),
        "[26-30]": (26, 30)
    }

    # Get user bucket
    bucket = [k for (k, v) in BUCKETS.items() if v[0] <= user_age <= v[1]][0]

    return bucket


def parse_user_age(data):
    """
    Creates a new column in the dataframe containing a string corresponding
    to the age bucket of the user
    :param data: pd.Dataframe | dataframe containing column "user_age"
    :return: pd.Dataframe | same dataframe with new columns added
    """
    if "user_age" not in data:
        raise IOError("The input dataframe does not contain "
                      "the column 'user_age'")

    data["user_age_bucket"] = data["user_age"].map(get_user_age_bucket)


if __name__ == "__main__":
    path = "my/path/to/dataset"
    # Load data
    data = pd.read_csv(path)
    # Transform the time columns
    parse_ts_listen(data)
    parse_release_date(data)
