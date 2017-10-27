"""
This file contains some functions one can use to convert the categorical variables to
encoded variables.If functions copied outside this file, do not forget imports.
"""

from utils.preprocessing import *
import pandas as pd



def time_of_day_encoded(data):
    """
    Convert "time_of_day" variable into dummy/indicator variables
    :param data: pd.DataFrame
    :return: pd.DataFrame | input dataframe with additional columns
    """
    # TODO: check redundant work with CleanTransformer instances
    parse_ts_listen(data)
    parse_moment_of_day(data)

    data = pd.concat([data, pd.get_dummies(data.moment_of_day)], axis=1)

    return data


def age_bucket_encoded(data):
    """
    Convert "age_bucket" variable into dummy/indicator variables
    :param data: pd.DataFrame
    :return: pd.DataFrame | input dataframe with additional columns
    """
    parse_user_age(data)

    # Joining the new dataset with the encoding to the original data
    data = pd.concat([data, pd.get_dummies(data.user_age_bucket)], axis=1)

    return data


def user_encoded_data(data):
    """
    Convert the dataset to a dataset with encoded columns for age buckets and time of day
    :param data: pd.DataFrame
    :return: pd.DataFrame | input dataframe with additional columns
    """
    data = time_of_day_encoded(data) #storing new dataset
    data = age_bucket_encoded(data) #storing new dataset
    data = data.drop('user_age_bucket',1) # removing the 'user_age_bucket' column
    data = data.drop('moment_of_day',1) # removing the 'moment_of_day' column
  
    return data



if __name__ == "__main__":

    path = "my/path/to/dataset"
    # Load data
    data = pd.read_csv(path)
    #transform data
    time_of_day_encoded(data)
    age_bucket_encoded(data)
    user_encoded_data(data)
