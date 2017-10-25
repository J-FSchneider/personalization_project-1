from utils.preprocessing import *


def time_of_day_encoded(data):
    """
    Convert "time_of_day" variable into dummy/indicator variables
    :param data: pd.DataFrame
    :return: pd.DataFrame | input dataframe with additional columns
    """
    # TODO: check redundant work with CleanTransformer instances
    parse_ts_listen(data)
    parse_moment_of_day(data)

    # TODO: this check is useless then if you call the function above
    if "moment_of_day" not in data:
        raise IOError("The input dataframe does not contain "
                      "the column 'moment_of_day'")

    data = pd.concat([data, pd.get_dummies(data.moment_of_day)], axis=1)

    return data


def age_bucket_encoded(data):
    """
    Convert "age_bucket" variable into dummy/indicator variables
    :param data: pd.DataFrame
    :return: pd.DataFrame | input dataframe with additional columns
    """
    parse_user_age(data)

    # TODO: this check is useless then if you call the function above
    if "user_age_bucket" not in data:
        raise IOError("The input dataframe does not contain "
                      "the column 'user_age_bucket'")

    # Joining the new dataset with the encoding to the original data
    data = pd.concat([data, pd.get_dummies(data.user_age_bucket)], axis=1)

    return data

# TODO: change name of function to more explicit one
def modified_data(data) :
    """
    This function calls the functions mentioned above and stores the values in the dataset 
    to output the new dataset with the encoded values. It also deletes the 
    'user_age_bucket' and 'moment_of_day' column as they are not required anymore
    """
    # TODO: change docstring above to a standard one
    data = time_of_day_encoded(data) #storing new dataset
    data = age_bucket_encoded(data) #storing new dataset
    # TODO: use data.drop instead of del
    del data['user_age_bucket'] #removing the 'user_age_bucket' column
    del data['moment_of_day'] #Removing the 'moment_of_day' column

    # TODO: are you sure that is the final dataframe one can use to do the K-NN analysis?
    return data

if __name__ == "__main__":
    #TODO: show how these functions can be used to generate the desired df from scratch
