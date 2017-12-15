from utils.preprocessing import *
import warnings
from scipy.stats import chi2_contingency


def song_time_of_day(data, user_number, min_num_songs=2):
    """
    The function calculates the ratio of common songs to total songs a given
    user listens to during different time periods in the day

    :param data: pd.dataFrame| Data set to use
    :param user_number: int| the user for which the information is required
    :param min_num_songs: int|  song cutoff for the user
    :return: The ratio of songs
    """

    warnings.filterwarnings("ignore")
    if user_number not in data['user_id']:
        raise IOError("The input dataframe does not contain the user")
    else:

        user = data.loc[data['user_id'] == user_number]
        parse_ts_listen(user)
        parse_moment_of_day2(user)

    if min_num_songs >= max(user.groupby(['media_id', 'moment_of_day']).count()['ts_listen']) - 2:
        raise IOError("Minimum number of songs is more "
                      "than the maximum songs listened by the user")

    else:

        df = user.groupby(['media_id', 'moment_of_day']).count()[
            user.groupby(['media_id', 'moment_of_day']).count()['ts_listen'] > min_num_songs]

        df = df.reset_index()

        user_pivot = df.pivot(index='moment_of_day',
                              columns='media_id',
                              values='genre_id')

        user_pivot = user_pivot.fillna(0)

        user_pivot = user_pivot.applymap(lambda x: 1 if x > 0 else 0)

        tpose = user_pivot.transpose()

        tpose['mult'] = tpose[tpose.column[0]] * tpose[tpose.column[1]] * tpose[tpose.column[2]]

        percent = tpose.mult.sum() / user_pivot.shape[1]

        return percent


def chisq_test(data, feature1, feature2):
    """
    Calculates the chi-square test of independence of two events.

    :param data: pd.DataFrame | Data set for analysis
    :param feature1: str | feature1 to check for independence
    :param feature2: str | feature2 to check for independence
    :return: returns the original contingency table, chi sq statistic, p value
    degrees of freedom and the table of expected values
    """

    chi = pd.crosstab(data[feature1], data[feature2])
    val = chi2_contingency(chi, correction=True, lambda_=None)

    return chi, val
