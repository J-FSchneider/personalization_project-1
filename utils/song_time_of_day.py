from utils.preprocessing import *
import warnings


def song_time_of_day(data, user_number, min_num_songs=2):
    """


    :param data: dataset to use
    :param user_number: the user for which the information is required
    :param min_num_songs: song cutoff for the user
    :return:
    """

    warnings.filterwarnings("ignore")
    if user_number not in data['user_id']:
        raise IOError("The input dataframe does not contain the user")
    else:

        user = data.loc[data['user_id'] == user_number]
        parse_ts_listen(user)
        parse_moment_of_day2(user)

    if min_num_songs >= max(user.groupby(['media_id', 'moment_of_day']).count()['ts_listen']) - 2:
        raise IOError("Minimum number of songs is more than the maximum songs listened by the user")

    else:

        df = user.groupby(['media_id', 'moment_of_day']).count()[
            user.groupby(['media_id', 'moment_of_day']).count()['ts_listen'] > min_num_songs]

        df = df.reset_index()

        user_pivot = df.pivot(index='moment_of_day', columns='media_id', values='genre_id')

        user_pivot = user_pivot.fillna(0)

        user_pivot = user_pivot.applymap(lambda x: 1 if x > 0 else 0)

        tpose = user_pivot.transpose()

        tpose['mult'] = tpose[tpose.column[0]] * tpose[tpose.column[1]]

        percent = tpose.mult.sum() / user_pivot.shape[1]

        return percent
