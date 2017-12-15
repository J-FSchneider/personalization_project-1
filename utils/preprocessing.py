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
    Convert timestamp to datetime.
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


def get_moment_of_day2(ts_listen):
    """
    Gets the moment of the day
    :param ts_listen: int | timestamp of listening
    :return: string | element of MOMENTS.keys()
    """

    # Define the moments of the day
    MOMENTS = {
        "morning": (5, 14),
        "afternoon_evening": (15, 23),
        "late_night": (0, 4)

    }

    # Get the hour of listening
    hour = datetime.datetime.strptime(convert_ts(ts_listen),
                                      "%Y-%m-%d %H:%M:%S").hour

    # Get the corresponding moment
    moment = [k for (k, v) in MOMENTS.items() if v[0] <= hour <= v[1]][0]

    return moment


def get_moment_of_week(ts_listen):
    """
    Gets the moment of the day along with weekday/weekend specifications
    :param ts_listen: int | timestamp of listening
    :return: string | element of MOMENTS.keys()
    """
     #TODO:Define the bins for weekend and weekday

    # Define the moments of the day
    Weekday_MOMENTS = {
        "weekday_morning": (5, 14),
        "weekday_afternoon_to_evening": (15, 23),
        "weekday_late_night": (0, 4)
    }

    # Define the moments of the day for weekends

    Weekend_MOMENTS = {
        "weekend_morning": (5, 14),
        "weekend_afternoon_evening": (15, 23),
        "weekend_late_night": (0, 4)
    }

    # Get the hour of listening
    hour = datetime.datetime.strptime(convert_ts(ts_listen),
                                      "%Y-%m-%d %H:%M:%S").hour
    # Get the day of listening

    day = datetime.datetime.fromtimestamp(ts_listen).weekday()

    # Checking for weekend and getting corresponding bucket
    if day == 4 or day == 5 or day == 6:
        moment = \
        [k for (k, v) in Weekend_MOMENTS.items() if v[0] <= hour <= v[1]][0]
    else:
        moment = \
        [k for (k, v) in Weekday_MOMENTS.items() if v[0] <= hour <= v[1]][0]

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


def parse_moment_of_day2(data):
    """
    Creates a new column in the dataframe containing a string corresponding
    to the moment of the day
    :param data: pd.Dataframe | dataframe containing column "ts_listen"
    :return: pd.Dataframe | same dataframe with new columns added
    """
    if "ts_listen" not in data:
        raise IOError("The input dataframe does not contain "
                      "the column 'ts_listen'")

    data["moment_of_day"] = data["ts_listen"].map(get_moment_of_day2)


def parse_moment_of_week(data):
    """
    Creates a new column in the dataframe containing a string corresponding
    to the moment of the day
    :param data: pd.Dataframe | dataframe containing column "ts_listen"
    :return: pd.Dataframe | same dataframe with new columns added
    """
    if "ts_listen" not in data:
        raise IOError("The input dataframe does not contain "
                      "the column 'ts_listen'")

    data["moment_of_week"] = data["ts_listen"].map(get_moment_of_week)

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

##############################################################################
#                           Track related functions                          #
##############################################################################


def get_track_age_bucket(track_release_date):
    """
    Get the "generation" of a track given its release date
    :param track_release_date: int | format YYYYMMDD : date of release
    :return: string | decade of release of the track
    """

    # Buckets defined thanks to :
    BUCKETS = {
        "old": (1950, 1979),
        #"50s": (1950, 1959),
        #"60s": (1960, 1969),
        #"70s": (1970, 1979),
        "80s": (1980, 1989),
        "90s": (1990, 1999),
        "00s": (2000, 2009),
        "10s": (2010, 2019),
    }

    # Get release year of song: parse YYYYMMDD
    year = track_release_date // 10000

    # Deal with outliers (cf. dates distribution)
    # TODO: remove once the data has been cleaned
    if year == 3000:
        return "00s"  # teenage years of most users
    elif year < 1950:
        return "old"
    else:
        # Get bucket
        bucket = [k for (k, v) in BUCKETS.items() if v[0] <= year <= v[1]][0]
        return bucket


def parse_track_age_bucket(data):
    """
    Creates a new column in the dataframe containing a string corresponding
    to the age bucket of the track
    :param data: pd.Dataframe | dataframe containing column "release_date"
    :return: pd.Dataframe | same dataframe with new columns added
    """
    if "release_date" not in data:
        raise IOError("The input dataframe does not contain "
                      "the column 'release_date'")

    data["track_age_bucket"] = data["release_date"].map(get_track_age_bucket)


def get_track_tempo_bucket(bpm):
    """
    Get the tempo of the track as defined by
    https://fr.wikipedia.org/wiki/Battement_par_minute
    :param tempo: int | Tempo of the track
    :return: string | tempo of the track
    """

    # TODO: Make sure that the track dataframe does not contain bpm with value 0
    # TODO: adjust to be more specific if needed
    BUCKET_BPM = {
        "very_slow": (0, 65),
        "slow": (66, 80),
        "moderate": (81, 99),
        "fast": (100, 120),
        "very_fast": (121, 1000)
    }

    bpm = round(bpm)

    # Get bucket
    bucket = [k for (k, v) in BUCKET_BPM.items() if v[0] <= bpm <= v[1]][0]

    return bucket


def get_track_energy(energy):
    """
    Bucketize the energy levels of the song
    https://developer.spotify.com/web-api/object-model/#audio-features-object
    :param energy: float | the energy measurement of the track
    :return: str | bucket where the track fells in
    """
    energy_bins = {
        "low": (0, 0.33),
        "med": (0.34, 0.66),
        "high": (0.67, 1)
    }

    energy = round(energy, 2)

    # Get bucket
    bucket = [k for (k, v) in energy_bins.items() if v[0] <= energy <= v[1]][0]

    return bucket


def get_track_danceability(danceability):
    """
    Determine if song is danceable or not
    https://developer.spotify.com/web-api/object-model/#audio-features-object
    :param danceability: float | the energy measurement of the track
    :return: str | bucket where the track fells in
    """
    bins = {
        "no-dance": (0, 0.40),
        "yes-dance": (0.41, 1)
    }

    danceability = round(danceability, 2)

    # Get bucket
    bucket = [k for (k, v) in bins.items() if v[0] <= danceability <= v[1]][0]

    return bucket


def parse_track_danceability_bucket(data):
    """
    Creates a new column in the dataframe containing a string corresponding
    to the energy buckect of the track
    :param data: pd.Dataframe | dataframe containing column "deezer_bpm"
    :return: pd.Dataframe | same dataframe with new columns added
    """
    if "danceability" not in data:
        raise IOError("The input dataframe does not contain "
                      "the column 'danceability'")

    data["track_danceability_bucket"] = data["danceability"].\
        map(get_track_danceability)


def get_track_speechiness(speech):
    """
    Determine if song is danceable or not
    https://developer.spotify.com/web-api/object-model/#audio-features-object
    :param danceability: float | the energy measurement of the track
    :return: str | bucket where the track fells in
    """
    bins = {
        "low-speech": (0, 0.33),
        "medium-speech": (0.34, 0.66),
        "high-speech": (0.67, 1)
    }

    speech = round(speech, 2)

    # Get bucket
    bucket = [k for (k, v) in bins.items() if v[0] <= speech <= v[1]][0]

    return bucket


def parse_track_speechiness_bucket(data):
    """
    Creates a new column in the dataframe containing a string corresponding
    to the energy buckect of the track
    :param data: pd.Dataframe | dataframe containing column "deezer_bpm"
    :return: pd.Dataframe | same dataframe with new columns added
    """
    if "speechiness" not in data:
        raise IOError("The input dataframe does not contain "
                      "the column 'speechiness'")

    data["track_speechiness_bucket"] = data["speechiness"].\
        map(get_track_speechiness)


def get_track_valence(valence):
    """
    Bucketize the "positiveness" of the song
    https://developer.spotify.com/web-api/object-model/#audio-features-object
    :param valence: float | the energy measurement of the track
    :return: str | bucket where the track fells in
    """
    bins = {
        "negative": (0, 0.3),
        "positive": (0.31, 0.79),
        "strong positive": (0.8, 1)
    }

    valence = round(valence, 2)

    # Get bucket
    bucket = [k for (k, v) in bins.items() if v[0] <= valence <= v[1]][0]

    return bucket


def parse_track_valence_bucket(data):
    """
    Creates a new column in the dataframe containing a string corresponding
    to the energy buckect of the track
    :param data: pd.Dataframe | dataframe containing column "deezer_bpm"
    :return: pd.Dataframe | same dataframe with new columns added
    """
    if "valence" not in data:
        raise IOError("The input dataframe does not contain "
                      "the column 'valence'")

    data["track_valence_bucket"] = data["valence"].\
        map(get_track_valence)


def parse_track_tempo_bucket(data):
    """
    Creates a new column in the dataframe containing a string corresponding
    to the tempo of the track
    :param data: pd.Dataframe | dataframe containing column "deezer_bpm"
    :return: pd.Dataframe | same dataframe with new columns added
    """
    if "deezer_bpm" not in data:
        raise IOError("The input dataframe does not contain "
                      "the column 'deezer_bpm'")

    data["track_tempo_bucket"] = data["deezer_bpm"].\
        map(get_track_tempo_bucket)


def parse_track_energy_bucket(data):
    """
    Creates a new column in the dataframe containing a string corresponding
    to the energy buckect of the track
    :param data: pd.Dataframe | dataframe containing column "deezer_bpm"
    :return: pd.Dataframe | same dataframe with new columns added
    """
    if "energy" not in data:
        raise IOError("The input dataframe does not contain "
                      "the column 'energy'")

    data["track_energy_bucket"] = data["energy"].\
        map(get_track_energy)


def get_media_duration_bucket(media_duration):
    """
    Create very short/short/medium/long categories with media_duration
    Parameters
    ----------
    df: pd.DataFrame | contains a column media_duration
    """

    BUCKET_DURATION = {
        "very_short_duration": (0, 150),
        "short_duration": (151, 209),
        "medium_duration": (210, 299),
        "long_duration": (300, 10000)
    }

    media_duration = round(media_duration)

    bucket = [k for (k, v) in BUCKET_DURATION.items()
              if v[0] <= media_duration <= v[1]][0]

    return bucket


def parse_media_duration_bucket(data):
    """
    Creates a new column in the dataframe containing a string corresponding
    to the duration bucket of the track
    :param data: pd.Dataframe | dataframe containing column "media_duration"
    :return: pd.Dataframe | same dataframe with new columns added
    """
    if "media_duration" not in data:
        raise IOError("The input dataframe does not contain "
                      "the column 'media_duration'")

    data["track_duration_bucket"] = data["media_duration"].\
        map(get_media_duration_bucket)


def join_survey_data(survey, deezer):
    """
    Parse the survey data into the deezer dataset.
    :param survey: pd.df | the survey data
    :param deezer: pd.df | the deezer data
    :return: pd.df | the deezer data with the parsed survey data
    """
    df = survey.rename(columns={'Name': 'user_id', 'Age': 'user_age',
                                'Gender': 'user_gender',
                                'deezer_id': 'media_id'})

    for index, row in df.iterrows():
        if pd.isnull(row['time']):
            continue
        time = row['time'].split(',')
        if row['user_gender'] == 'Male':
            user_gender = 1
        else:
            user_gender = 0
        if time == None:
            if row['rating'] == 0:
                for i in [1480513129, 1479067262, 1478675619]:
                    new = pd.DataFrame(np.array([[999999, i,
                                                  row['media_id'], 999999,
                                                  0, 20001010, 1, 0, 999,
                                                  1, user_gender,
                                                  row['user_id'], None,
                                                  row['user_age'], 0]]),
                                       columns=['genre_id', 'ts_listen',
                                                'media_id', 'album_id',
                                                'context_type',
                                                'release_date',
                                                'platform_name',
                                                'platform_family',
                                                'media_duration',
                                                'listen_type',
                                                'user_gender', 'user_id',
                                                'artist_id', 'user_age',
                                                'is_listened'])
                    deezer = deezer.append(new)
        elif 'Anytime' in time:
            for i in [1480513129, 1479067262, 1478675619]:
                new = pd.DataFrame(np.array([[999999, i, row['media_id'],
                                              999999, 0, 20001010, 1, 0,
                                              999, 1, user_gender,
                                              row['user_id'], None,
                                              row['user_age'], 0]]),
                                   columns=['genre_id', 'ts_listen',
                                            'media_id', 'album_id',
                                            'context_type',
                                            'release_date',
                                            'platform_name',
                                            'platform_family',
                                            'media_duration',
                                            'listen_type', 'user_gender',
                                            'user_id', 'artist_id',
                                            'user_age',
                                            'is_listened'])
                deezer = deezer.append(new)
        else:
            t_dict = {'Morning': 0, 'Afternoon': 0, 'Evening': 0}
            for t in time:
                t_dict[t] = 1
            for i in [('Morning', 1480513129), ('Afternoon', 1479067262),
                      ('Evening', 1478675619)]:
                new = pd.DataFrame(np.array([[999999, i[1],
                                              row['media_id'], 999999, 0,
                                              20001010, 1, 0, 999, 1,
                                              user_gender,
                                              row['user_id'], None,
                                              row['user_age'],
                                              t_dict[i[0]]]]),
                                   columns=['genre_id', 'ts_listen',
                                            'media_id', 'album_id',
                                            'context_type',
                                            'release_date',
                                            'platform_name',
                                            'platform_family',
                                            'media_duration',
                                            'listen_type', 'user_gender',
                                            'user_id', 'artist_id',
                                            'user_age',
                                            'is_listened'])
                deezer = deezer.append(new)

    return deezer


if __name__ == "__main__":
    path = "my/path/to/dataset"
    # Load data
    data = pd.read_csv(path)
    # Transform the time columns
    parse_ts_listen(data)
    parse_release_date(data)
