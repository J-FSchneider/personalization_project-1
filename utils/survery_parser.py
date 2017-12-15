import pandas as pd


def create_survey_ids(survey):
    """
    Create numeric ids
    :param survey:
    :return:
    """


    x = survey['Name'].unique().tolist()
    ids = {}
    i = 0
    for s in x:
        ids[s] = 9999000 + i
        i += 1
    return ids



def join_survey_data(survey, deezer):
    """
    Joins the survey dataframe with the deezer dataframe
    :param survey:
    :param deezer:
    :return:
    """


    df = survey.rename(columns={'Age': 'user_age', 'Gender': 'user_gender',
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
        elif 'Anytime' in time:
            for i in [1480513129, 1479067262, 1478675619]:
                new = pd.DataFrame(np.array([[999999, i, row['media_id'],
                                              999999, 0, 20001010, 1, 0, 999,
                                              1, user_gender,
                                              row['user_id'], None,
                                              row['user_age'], 0]]),
                                   columns=['genre_id', 'ts_listen',
                                            'media_id', 'album_id',
                                            'context_type',
                                            'release_date', 'platform_name',
                                            'platform_family',
                                            'media_duration',
                                            'listen_type', 'user_gender',
                                            'user_id', 'artist_id', 'user_age',
                                            'is_listened'])
                deezer = deezer.append(new)
        else:
            t_dict = {'Morning': 0, 'Afternoon': 0, 'Evening': 0}
            for t in time:
                t_dict[t] = 1
            for i in [('Morning', 1480513129), ('Afternoon', 1479067262),
                      ('Evening', 1478675619)]:
                new = pd.DataFrame(np.array([[999999, i[1], row['media_id'],
                                              999999, 0, 20001010, 1, 0, 999,
                                              1, user_gender,
                                              row['user_id'], None,
                                              row['user_age'], t_dict[i[0]]]]),
                                   columns=['genre_id',
                                            'ts_listen',
                                            'media_id',
                                            'album_id',
                                            'context_type',
                                            'release_date', 'platform_name',
                                            'platform_family',
                                            'media_duration',
                                            'listen_type', 'user_gender',
                                            'user_id', 'artist_id', 'user_age',
                                            'is_listened'])
                deezer = deezer.append(new)

    return deezer

def parse_survey_data(survey_path, deezer_path, output_path):
    """

    :param survey_path:
    :param deezer_path:
    :param output_path:
    :return:
    """
    survey = pd.read_csv(survey_path)
    ids = create_survey_ids(survey)
    survey['user_id'] = survey['Name'].map(ids)
    deezer_data = pd.read_csv(deezer_path, nrows=1)
    deezer_data = join_survey_data(survey,deezer_data)
    deezer_data2 = pd.read_csv(deezer_data)
    deezer_data = deezer_data.append(deezer_data2)
    deezer_data.to_csv(output_path, index_label=False)

    return None