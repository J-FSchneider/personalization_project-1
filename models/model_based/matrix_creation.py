import numpy as np
import pandas as pd
# TODO: unused imports to be deleted

def sample_data_by_freq(data):
    # TODO: Add standard docstring
    # This function returns the items and users with the most frequency in the is_listened column

    # List of unique users (with media count)
    user_rating_count = data.groupby('user_id')['media_id'].nunique()
    # List of users in ascending order (by number of songs, not unique, listened to)
    user_rating_count = user_rating_count.sort_values(axis=0, ascending=False)
    # select 10,000 most active users
    # TODO: check variable u_sel never used
    user = user_rating_count[:10000].index
    
    # List of unique media ids (with user count)
    media_count = data.groupby('media_id')['user_id'].nunique()
    # List of media in ascending order (by number of users)
    media_count = media_count.sort_values(axis=0, ascending=False)
    # Select top 100 items
    # TODO: check variable m_sel never used
    media = media_count[:100].index
    # return list with media_ids

    return user, media

def sample_data_50_50(data):
    user_rating_count = data.groupby('user_id')['media_id'].nunique()
    # List of users in ascending order (by number of songs, not unique, listened to)
    user_rating_count = user_rating_count.sort_values(axis=0, ascending=False)
    # select 10,000 most active users
    # TODO: check variable u_sel never used
    user = user_rating_count[:10000].index
    

    # List of unique media ids (with user count)
    media_count = data.groupby('media_id')['user_id'].nunique()
    # List of media in ascending order (by number of users)
    media_count = media_count.sort_values(axis=0, ascending=False)
    # Select top 100 items

    # TODO: check variable m_sel never used
    media1 = media_count[:50]
    media2 = media_count[51:10000].sample(n=50)
    media = media1.append(media2).index

    # return list with media_ids

    return user, media

def sample_artist_by_freq(data):

    ##### IMPLICATIONS => DIFFERENT RATINGS 

    ###### NOT READY YET ########
    user_rating_count = data.groupby('user_id')['artist_id'].nunique()
    # List of users in ascending order (by number of songs, not unique, listened to)
    user_rating_count = user_rating_count.sort_values(axis=0, ascending=False)
    # select 10,000 most active users
    # TODO: check variable u_sel never used
    user = user_rating_count[:10000].index
    
    # List of unique media ids (with user count)
    media_count = data.groupby('artist_id')['user_id'].nunique()
    # List of media in ascending order (by number of users)
    media_count = media_count.sort_values(axis=0, ascending=False)
    # Select top 100 items
    # TODO: check variable m_sel never used
    media = media_count[:100].index
    # return list with media_ids

    return user, media    

def rating_matrix(data, user, media):
    # TODO: Add standard docstring
	# returns a rating matrix in wide format given a long format table, sampled by the specified user and media ids
    # data: the raw_data to build the matrix from
    # user: an df of user_ids to select from data
    # media: an df of media_ids which to include into the ratings matrix

    # sample users
    users = data[data['user_id'].isin(user)]

    # sample items
    items = users[users['media_id'].isin(media)]

    # drop duplicates (assumption: keep the first of an item)
    matrix = items.drop_duplicates(subset=['media_id', 'user_id'], keep='first')

    # get long format into wide format
    # TODO: check memory usage, improve to sparse matrix storage if necessary
    matrix = matrix.pivot(index='user_id',
                          columns='media_id',
                          values='is_listened')

    return matrix
