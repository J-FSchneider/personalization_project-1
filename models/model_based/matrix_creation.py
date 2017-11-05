import numpy as np


def binary_matrix_popular_items(data, n_users=None, n_items=100, min_rating=10):
    """
    Creates a sampled matrix from the original data matrix, by sampling
    n_users users and n_items media
    :param data: pd.DataFrame | this is the input data that should be sampled
    :param n_users: None or int | number of users to sample. If None, all users
                                   are kept.
    :param n_items: int | number of items to keep
    :param min_rating: int | minimum rating per user
    :return matrix: pd.DataFrame | Ratings matrix with the "is_listened" feature
                                   of the n_items most popular items
    """
    
    # Get items by popularity
    media_count = data.groupby('media_id')['user_id'].nunique()
    media_count = media_count.sort_values(axis=0, ascending=False)
    
    # Select top 100 items
    media = media_count[:n_items].index
    
    # Filter data
    matrix = data[data['media_id'].isin(media)]
    
    # Make pivot table
    matrix = matrix.pivot_table(index='user_id', columns='media_id',
                                values='is_listened', aggfunc='median')
    # Filter users with less than 10 ratings in items
    m = matrix.count(axis=1) > min_rating

    matrix = matrix[matrix.index.isin(m[m==True].index)]
    matrix = matrix.applymap(lambda x: x if np.isnan(x) else round(x))

    if n_users:
        if n_users > len(matrix):
            print('Matrix has less than ' + str(n_users) + ' users. '
                  'Return maximum Matrix!')
            return matrix
        else:
            return matrix[:n_users]
    else: 
        return matrix


def binary_matrix_50_50(data, n_users=None, n_items=100, min_rating=10):
    """
    Creates a sampled matrix from the original data matrix, by sampling
    n_users users and n_items media from the long tail and the most popular
    items (50% popular and 50% in long tail).
    :param data: pd.DataFrame | this is the input data that should be sampled
    :param n_users: None or int | number of users to sample. If None, all users
                                   are kept.
    :param n_items: int | number of items to keep
    :param min_rating: int | minimum rating per user
    :return matrix: pd.DataFrame | Ratings matrix with the "is_listened" feature
                                   of the n_items selected
    """
    # Get items by popularity
    media_count = data.groupby('media_id')['user_id'].nunique()
    media_count = media_count.sort_values(axis=0, ascending=False)
    
    # Select top 50%/50% items from popular and long tail
    media1 = media_count[:int(n_items/2)]
    media2 = media_count[int(n_items/2) + 1:10000].sample(n=int(n_items/2))
    media = media1.append(media2).index
    
    # Filter data
    matrix = data[data['media_id'].isin(media)]
    
    # Make pivot table
    matrix = matrix.pivot_table(index='user_id', columns='media_id',
                                values='is_listened', aggfunc='median')
    # Filter users with less than 10 ratings in items
    m = matrix.count(axis=1) > min_rating
    matrix = matrix[matrix.index.isin(m[m==True].index)]
    # Round the values to avoid media = 0.5
    matrix = matrix.applymap(lambda x: x if np.isnan(x) else round(x))
    
    if n_users:
        if n_users > len(matrix):
            print('Matrix has less than ' + str(n_users) + ' users. '
                  'Return maximum Matrix!')
            return matrix
        else:
            return matrix[:n_users]
    else: 
        return matrix


def hit_rate_matrix_popular_items(data,
                                  n_users=None,
                                  n_items=100,
                                  min_rating=10):
    """
    Creates a sampled matrix from the original data matrix, by sampling
    n_users users and n_items media
    :param data: pd.DataFrame | this is the input data that should be sampled
    :param n_users: None or int | number of users to sample. If None, all users
                                   are kept.
    :param n_items: int | number of items to keep
    :param min_rating: int | minimum rating per user
    :return matrix: pd.DataFrame | Ratings matrix with the "hit_rate" metric
                                   for the users and items kept
    """
    # Get items by popularity
    media_count = data.groupby('media_id')['user_id'].nunique()
    media_count = media_count.sort_values(axis=0, ascending=False)
    
    # Select top 100 items
    media = media_count[:n_items].index
    
    # Filter data
    matrix = data[data['media_id'].isin(media)]
    
    # Make pivot table
    matrix = matrix.pivot_table(index='user_id', columns='media_id',
                                values='is_listened', aggfunc='mean')
    # Filter users with less than 10 ratings in items
    m = matrix.count(axis=1) > min_rating
    matrix = matrix[matrix.index.isin(m[m==True].index)]
    
    if n_users:
        if n_users > len(matrix):
            print('Matrix has less than ' + str(n_users) + ' users.'
                  ' Return maximum Matrix!')
            return matrix
        else:
            return matrix[:n_users]
    else: 
        return matrix


def hit_rate_matrix_50_50(data, n_users=None, n_items=100, min_rating=10):
    """
    Creates a sampled matrix from the original data matrix, by sampling
    n_users users and n_items media from the long tail and the most popular
    items (50% popular and 50% in long tail).
    :param data: pd.DataFrame | this is the input data that should be sampled
    :param n_users: None or int | number of users to sample. If None, all users
                                  are kept.
    :param n_items: int | number of items to keep
    :param min_rating: int | minimum rating per user
    :return matrix: pd.DataFrame | Ratings matrix with the "hit_rate" metric
                                   for the users and items kept
    """
    # Get items by popularity
    media_count = data.groupby('media_id')['user_id'].nunique()
    media_count = media_count.sort_values(axis=0, ascending=False)
    
    # Select top 50%/50% items from popular and long tail
    media1 = media_count[:int(n_items/2)]
    media2 = media_count[int(n_items/2) + 1: 10000].sample(n=int(n_items/2))
    media = media1.append(media2).index
    
    # Filter data
    matrix = data[data['media_id'].isin(media)]
    
    # Make pivot table
    matrix = matrix.pivot_table(index='user_id', columns='media_id',
                                values='is_listened', aggfunc='mean')
    # Filter users with less than 10 ratings in items
    m = matrix.count(axis=1) > min_rating
    matrix = matrix[matrix.index.isin(m[m==True].index)]
    
    if n_users:
        if n_users > len(matrix):
            print('Matrix has less than ' + str(n_users) + ' users.'
                  ' Return maximum Matrix!')
            return matrix
        else:
            return matrix[:n_users]
    else: 
        return matrix
