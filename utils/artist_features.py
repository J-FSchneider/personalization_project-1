"""
    Here are some functions that take the Dreezer data set and
    create a new table at the artist level that incorporates
    new features from the data set
"""

# Import Packages
# import numpy as np
import pandas as pd

# Below are the functions that create the popularity features for the artists


def artist_pop(df, target, column):
    """
    This function constructs popularity metrics for the artist from
    the Deezer dataset.

    The first popularity metric is "row_pop"
    which is a notion of popularity based on the number of times
    that the artist was played (in other words the row count).

    The second metric is a notion of how many distinct users
    was the artist listened by (the unique user count).

    :param df: pd.DataFrame | DataFrame containing the columns
                              "target" and "column"
    :param target: str | The name of the identifier columns
    :param column: The name of the column where the unique count
                   is gong to be perform
    :return master_table: pd.DataFrame | DataFrame at the artist level
                          that contains the popularity features
    """
    # TODO: change name of parameter column as it can be confusing

    if target not in df or column not in df:
        raise IOError("One of the column names entered is not contained"
                      "in the dataframe.")

    # TODO: Delete intermediary variables, unecessary
    new_column1 = "row_pop"
    criteria1 = "count"
    var_agg = {target: criteria1}
    row_pop = df.groupby(by=target).agg(var_agg)
    row_pop.columns = [new_column1]
    row_pop = row_pop.reset_index()
    master_table = row_pop.sort_values(new_column1, ascending=False)

    # TODO: Delete intermediary variables, unecessary
    new_column2 = "user_pop"
    criteria2 = "nunique"
    var_agg = {column: criteria2}
    user_pop = df.groupby(by=target).agg(var_agg)
    user_pop.columns = [new_column2]
    user_pop = user_pop.reset_index()

    master_table = pd.DataFrame.merge(master_table, user_pop, on=target,
                                      how="left")
    return master_table


def popularity_cat(df, target, column):
    """
    This function generates the categorical columns for the artist's
    popularity features by first generating categories based on certain
    thresholds and then pivoting them into new columns.

    :param df: pd.DataFrame | DataFrame containing the columns
                              "target" and "column"
    :param target: str | The name of the identifier columns
    :param column: str | The name of the column where the threshold
                         is going to be applied
    :return df: pd.DataFrame | DataFrame with the artist's identifier
                               and the category columns
    """
    # TODO: change name of parameter column as it can be confusing

    if target not in df or column not in df:
        raise IOError("One of the column names entered is not contained"
                      "in the dataframe.")

    # TODO: Delete intermediary variables, unecessary
    n = len(df)
    cat1 = "unpop"
    cat2 = "known"
    cat3 = "pop"
    pos = df.columns.get_loc(column)
    threshold1 = 15
    threshold2 = 157
    cat_column = "artist"
    df[cat_column] = cat1
    cat_pos = df.columns.get_loc(cat_column)
    # TODO: .iloc is very inefficient, and no need to use for loop, instead use pandas' slicing functions
    """
    for i in range(n):
        if df.iloc[i, pos] > threshold2:
            df.iloc[i, cat_pos] = cat3
        elif (df.iloc[i, pos] > threshold1) & (df.iloc[i, pos] <= threshold2):
            df.iloc[i, cat_pos] = cat2
    """
    # pop artists
    df.loc[df[column] > threshold1, cat_column] = cat2
    # know artists
    df.loc[df[column] > threshold2, cat_column] = cat3

    # TODO: remove unecessary temp variables, just makes more lines to read
    column_selection = [target, cat_column]
    df = df[column_selection]
    df = pd.get_dummies(df)

    return df


def art_hits(df, target, column, threshold=0.75):
    """
    This function calculates the number of  songs that generate a certain
    percentage of the total number of songs that the artist was played upon.

    :param df: Deezer's database in the DataFrame format
    :param target: the name of the identifier columns
    :param column: the name of the column where filtering is going to take place
    :param threshold: is the percentage of interest
    :return result: a DataFrame which contains the artist's id and the no. of songs that
    pass the threshold
    """
    # TODO: take into consideration the previous comments for this function too
    criteria1 = "count"
    criteria2 = "sum"
    new_col = "song_count"
    var_agg = {column: criteria1}
    hits = df.groupby(by=[target, column]).agg(var_agg)
    hits.columns = [new_col]
    hits = hits.reset_index()
    hits = hits.sort_values([target, new_col], ascending=[1, 0])

    var_agg = {new_col: criteria2}
    total = hits.groupby(by=target).agg(var_agg)
    total_col = "tot_count"
    total.columns = [total_col]
    total = total.reset_index()

    hits = hits.merge(total, on=target, how="left")
    per_col = "per_tot"
    hits[per_col] = hits[new_col] / hits[total_col]

    n = len(hits)
    id_pos = hits.columns.get_loc(target)
    acc_col = "cul_tot"
    flag_col = "flag"
    hits[acc_col] = hits[per_col]
    acc_pos = hits.columns.get_loc(acc_col)
    per_pos = hits.columns.get_loc(per_col)
    hits[flag_col] = 1
    flag_pos = hits.columns.get_loc(flag_col)
    # TODO: as in previous functions, get rid of .iloc
    for i in range(1, n, 1):
        if hits.iloc[i, id_pos] == hits.iloc[i - 1, id_pos]:
            hits.iloc[i, acc_pos] = hits.iloc[i - 1, acc_pos] \
                                    + hits.iloc[i, per_pos]

    for j in range(n - 1):
        if hits.iloc[j, id_pos] == hits.iloc[j - 1, id_pos]:
            if (hits.iloc[j, acc_pos] > threshold) & (hits.iloc[j, acc_pos] < 1):
                hits.iloc[j + 1, flag_pos] = 0

    criteria = "sum"
    column = "flag"
    var_agg = {column: criteria}
    result = hits.groupby(by=target).agg(var_agg)
    result.columns = ["no_songs"]
    result = result.reset_index()

    return result


def hits_cat(df, target):
    """
    This function generates the categorical columns for the artist's "hits" features
    by first generating categories based on certain thresholds and then pivoting
    them into new columns
    :param df: Dreezer's database in the DataFrame format
    :param target: the name of the identifier columns
    :return df:a DataFrame with the artist's identifier and the hits category columns
    """
    # TODO: take into consideration the previous comments for this function too
    column = "no_songs"
    n = len(df)
    cat1 = "one"
    cat2 = "medium"
    cat3 = "many"
    pos = df.columns.get_loc(column)
    threshold1 = 3
    threshold2 = 15
    cat_column = "hits"
    df[cat_column] = cat1
    cat_pos = df.columns.get_loc(cat_column)
    # TODO: same here
    for i in range(n):
        if df.iloc[i, pos] > threshold2:
            df.iloc[i, cat_pos] = cat3
        elif (df.iloc[i, pos] > threshold1) & (df.iloc[i, pos] <= threshold2):
            df.iloc[i, cat_pos] = cat2

    column_selection = [target, cat_column]
    df = df[column_selection]
    df = pd.get_dummies(df)

    return df

if __name__ == "__main__":
    #TODO: show how these functions can be used to generate the desired df