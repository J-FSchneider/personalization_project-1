# File description
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
    Dreezer's database. The first popularity metric is "row_pop"
    which is a notion of popularity based on the number of times
    that the artist was played (in other words the row count).
    The second metric is a notion of how many distinct users
    was the artist listened by (the unique user count)
    
    :param df: Dreezer's database in the DataFrame format
    :param target: the name of the identifier columns
    :param column: the name of the column where the unique count
    is gong to be perform
    :return master_table: a DataFrame at the artist level that
    contains the popularity features
    """
    new_column1 = "row_pop"
    criteria1 = "count"
    var_agg = {target: criteria1}
    row_pop = df.groupby(by=target).agg(var_agg)
    row_pop.columns = [new_column1]
    row_pop = row_pop.reset_index()
    master_table = row_pop.sort_values(new_column1, ascending=False)

    new_column2 = "user_pop"
    criteria2 = "nunique"
    var_agg = {column: criteria2}
    user_pop = df.groupby(by=target).agg(var_agg)
    user_pop.columns = [new_column2]
    user_pop = user_pop.reset_index()

    master_table = pd.DataFrame.merge(master_table, user_pop, on=target, how="left")
    return master_table
