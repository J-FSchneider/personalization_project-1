"""
    Here are some functions that take the Dreezer data set and
    create a new table at the artist level that incorporates
    new features from the data set
"""

# Import Packages
# import numpy as np
import pandas as pd

# Below are the functions that create the popularity features for the artists


def artist_pop(df, target, agg_col):
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
    :param agg_col: The name of the column where the unique count
                   is gong to be perform
    :return master_table: pd.DataFrame | DataFrame at the artist level
                          that contains the popularity features
    """

    if target not in df or agg_col not in df:
        raise IOError("One of the column names entered is not contained"
                      "in the dataframe.")

    row_pop = df.groupby(by=target).agg({target: "count"})
    row_pop.columns = ["row_pop"]
    row_pop = row_pop.reset_index()
    master_table = row_pop.sort_values("row_pop", ascending=False)

    user_pop = df.groupby(by=target).agg({agg_col: "nunique"})
    user_pop.columns = ["user_pop"]
    user_pop = user_pop.reset_index()

    master_table = pd.DataFrame.merge(master_table, user_pop, on=target,
                                      how="left")
    return master_table


def popularity_cat(df, target, thres_col):
    """
    This function generates the categorical columns for the artist's
    popularity features by first generating categories based on certain
    thresholds and then pivoting them into new columns.

    :param df: pd.DataFrame | DataFrame containing the columns
                              "target" and "column"
    :param target: str | The name of the identifier columns
    :param thres_col: str | The name of the column where the threshold
                         is going to be applied
    :return df: pd.DataFrame | DataFrame with the artist's identifier
                               and the category columns
    """
    if target not in df or thres_col not in df:
        raise IOError("One of the column names entered is not contained"
                      "in the dataframe.")

    df["artist"] = "unpop"
    # pop artists
    df.loc[df[thres_col] > 15, "artist"] = "known"
    # know artists
    df.loc[df[thres_col] > 157, "artist"] = "pop"

    df = df[[target, "artist"]]
    df = pd.get_dummies(df)

    return df


def art_hits(df, target, filt_col, threshold=0.75):
    """
    This function calculates the number of  songs that generate a certain
    percentage of the total number of songs that the artist was played upon.

    :param df: pd.DataFrame |  Deezer's database in the DataFrame format
    :param target: str | the name of the identifier columns
    :param filt_col: str | the name of the column where
                           filtering is going to take place
    :param threshold: double | is the percentage of interest
    :return result: pd.DataFrame |  a DataFrame which contains
                                    the artist's id and the no. of songs that
                                    pass the threshold
    """
    hits = df.groupby(by=[target, filt_col]).agg({filt_col: "count"})
    hits.columns = ["song_count"]
    hits = hits.reset_index()
    hits = hits.sort_values([target, "song_count"], ascending=[1, 0])

    total = hits.groupby(by=target).agg({"song_count": "sum"})
    total.columns = ["tot_count"]
    total = total.reset_index()

    hits = hits.merge(total, on=target, how="left")
    # TODO: issue when python 2.x divide two integers, but for python 3.x OK
    hits["per_tot"] = hits["song_count"] / hits["tot_count"]

    n = len(hits)
    id_pos = hits.columns.get_loc(target)
    hits["cul_tot"] = hits["per_tot"]
    acc_pos = hits.columns.get_loc("cul_tot")
    per_pos = hits.columns.get_loc("per_tot")
    hits["flag"] = 1
    flag_pos = hits.columns.get_loc("flag")
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # TODO: as in previous functions, get rid of .iloc
    # Need help here, no idea how to do this without .iloc... open for suggestions
    for i in range(1, n, 1):
        if hits.iloc[i, id_pos] == hits.iloc[i - 1, id_pos]:
            hits.iloc[i, acc_pos] = hits.iloc[i - 1, acc_pos] \
                                    + hits.iloc[i, per_pos]

    for j in range(n - 1):
        if hits.iloc[j, id_pos] == hits.iloc[j - 1, id_pos]:
            if (hits.iloc[j, acc_pos] > threshold) & (hits.iloc[j, acc_pos] < 1):
                hits.iloc[j + 1, flag_pos] = 0

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    result = hits.groupby(by=target).agg({"flag": "sum"})
    result.columns = ["no_songs"]
    result = result.reset_index()

    return result


def hits_cat(df, target):
    """
    This function generates the categorical columns for the artist's "hits" features
    by first generating categories based on certain thresholds and then pivoting
    them into new columns
    :param df: pd.DataFrame | Dreezer's database in the DataFrame format
    :param target: str |  the name of the identifier columns
    :return df: pd.DataFrame | a DataFrame with the artist's identifier
                               and the hits category columns
    """
    if target not in df:
        raise IOError("One of the column names entered is not contained"
                      "in the dataframe.")
    df["hits"] = "one"
    df.loc[df["no_songs"] > 3, "hits"] = "medium"
    df.loc[df["no_songs"] > 15, "hits"] = "many"

    df = df[[target, "hits"]]
    df = pd.get_dummies(df)

    return df


if __name__ == "__main__":
    df = pd.read_csv("train.csv", nrow=1000)
    pop_df = artist_pop(df, "artist_id", "user_id")
    pop_cat_df = popularity_cat(df, "artist_id", "row_pop")
    hits_df = art_hits(df, "artist_id", "media_id")
    hits_cat_df = hits_cat(df, "artist_id")
