# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Import Packages
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import pandas as pd
# =========================================================================


def hit_rate(df):
    """
    This function constructs the (user, media) hit rate
    and adds it to the Data Frame provided
    :param df: pd.DataFrame | the raw data
    :return df: pd.DataFrame | the raw data plus the hit rate column
    """
    df_filtered = df[df["listen_type"] == 1]
    var_agg = {"is_listened": "sum", "listen_type": "count"}
    table = df_filtered.groupby(by=["user_id", "media_id"]).agg(var_agg)
    table.columns = ["acceptance", "exposure"]
    table = table.reset_index()
    table["hit_rate"] = table["acceptance"] / table["exposure"]
    table = table[["user_id", "media_id", "hit_rate", "exposure"]]
    df = pd.merge(df, table, on=["user_id", "media_id"], how="left")
    return df


def hr_evolution(df, user_range):
    """
    This function constructs a pd.DataFrame that summarizes the
    mean and median evolution of the hit rate and the exposure on
    flow as the number of users increases
    :param df: pd.DataFrame | the DataFrame where to compute
    :param user_range: list | a list of threshold were to filter
                            the users
    :return output_df: pd.DataFrame | the report of the evolution
                                    of the metrics
    """
    size = len(user_range)
    x = np.zeros((size, 5))
    for i in range(size):
        x[i, 0] = user_range[i]
        users = df[df["user_id"] <= user_range[i]]
        x[i, 1] = users["hit_rate"].mean()
        x[i, 2] = users["hit_rate"].median()
        x[i, 3] = users["exposure"].mean()
        x[i, 4] = users["exposure"].median()

    output_df = pd.DataFrame(x)
    output_df.columns = ["users",
                         "hr_mean",
                         "hr_med",
                         "exp_mean",
                         "exp_med"]
    per = output_df[["hr_mean", "exp_mean"]]
    per = per.pct_change()
    per.columns = ["hr_mean_pct", "exp_mean_pct"]
    output_df = output_df.merge(per, how="left",
                                left_index=True,
                                right_index=True)
    return output_df


def song_rank(df):
    """
    This function add the ranking of the songs to the
    original data. The ranking is based on teh number of
    times that the song was listened to
    :param df: pd.DataFrame | the data
    :return df: pd.DataFrame | the data plus a column that
                                contains the song rank
    """
    target = "media_id"
    criteria = "count"
    var_agg = {target: criteria}
    songs = df.groupby(by=target).agg(var_agg)
    songs.columns = [criteria]
    songs = songs.reset_index()
    songs = songs.sort_values(by=criteria, ascending=False)
    # songs["total"] = np.sum(songs[criteria])
    # songs["cum"] = np.cumsum(songs[criteria])
    # songs["cum_per"] = songs["cum"] / songs["total"]
    songs["song_rank"] = songs[criteria].rank(ascending=False)
    songs = songs[[target, "song_rank"]]
    df = pd.merge(df, songs, on=target, how="left")
    return df


def relevant_elements(df, target, threshold=0.8, criteria="count"):
    """
    This function generates the list of relevant elements
    (that meet the threshold) from the target column
    in the DataFrame provided
    :param df: pd.DataFrame | the DataFrame where data is
                                extracted
    :param target: string | name of the column where to
                            aggregate and threshold
    :param threshold: float | the threshold parameter
    :param criteria: string | the aggregation method
    :return output_list: list | the list of elemnts from
                                the target
    """
    # Set variable names
    total = criteria + "_tot"
    culm = criteria + "_cul"
    per = criteria + "_per"
    # Construct aggregation
    var_agg = {target: criteria}
    tmp = df.groupby(by=target).agg(var_agg)
    tmp.columns = [criteria]
    tmp = tmp.reset_index()
    tmp = tmp.sort_values(criteria, ascending=False)
    tmp[total] = np.sum(tmp[criteria])
    tmp[culm] = np.cumsum(tmp[criteria])
    tmp[per] = tmp[culm] / tmp[total]
    # Threshold the data
    output_df = tmp[tmp[per] < threshold]
    # Generate output list
    output_list = output_df[target].tolist()
    return output_list


def plot_relevant_elements(df,
                           user_range,
                           target,
                           threshold=0.8,
                           criteria="count"):
    """
    This function creates a pd.DataFrame that shows the evolution
    of the number of songs or artists that are being added
    into the data set while the number of users is growing
    :param df: pd.DataFrame | the DataFrame where to extract the data
    :param user_range: list | the list of user ranges to plot
    :param target: string | the name of the column where to make the
                            analysis
    :param threshold: float | the threshold the notion of relevance
    :param criteria: string | the additive criteria for the analysis
    :return output_df: pd.DataFrame | contains the evolution for
                                        each element in the user
                                        range
    """
    n = len(user_range)
    x = np.zeros((n, 5))
    list_old = []
    for i in range(n):
        list_new = list_old
        x[i, 0] = user_range[i]
        df_filtered = df[df["user_id"] <= user_range[i]]
        list_old = relevant_elements(df_filtered,
                                     target,
                                     threshold,
                                     criteria)
        x[i, 1] = len(list_old)
        a = set(list_new)
        b = set(list_old)
        c = b.difference(a)
        d = a.difference(b)
        # if len(c) == 0:
        #     c = a.difference(b)
        diff = np.max([c, d])

        inter = set.intersection(a, b)
        x[i, 2] = len(diff)
        x[i, 3] = len(inter)
        x[i, 4] = x[i, 3] / x[i, 1]

    output_df = pd.DataFrame(x)
    output_df.columns = ["users",
                         "no_"+target[0:3],
                         "diff_"+target[0:3],
                         "inter_"+target[0:3],
                         "inter_per_" + target[0:3]]
    return output_df


def prep_test(test):
    test = test[["user_id", "media_id"]]
    test.columns = ["userID", "itemID"]
    df = test.copy()
    df["ratings_org"] = 1
    df = df.sort_values(by="userID")
    return df


def prep_for_model(data_users):
    df = data_users[["user_id", "media_id", "hit_rate"]]
    df = df.dropna()
    df = df.groupby(by=["user_id", "media_id"])["hit_rate"].mean()
    df = df.reset_index()
    df.columns = ["userID", "itemID", "ratings"]
    return df

# =========================================================================
