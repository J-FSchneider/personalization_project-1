# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Imports
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import pandas as pd
import numpy as np
from utils.df_trans import df_summ
from utils.df_trans import df_tot
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# User / time of day analysis
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def tod_pivot(data, audio_df, time_threshold=0.1):
    """
    This function analysis the song's that the users
    listen to throughout all the moments of day
    :param data: pd.DataFrame|  the pd.DataFrame outputted from Pipeline
    :param audio_df: pd.DataFrame | a pd.DataFrame containing Spotify's
                audio features
    :param time_threshold: float | the % of user activity that needs
                        to concentrate in that moment of day to be
                        called "relevant"
    :return: user_final: pd.DataFrame | a pd.DataFrame containing the
                    the % of songs that a user listens to during all the
                    moments of day as well as it's relevance in the
                    data set (calculated as activity)
    :return: final: pd.DataFrame | a pd.DataFrame that contains the same
                information as user_final but at a user / song level.
                It also contains the song audio features
    """
    # Name of the identifiers
    user = "user_id"
    song = "media_id"
    time = "moment_of_day"

    # Set new variables
    st = "st_count"
    sr = "song_rel"
    srp = "song_rel_per"
    t = "time_sum"
    tp = "time_per"
    tr = "time_rel"
    s = "song_sum"
    tot = "total"
    w = "weights"
    indi = "indicator"
    m = "mult"
    wi = "weight_ind"

    # Create granular user/song/time table
    master = df_summ(df=data,
                     index=[user, song, time],
                     rename=st,
                     target=song,
                     criteria="count")

    # Do the time table
    time_agg = df_summ(df=master,
                       index=[user, time],
                       rename=t,
                       target=st,
                       criteria="sum")
    time_agg = df_tot(df=time_agg,
                      index=user,
                      rename=tot,
                      target=t,
                      criteria="sum")
    time_agg[tp] = time_agg[t] / time_agg[tot]
    time_agg[tr] = \
        time_agg[tp].apply(lambda x: 1 if x > time_threshold else 0)
    time_rel = time_agg[[user, time, tr]]
    time_vars = time_agg[time].unique()  # List to undo pivot

    # Do the song table
    song_agg = df_summ(df=master,
                       index=[user, song],
                       rename=s,
                       target=st,
                       criteria="sum")
    song_agg = df_tot(df=song_agg,
                      index=user,
                      rename=tot,
                      target=s,
                      criteria="sum")
    song_agg[w] = song_agg[s] / song_agg[tot]
    user_song_rel = df_summ(df=song_agg,
                            index=user,
                            rename=sr,
                            target=s,
                            criteria="sum")
    user_song_rel[tot] = np.sum(user_song_rel[sr])
    user_song_rel[srp] = user_song_rel[sr] / user_song_rel[tot]
    user_song_rel = user_song_rel[[user, srp]]

    # Create user pivot
    user_pivot = pd.pivot_table(master,
                                values=st,
                                index=[user, song],
                                columns=time)
    user_pivot = user_pivot.fillna(0)
    user_pivot = user_pivot.reset_index()
    user_melt = pd.melt(user_pivot,
                        id_vars=[user, song],
                        value_vars=time_vars)
    user_melt = pd.merge(user_melt, time_rel,
                         on=[user, time],
                         how="left")
    user_melt[indi] = user_melt["value"][user_melt[tr] == 1]
    user_melt = user_melt.fillna(1)
    user_melt[indi] = \
        user_melt[indi].apply(lambda x: 1 if x > 0 else 0)
    user_melt = user_melt[[user, song, time, indi]]
    user_pivot = pd.pivot_table(user_melt,
                                values=indi,
                                index=[user, song],
                                columns=time)

    user_pivot[m] = 1
    for column in user_pivot.columns:
        user_pivot[m] = user_pivot[m] * user_pivot[column]

    user_pivot = user_pivot.reset_index()
    user_mult = user_pivot[[user, song, m]]

    # Do the final table aggregations
    final = pd.merge(song_agg, user_mult,
                     on=[user, song],
                     how="left")
    final[wi] = final[m] * final[w]
    final = pd.merge(final, audio_df,
                     on=song,
                     how="left")
    user_final = df_summ(df=final,
                         index=user,
                         rename="final",
                         target=wi,
                         criteria="sum")
    user_final = user_final.sort_values(by=user)
    user_final = pd.merge(user_final, user_song_rel,
                          on=user,
                          how="left")
    final = final.sort_values(by=[user, s], ascending=[1, 0])
    return user_final, final
# =========================================================================
