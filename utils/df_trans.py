"""
    This script contains functions that make specific
    transformations to pd.DataFrames
"""
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Imports
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import pandas as pd
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Summarize by given index
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def df_summ(df, index, rename, target, criteria="sum"):
    """
    This function summarizes a pd.DataFrame based on the index and
    target column provided. The aggregation is done by the criteria
    and the summarized pd.DataFrame is returned
    :param df: pd.DataFrame | the pd.DataFrame to summarize
    :param index: list | columns for the groupby
    :param rename: str | name of the new column
    :param target: str | column to based the aggregation on
    :param criteria: str | criteria for value aggregation
    :return table: pd.DataFrame | the pd.DataFrame summarized
    """
    var_agg = {target: criteria}
    table = df.groupby(by=index).agg(var_agg)
    table = table.rename(columns={target: rename})
    table = table.reset_index()
    return table
# =========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Column totals function
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def df_tot(df, index, rename, target, criteria="count"):
    """
    This function generates the aggregation of a pd.DataFrame
    based on the list of indexes, the target column and
    the aggregation criteria
    :param df: pd.DataFrame | original pd.DataFrame
    :param index: list | a list of columns for group_by
    :param rename: str | name fo the new column
    :param target: str | the name of the column for the aggregation
    :param criteria: str | criteria for values aggregation
    :return result: pd.DataFrame | the original pd.DataFrame with
                and an extra column at the end with the totals
    """
    var_agg = {target: criteria}
    table = df.groupby(by=index).agg(var_agg)
    table = table.rename(columns={target: rename})
    table = table.reset_index()
    result = pd.merge(df, table, on=index, how="left")
    result = result.sort_values(rename, ascending=False)
    return result
# =========================================================================
