import numpy as np

"""
This module contains the similarity measures one might use to
compute the similarity matrix in the neighborhood-based models
(user-based, item-based or hybrid).
"""


def ochiai(array_i, array_j):
    """
    Cosine similarity for binary and unary inputs
    :param array_i: np.array | array containing binary input
    :param array_j: np.array | array containing binary input
    :return: float | Ochiai similarity between the input arrays
    """
    supp_i_inter_j = np.nansum(array_i * array_j)
    supp_i, supp_j = np.nansum(array_i), np.nansum(array_j)
    return supp_i_inter_j / np.sqrt(supp_i * supp_j)


def cosine(array_i, array_j):
    # TODO: implement
    pass
