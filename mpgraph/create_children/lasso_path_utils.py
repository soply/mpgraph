# coding: utf8
""" Implementation of utilities to calculate the Lasso-path knots/regularisation
    parameters. Different variants are possible, ie. variants to calculate all
    regularisation parameters, just a subset/selection, or only a specific
    type. """

__author__ = "Timo Klock"

import numpy as np

def calc_hit_cand_selection(AI, AJ, y, signum, prescribed_sign = None):
    """ This method calculates the 'hit candidates' for the Lasso path algorithm.
    The hit candidates correspond to a subgradient that hits the required
    boundary |s| < 1, i.e. for alpha > alpha_hit[j] subgradient to index j would
    exceed the boudary, and thus the index should be taken into the active set.

    Parameters
    ----------
    AI : array, shape (n_measurements, n_current_support)
        The sensing matrix A in the problem 1/2 || Ax - y ||_2^2 + alpha ||x||_1
        restricted to the index set given by support.

    AJ : array, shape (n_measurements, n_current_support)
        The sensing matrix A in 1/2 || Ax - y ||_2^2 + alpha ||x||_1
        restricted to the indices to which we want to calculate the hit value.

    y : array, shape (n_measurements)
        The vector of measurements

    signum : array, shape (n_current_support)
        Contains the signs of the coefficients that are currently active, e.g.
        signum = np.array([1., -1., 1.]).

    prescribed_sign : Integer from {-1, 1} or None if the sign from nominator
            shall be taken as the sign.
        Allows to use different signs in the calculations of the alpha_hit
        candidates. This becomes necessary in the special case of an entry
        dropping out of the support. From this new tile we essentially have two
        hitting boundaries, both of which belong to different signs. The
        'switch_sign' == False variant gives the upper boundary, ie. the
        boundary where the entry dropped out.

    Returns
    -----------
    Potential regularisation parameters for a selection of entries in the
    support, so-called 'hit candidates' (as a numpy array of size
    (n_entries_to_J)).
    """
    AtyJ = AJ.T.dot(y)
    if len(signum) == 0:
        hit_cand = np.abs(AtyJ)
        use_sign = np.sign(AtyJ)
    else:
        AjT_Ai_inverse_AtA = AJ.T.dot(AI).dot(\
            np.linalg.inv(AI.T.dot(AI)))
        hit_top = (AJ.T.dot(np.identity(y.shape[0])) - \
            AjT_Ai_inverse_AtA.dot(AI.T)).dot(y)
        aux_bot = AjT_Ai_inverse_AtA.dot(signum)
        use_sign = np.sign(hit_top)
        if prescribed_sign is not None:
            use_sign = prescribed_sign
        hit_bot = use_sign - aux_bot
        hit_cand = np.divide(hit_top, hit_bot)
    return hit_cand, use_sign

def calc_hit_cand(A, y, support, signum):
    """ This method calculates the 'hit candidates' for the Lasso path algorithm.
    The hit candidates correspond to a subgradient that hits the required
    boundary |s| < 1, i.e. for alpha > alpha_hit[j] subgradient to index j would
    exceed the boudary, and thus the index should be taken into the active set.

    Note that in we fill the entries that are related to support in this
    implementation with -1's in the return vector. Since the support is usually
    small compared to the number of features, this does not produce a large
    amount of overhead.

    Parameters
    ----------
    A : array, shape (n_measurements, n_features)
        The sensing matrix A in the problem 1/2 || Ax - y ||_2^2 + alpha ||x||_1.

    y : array, shape (n_measurements)
        The vector of measurements

    support : array, shape (n_current_support)
        Contains the indices to the current active support, e.g.
        support = np.array([5, 10, 11]).

    signum : array, shape (n_current_support)
        Contains the signs of the coefficients that are currently active, e.g.
        signum = np.array([1, -1, 1]).

    Returns
    -----------
    Potential regularisation parameters for all entries in the support,
    so-called 'hit candidates' (as a numpy array of size (n_features)).
    """
    full_index_set = np.arange(A.shape[1])
    J = np.setdiff1d(full_index_set, support)
    AJ = A[:, J]
    AI = A[:, support]
    hit_cand_J, use_sign_J = calc_hit_cand_selection(AI, AJ, y, signum)
    hit_cand = -1.0 * np.ones(A.shape[1])
    use_sign = np.zeros(A.shape[1])
    use_sign[J] = use_sign_J
    hit_cand[J] = hit_cand_J
    return hit_cand, use_sign

def calc_cross_cand_selection(AI, y, signum, indices):
    """ This method calculates the "cross candidates" for the Lasso path
    algorithm. Such cross candidates correspond to regularisation parameters
    at which an entry which was previously active in the support becomes zero
    again, hence the entry crosses/hits zero. Therefore such entries are kicked
    out of the active set when such a candidate is the next regularisation
    parameter in the grid.

    Parameters
    ----------
    AI : array, shape (n_measurements, n_support)
        The sensing matrix A restricted to the i'th columns.

    y : array, shape (n_measurements)
        The vector of measurements

    signum : array, shape (n_current_support)
        Contains the signs of the coefficients that are currently active, e.g.
        signum = np.array([1, -1, 1]).

    indices: array, shape (n_indices)
        Contains the indices that are of interested ie. to which we want to
        calculate the cross value.

    Returns
    -----------
    Potential crossing candidates for a subset of entries in the support. The
    crossings candidates are regularisation parameters/knots where a specific
    entry would hit 0 again.
    """
    if len(signum) == 0:
        cross_candidates = np.array([])
    else:
        inverseAtA = np.linalg.inv(AI.T.dot(AI))[indices,:]
        cross_top = inverseAtA.dot(AI.T).dot(y)
        cross_bot = inverseAtA.dot(signum)
        cross_candidates = np.divide(cross_top, cross_bot)
    return cross_candidates

def calc_cross_cand_bottom_selection(AI, signum, indices):
    """ This method calculates the denominator for "cross candidates" for the
    LP algorithm. Cross candidates correspond to regularisation parameters
    at which an entry which was previously active in the support becomes zero
    again, hence the entry crosses/hits zero. Therefore such entries are kicked
    out of the active set when such a candidate is the next regularisation
    parameter in the grid.

    Parameters
    ----------
    AI : array, shape (n_measurements, n_support)
        The sensing matrix A restricted to the i'th columns.

    signum : array, shape (n_current_support)
        Contains the signs of the coefficients that are currently active, e.g.
        signum = np.array([1, -1, 1]).

    indices: array, shape (n_indices)
        Contains the indices that are of interested ie. to which we want to
        calculate the cross value.

    Returns
    -----------
    Denominator of potential crossing candidates for a subset of entries in the
    support. The crossings candidates are regularisation parameters/knots where
    a specific entry would hit 0 again.
    """
    if len(signum) == 0:
        cross_bot = np.array([])
    else:
        inverseAtA = np.linalg.inv(AI.T.dot(AI))[indices,:]
        cross_bot = inverseAtA.dot(signum)
    return cross_bot

def calc_cross_cand(A, y, support, signum):
    """ This method calculates the "cross candidates" for the Lasso path
    algorithm. Such cross candidates correspond to regularisation parameters
    at which an entry which was previously active in the support becomes zero
    again, hence the entry crosses/hits zero. Therefore such entries are kicked
    out of the active set when such a candidate is the next regularisation
    parameter in the grid.

    Note that in we fill the entries that are related to support in this
    implementation with -1's in the return vector. Since the support is usually
    small compared to the number of features, this does not produce a large
    amount of overhead.

    Parameters
    ----------
    A : array, shape (n_measurements, n_features)
        The sensing matrix A in the problem 1/2 || Ax - y ||_2^2 + alpha ||x||_1.

    y : array, shape (n_measurements)
        The vector of measurements

    support : array, shape (n_current_support)
        Contains the indices to the current active support, e.g.
        support = np.array([5, 10, 11]).

    signum : array, shape (n_current_support)
        Contains the signs of the coefficients that are currently active, e.g.
        signum = np.array([1, -1, 1]).

    Returns
    -----------
    Potential crossing candidates for a all entries in the support. The
    crossings candidates are regularisation parameters/knots where a specific
    entry would hit 0 again.
    """
    AI = A[:,support]
    cross_cand_I = calc_cross_cand_selection(AI, y, signum,
                                             np.arange(AI.shape[1]))
    cross_candidates = -1.0 * np.ones(A.shape[1])
    cross_candidates[support] = cross_cand_I
    return cross_candidates

def calc_all_cand(A, y, support, signum):
    """ This method calculates both cross and ht candidates for the Lasso path
    algorithm and stores them in a single vector. Read function documentations
    to cross and hit candidate calculation to obtain further information about
    what a specific candidate means for the solution.

    Parameters
    ----------
    A : array, shape (n_measurements, n_features)
        The sensing matrix A in the problem 1/2 || Ax - y ||_2^2 + alpha ||x||_1.

    y : array, shape (n_measurements)
        The vector of measurements

    support : array, shape (n_current_support)
        Contains the indices to the current active support, e.g.
        support = np.array([5, 10, 11]).

    signum : array, shape (n_current_support)
        Contains the signs of the coefficients that are currently active, e.g.
        signum = np.array([1, -1, 1]).

    Returns
    -----------
    Calculates all candidates for all entries. If an entry is in the support,
    the respective crossing candidate is calculated (ie. the reg. parameter
    where the entry hits zero again). If an entry is not in the support, the
    respective hitting candidate (with sign given by nominator) is calculated.
    Returns value is numpy array of shape (n_features).
    """
    cross_candidates = calc_cross_cand(A, y, support, signum)
    candidates, used_sign = calc_hit_cand(A, y, support, signum)
    candidates[support] = cross_candidates[support]
    return candidates, used_sign
