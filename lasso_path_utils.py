#coding: utf8
import numpy as np

def calc_hit_cand_selection(AI, AJ, y, support, signum):
    """ This method calculates the 'hit_candidates' for the Lasso path algorithm.
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

    support : array, shape (n_current_support)
        Contains the indices to the current active support, e.g.
        support = np.array([5., 10., 11.]).

    signum : array, shape (n_current_support)
        Contains the signs of the coefficients that are currently active, e.g.
        signum = np.array([1., -1., 1.]).
    """
    AtyJ = AJ.T.dot(y)
    if len(support) == 0:
        hit_cand = np.abs(AtyJ)
        hit_signs = np.sign(AtyJ)
    else:
        AjT_Ai_inverse_AtA = AJ.T.dot(AI).dot(\
            np.linalg.inv(AI.T.dot(AI)))
        hit_top = (AJ.T.dot(np.identity(y.shape[0])) - \
            AjT_Ai_inverse_AtA.dot(AI.T)).dot(y)
        # Check if plus or minus gives the upper bound for alpha
        # FIXME: Note that Tibsharani paper can not transferred to this 1-by-1
        # FIXME 2: Rethink this...
        aux_bot = AjT_Ai_inverse_AtA.dot(signum)
        use_sign = np.sign(hit_top)
        # In special cases we have to override this choice! See below
        use_sign[aux_bot > 1.0] = -1.0 # In this case it is a negative value anyway...
        use_sign[aux_bot < (-1.0)] = 1.0
        hit_bot = use_sign - aux_bot
        hit_cand = np.divide(hit_top, hit_bot)
        hit_signs = np.sign(hit_cand)
    return hit_cand, hit_sign

def calc_hit_cand(A, y, support, signum):
    """ This method calculates the 'hit_candidates' for the Lasso path algorithm.
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
        support = np.array([5., 10., 11.]).

    signum : array, shape (n_current_support)
        Contains the signs of the coefficients that are currently active, e.g.
        signum = np.array([1., -1., 1.]).
    """
    full_index_set = np.arange(A.shape[1])
    J = np.setdiff1d(full_index_set, support)
    AJ = A[:, J]
    AI = A[:, support]
    hit_cand_J, hit_sign_J = calc_hit_cand_selection(AI, AJ, y, support, signum)
    hit_cand = -1.0 * np.ones(A.shape[1])
    hit_sign = np.zeros(A.shape[1])
    hit_sign[J] = hit_sign_J
    hit_cand[J] = hit_cand_J
    return hit_cand, hit_sign
