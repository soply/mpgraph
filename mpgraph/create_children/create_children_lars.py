# coding: utf8
""" Implementation of find_children method with the LAR regression algorithm.
    LAR regression equals the Lasso-path algorithm with the exception that we do
    not allow entries to drop out once they have been joined.
    Main usage should be conducted through 'create_children_lars', the other
    methods are auxiliary methods."""

__author__ = "Timo Klock"

import numpy as np
from scipy import optimize

from lasso_path_utils import calc_hit_cand, calc_hit_cand_selection
from ..mp_utils import calc_B_y_beta, calc_B_y_beta_selection

def create_children_lars(tiling_element, beta_min, beta_max):
    """ Interface method to search children in an interval (beta_min, beta_max)
    through the LAR regression algorithm.
    Calls main method 'divideAndConquer' to find the boundaries, added
    indices and corresponding signs first. Afterwards, post-processes the
    children to put them them in a more convenient form to process them in
    tilingElement.py.

    Parameters
    --------------
    tiling_element : object of class tilingElement
        Tiling element whose children we search for

    beta_min : double
        Lower bound of the interval in which we search children.

    beta_max : double (must be larger than beta_min)
        Uppoer bound of the interval in which we search children.

    Returns
    ----------------
    Returns a list with tuples where each tuple contains key information for a
    new tiling element. The tuples contain entries (minimal_parameter,
    maximal_parameters, new_support, new_signum) as specified above.
    """
    additional_indices, boundary_parameters, used_signs = \
                divideAndConquer(tiling_element.support,
                                         tiling_element.sign_pattern,
                                         beta_min,
                                         beta_max,
                                         tiling_element.options["env_minimiser"],
                                         tiling_element.svdAAt_U,
                                         tiling_element.svdAAt_S,
                                         tiling_element.A,
                                         tiling_element.y)
    children = lars_post_process_children(additional_indices,
                                          boundary_parameters,
                                          used_signs,
                                          tiling_element.support,
                                          tiling_element.sign_pattern)
    children.sort(key=lambda x: x[0][1]) # Sort by maximal beta parameter
    return children

def divideAndConquer(support, signum, beta_min, beta_max,
                     minimiser, svdAAt_U, svdAAt_S, A, y,
                     additional_indices=None,
                     used_signs=None,
                     boundary_parameters=None,
                     hit_candidates_min=None,
                     used_signs_min=None,
                     order_min=None,
                     hit_candidates_max=None,
                     used_signs_max=None,
                     order_max=None,
                     neglect_entries=None):
    """ Method that conducts the hard work to find the children of a tile that
    is represented by the first input parameters. Recursive method that calles
    itself over and over again in a divide-and-conquer manner. This explains
    all the 'None' arguments.

    Algorithm
    -------------
    The children of a tiling element in a specific range are defined by the
    indices that form the maximizing curve of a family of functions. These
    functions are so-called hit_candidates (ie. reg. parameters of the Lasso
    functional where the correlation of a specific index will be sufficiently
    large) With this method we search all segments in which a constant index
    and its respective hit_candidate maximizes this family. More information
    on these candidate functions can be found in [1]. Here we will just cover
    the algorithmic part.

    The algorithm follows a divide-and-conquer approach as follows:
        1) Calculate potential next regularisation parameters for beta_min and
           beta_max and order them increasingly.
        2) If at both points the maximizing index is the same, we assume that
           the respective index is maximizing the family in the whole interval.
        3) Elif there is a switch between two index, ie. the second largest
           index at beta_min is the largest at beta_max and vice versa. In this
           case we search a crossing between the respective curves. At this
           crossing we again calculate the potential reg. parameters and check
           if they two indices are still maximizing the family. If this is the
           case, we conclude that we found (beta_min, b_cross),
           (b_cross, beta_max) as two segments where the respective indices
           maximize the family.
        4) Else, it seems that there are multiple indices that exchange the
           maximizing role. We call the same method on subintervals
           (beta_min, 1/2 (beta_min + beta_max)) and
           (1/2 (beta_min + beta_max), beta_max). As inputs we give the old, and
           already calculated data, as well as the list in which new indices
           and signs have to be stored into.

    Parameters
    -------------
    support : array, shape (n_current_support)
        Contains the indices to the current active support, e.g.
        support = np.array([5, 10, 11]).

    signum : array, shape (n_current_support)
        Contains the signs of the coefficients that are currently active, e.g.
        signum = np.array([1, -1, 1]).

    beta_min : double
        Lower bound of the interval in which we search children.

    beta_max : double (must be larger than beta_min)
        Uppoer bound of the interval in which we search children.

    svdAAt_U : array, shape (n_measurements, n_measurements)
        Matrix U of the singular value decomposition of A*A^T.

    svdAAt_S : array, shape (n_measurements)
        Array S with singular values of singular value decomposition of A*A^T.

    A : array, shape (n_measurements, n_features)
        Sensing matrix of the problem.

    y : array, shape (n_measurements)
        The vector of measurements.

    Parameters used for recursiveness
    ----------------------------------
    additional_indices : list, stores newly added entries (order important)!

    used_signs : list, stores signs to newly added entries (order important)!

    boundary_parameters : list, stores the parameter where crossings are found,
        since these are exactly the beta's where children get replaced by other's.

    hit_candidates_min : Calculated 'hit_candidates' at beta_min

    used_signs_min : Signs that have been used in the 'hit_candidate'
        calculation. We pass this since it yields the sign's of potential newly
        added entries.

    order_min : Order of hit_candidates_min.

    hit_candidates_max/used_signs_max/order_max : Same as above but at beta_max.

    neglect_entries : If we have found suspicious entries to neglect, these are
         passed to further function calls to know about them.

    Returns
    ----------------
    additional_indices : Python list with additional indices (n_children)
    used_signs : Python list with signs corresponding to new indices (n_children)
    boundary_parameters : Python list with beta's at which the maximizing curve
        has been replaced, ie. beta's that belong to the boundaries of new
        children.

    Remarks
    -------------
    The proposed/explained algorithm works only under the assumption that each
    index contributes to maximizing the family only once in a consecutive
    interval. If this is not satisfied, it might come to problems, although the
    algorithm has some inherent robustness against this because we only evaluate
    endpoints of an interval. Moreover, the instantiated 'neglect_entries' list
    is used to neglect questionable indices with discontinous reg. parameter
    functions. Although this is seen very rarely, it can happen if the
    assumption does not hold.
    """

    def get_all_hit_cand(beta):
        B_beta, y_beta = calc_B_y_beta(A, y, svdAAt_U, svdAAt_S, beta)
        return calc_hit_cand(B_beta, y_beta, support, signum)

    def candidate_difference(beta, index1, index2):
        B_betaJ, B_betaI, y_beta = calc_B_y_beta_selection(A, y,
                                                           svdAAt_U, svdAAt_S,
                                                           beta, I=support,
                                                           J=[index1, index2])
        hit_candidates, used_signs = calc_hit_cand_selection(B_betaI, B_betaJ,
                                                             y_beta, signum)
        return (hit_candidates[0] - hit_candidates[1])

    def candidate_difference_squared(beta, index1, index2):
        B_betaJ, B_betaI, y_beta = calc_B_y_beta_selection(A, y,
                                                           svdAAt_U, svdAAt_S,
                                                           beta, I=support,
                                                           J=[index1, index2])
        hit_candidates, used_signs = calc_hit_cand_selection(B_betaI, B_betaJ,
                                                             y_beta, signum)
        return (hit_candidates[0] - hit_candidates[1]) ** 2
    # Initialisation
    if additional_indices is None:
        hit_candidates_max, used_signs_max = get_all_hit_cand(beta_max)
        hit_candidates_min, used_signs_min = get_all_hit_cand(beta_min)
        # If our heuristical assumption does not hold, it may be that the KKT
        # first KKT condition is not satisfied. In this case it is more robust
        # to additionally make this query and neglect entries smaller zero.
        # However, this may only happens if our assumption is violated.
        neglect_entries = np.where(np.logical_or(hit_candidates_min < 0,
                                         hit_candidates_max < 0))[0]
        hit_candidates_min[neglect_entries] = -1.0
        hit_candidates_max[neglect_entries] = -1.0
        order_max = np.argsort(hit_candidates_max)
        order_min = np.argsort(hit_candidates_min)
        additional_indices = [order_max[-1]]
        used_signs = [used_signs_max[order_max[-1]]]
        boundary_parameters = []
        boundary_parameters.append((hit_candidates_max[order_max[-1]], beta_max))
        boundary_parameters.append((hit_candidates_min[order_min[-1]], beta_min))

    if order_min[-1] == order_max[-1]:
        pass
    elif order_min[-1] == order_max[-2] and \
            order_min[-2] == order_max[-1]:
        # Search for crossing
        if minimiser == "scipy_bounded":
            res = optimize.minimize_scalar(candidate_difference_squared,
                                           args=(order_min[-1], order_max[-1]),
                                           bounds=(beta_min, beta_max),
                                           method="Bounded")
            x = res.x
            nfev = res.nfev
            fun = res.fun
            success = res.success
        elif minimiser == "scipy_brentq":
            fminfun = optimize.brentq
        elif minimiser == "scipy_brenth":
            fminfun = optimize.brenth
        elif minimiser == "scipy_ridder":
            fminfun = optimize.ridder
        elif minimiser == "scipy_bisect":
            fminfun = optimize.bisect
        if minimiser in ["scipy_brentq", "scipy_brenth",
                         "scipy_ridder", "scipy_bisect"]:
            beta_cross, res = fminfun(candidate_difference, beta_min,
                                      beta_max, xtol=1e-12,
                                      args=(order_max[-1], order_min[-1]),
                                      full_output=True)
            nfev = res.function_calls
            fun = 1e-12
            success = res.converged
        if success:
            hit_candidates_cross, used_signs_cross = get_all_hit_cand(beta_cross)
            hit_candidates_cross[neglect_entries] = -1.0
            order_cross = np.argsort(hit_candidates_cross)
            # Check if there is a third candidate larger than the two involved
            # candidates in the crossing
            if order_cross[-1] != order_min[-1] and \
                                            order_cross[-1] != order_max[-1]:
                # Case the crossing happened below another candidate. Make a
                # recursive call for both ends.
                divideAndConquer(support, signum, beta_cross, beta_max,
                                 minimiser, svdAAt_U, svdAAt_S, A, y,
                                 additional_indices=additional_indices,
                                 used_signs=used_signs,
                                 boundary_parameters=boundary_parameters,
                                 hit_candidates_min=hit_candidates_cross,
                                 used_signs_min=used_signs_cross,
                                 order_min=order_cross,
                                 hit_candidates_max=hit_candidates_max,
                                 used_signs_max=used_signs_max,
                                 order_max=order_max,
                                 neglect_entries=neglect_entries)
                divideAndConquer(support, signum, beta_min, beta_cross,
                                 minimiser, svdAAt_U, svdAAt_S, A, y,
                                 additional_indices=additional_indices,
                                 used_signs=used_signs,
                                 boundary_parameters=boundary_parameters,
                                 hit_candidates_min=hit_candidates_min,
                                 used_signs_min=used_signs_min,
                                 order_min=order_min,
                                 hit_candidates_max=hit_candidates_cross,
                                 used_signs_max=used_signs_cross,
                                 order_max=order_cross,
                                 neglect_entries=neglect_entries)
            else:
                additional_indices.append(order_min[-1])
                used_signs.append(used_signs_max[order_min[-1]])
                boundary_parameters.insert(-1,
                            (hit_candidates_cross[order_min[-1]], beta_cross))
        else:
            print """Warning: Could not find a crossing even if was suppposed to
                    find crossing at {0}, {1}, {2}, {3}""".format(
                                                    beta_cross, res, fun, nfev)
    else:
        # Bisection step
        beta_mid = 0.5 * (beta_min + beta_max)
        hit_candidates_mid, used_signs_mid = get_all_hit_cand(beta_mid)
        hit_candidates_mid[neglect_entries] = -1.0
        order_mid = np.argsort(hit_candidates_mid)
        if order_mid[-1] != order_max[-1]:
            divideAndConquer(support, signum, beta_mid, beta_max,
                             minimiser, svdAAt_U, svdAAt_S, A, y,
                             additional_indices=additional_indices,
                             used_signs=used_signs,
                             boundary_parameters=boundary_parameters,
                             hit_candidates_min=hit_candidates_mid,
                             used_signs_min=used_signs_mid,
                             order_min=order_mid,
                             hit_candidates_max=hit_candidates_max,
                             used_signs_max=used_signs_max,
                             order_max=order_max,
                             neglect_entries=neglect_entries)
        if order_mid[-1] != order_min[-1]:
            divideAndConquer(support, signum, beta_min, beta_mid,
                             minimiser, svdAAt_U, svdAAt_S, A, y,
                             additional_indices=additional_indices,
                             used_signs=used_signs,
                             boundary_parameters=boundary_parameters,
                             hit_candidates_min=hit_candidates_min,
                             used_signs_min=used_signs_min,
                             order_min=order_min,
                             hit_candidates_max=hit_candidates_mid,
                             used_signs_max=used_signs_mid,
                             order_max=order_mid,
                             neglect_entries=neglect_entries)
    return additional_indices, boundary_parameters, used_signs

def lars_post_process_children(additional_indices, boundary_parameters,
        used_signs, old_support, old_signum):
    """ Takes the results of 'divideAndConquer' method and translates them
    into a python list of tuples. Each tuple is related to a new tiling element
    and thus contains the following options:
    0 - minimal parameters: Tuple (alpha_min, beta_min) related to a parameter
                            pair where beta_min is the minimal beta for
                            which we can reach the respective support, as a
                            successor of the given 'old_support' and the
                            alpha_min is the related minimal alpha parameter for
                            which we obtain the respective support.
    1 - maximal parameters : Tuple (alpha_max, beta_max) related to a parameter
                            pair where beta_max is the maximal beta for which we
                            can reach the respective support, as a sucessor of
                            the given 'old_support', and the alpha_max is the
                            related minimal alpha parameter for which we obtain
                            the respective support.
    2 - new_support : The new support build from combining old_support with a
                      specific additional index.

    3 - new_signum : The new signum build from combining the old signum with a
                     specific additional sign.

    Parameters
    ---------------
    additional_indices: Python list of integers
        Each entry of the list is related to a new support by combining the old
        support with an entry of additional_indices.

    boundary_parameters : Python list of tuples (alpha, beta)
        Parameter pairs at which an additional index is replaced by another one,
        respectively at which we have a crossing in the related alpha-curves.

    used_signs : Python list of {-1.0, 1.0} entries.
        Contains an entry for each additional index to specify whether a
        related index/entry will form the new support as a positive entry or a
        negative entry. Thus, is used to form the new sign patterns related to
        the new supports.

    old_support : Python list of integer (n_old_support)
        Old support whose successors we search for.

    old_signum : Python list of {-1.0, 1.0} entries
        Sign pattern related to old support.

    Returns
    -----------------
    Returns a list with tuples where each tuple contains key information for a
    new tiling element. The tuples contain entries (minimal_parameter,
    maximal_parameters, new_support, new_signum) as specified above.
    """
    region_refinement = []
    for i, (index, sign) in enumerate(zip(additional_indices, used_signs)):
        new_support = np.append(old_support, index)
        new_signum = np.append(old_signum, sign)
        order = np.argsort(new_support)
        new_support, new_signum = new_support[order], new_signum[order]
        param_min = boundary_parameters[i+1]
        param_max = boundary_parameters[i]
        region_refinement.append([param_min, param_max, new_support,
            new_signum])
    return region_refinement
