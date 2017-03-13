# coding: utf8
""" Implementation of find_children method with the Lasso-path algorithm.
    The Lasso-path algorithm successively computes regularisation parameter at
    which either a new entry joins the support (if it's correlation equals the
    correlation of all entries in the support) or an entry crosses zero and drops
    from the support. """

__author__ = "Timo Klock"

import bisect

import numpy as np
from scipy import optimize

from lasso_path_utils import (calc_all_cand, calc_cross_cand,
                              calc_cross_cand_bottom_selection,
                              calc_cross_cand_selection, calc_hit_cand,
                              calc_hit_cand_selection)

from ..mp_utils import calc_B_y_beta, calc_B_y_beta_selection

# Numeric tolerance constance
# We consider a reg parameter candidate to be larger than the previously, selected
# candidate if its larger than the (previous_value - __TOL_CROSSING_NEGLECTION__).
__TOL_CROSSING_NEGLECTION__ = 1e-4
# Instead of evaluating functions for reg parameter candidates at the nodes directly, we
# build a weighted sum of beta_min, beta_max with __DERIVATION_FROM_NODE_BETAS__ as the
# weight.
__DERIVATION_FROM_NODE_BETAS__ = 1e-10

def create_children_lasso(tiling_element, beta_min, beta_max):
    """ Interface method to search children in an interval (beta_min, beta_max)
    through the Lasso-path algorithm.
    Calls main method 'divideAndConquer' to find the boundaries on a refinement of the
    interval (beta_min, beta_max), where the refinement is such that each sub-interval has
    only a single parent, and on each sub-interval the candidate functions are continuous.
    Adds indices and corresponding signs first. Afterwards, post-processes the
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
    # Extract data from tiling_element
    A, y, svdAAt_U, svdAAt_S, support, sign_pattern = tiling_element.A, \
        tiling_element.y, tiling_element.svdAAt_U, tiling_element.svdAAt_S, \
        tiling_element.support, tiling_element.sign_pattern
    children = []
    # Loop through different parents to refine (beta_min, beta_max) for the
    # first time.
    for parent in tiling_element.parents:
        # Check if there is actual overlap between parent range and given range
        if (parent[1] - beta_max < -1e-14 and beta_min - parent[2] < -1e-14):
            # Get entry that last changed
            last_entry_changed = np.setxor1d(tiling_element.support,
                                             parent[0].support)
            assert len(last_entry_changed) == 1
            last_entry_changed = last_entry_changed[0]
            # Get sign to the last entry that changed
            if last_entry_changed in tiling_element.support:
                sign_last_entry_changed = tiling_element.sign_pattern[
                                tiling_element.support == last_entry_changed]
            else: # last_entry_changed in parent's support
                sign_last_entry_changed = parent[0].sign_pattern[
                                parent[0].support == last_entry_changed]
            # Get inner beta range due to parents
            beta_min_inner = np.maximum(beta_min, parent[1])
            beta_max_inner = np.minimum(beta_max, parent[2])
            # Check for discontinuities
            discontinuities = find_crossing_discontinuities(A, y, svdAAt_U,
                                        svdAAt_S, support, sign_pattern,
                                        beta_min_inner, beta_max_inner)
            discontinuities.extend([beta_min_inner, beta_max_inner])
            discontinuities = sorted(discontinuities)
            # Loop through intervals without discontinuities
            inner_children = []
            beta_l = discontinuities.pop(0)
            while len(discontinuities) > 0:
                beta_r = discontinuities.pop(0)
                additional_indices, boundary_parameters, used_signs = \
                    divideAndConquer(support, sign_pattern,
                                     beta_l, beta_r,
                                     tiling_element.options["env_minimiser"],
                                     svdAAt_U, svdAAt_S, A, y,
                                     last_entry_changed,
                                     sign_last_entry_changed, 1)
                inner_children.extend(lasso_post_process_children(
                                      additional_indices, boundary_parameters,
                                      used_signs, support, sign_pattern))
                beta_l = beta_r
            children.extend(lasso_children_merge(inner_children))
    children.sort(key=lambda x: x[0][1])
    children = lasso_children_merge(children)
    return children

def divideAndConquer(support, signum, beta_min, beta_max,
                     minimiser, svdAAt_U, svdAAt_S, A, y,
                     last_entry_changed,
                     sign_last_entry_changed,
                     recursion_level,
                     additional_indices=None,
                     used_signs=None,
                     boundary_parameters=None,
                     candidates_min=None,
                     used_signs_min=None,
                     order_min=None,
                     candidates_max=None,
                     used_signs_max=None,
                     order_max=None,
                     neglect_entries=None,
                     spotted_discontinuities=None):
    """ Method that conducts the hard work to find the children of a tile that
    is represented by the first input parameters. Recursive method that calls
    itself over and over again in a divide-and-conquer manner. This explains
    all the 'None' arguments (used in recurve calls).

    Algorithm
    -------------
    The children of a tiling element in a specific range are defined by the
    indices that form the maximizing curve of a family of functions. These
    functions are so-called candidates (ie. penalty parameters of the Lasso
    functional where either the correlation of a specific index will be
    sufficiently large; or entries that are in the support changed slope and are
    crossing zero). With this method we search all segments in which a constant index
    and its respective candidate maximizes this family. More information
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

    last_entry_changed : integer
        The index to the entry that changed from the parent tile to the given
        tiling element. Note that there is only one specific parent since we do a
        refinement of the range do achieve this beforehand
        (in create_children_lasso).

    sign_last_entry_changed : {-1, 1}
        Sign of the 'last_entry_changed' in either the support of the given
        tiling element (if it was added before) or the distinct parent tile.

    recursion_level : int
        Recursion depth (should be starting from 1 in the initial call).

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

    spotted_discontinuities : list of integers
        It may happen that the assumptions that make this algorithm feasible
        (see 'Remarks') do not hold. In this case, we may spot a discontinuity
        penalty parameter, or candidate, function during the run of this algorithm.
        If this is the case, we will add it to this list and rerun the whole
        algorithm but neglecting this entry (adding it to neglect_entries).


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
    def get_substitute_for_last_entry_changed(beta):
        B_betaJ, B_betaI, y_beta = calc_B_y_beta_selection(A, y,
                                svdAAt_U, svdAAt_S, beta, I=support,
                                J=np.array([last_entry_changed]).astype('int'))
        return calc_hit_cand_selection(B_betaI, B_betaJ, y_beta, signum,
                                    prescribed_sign = sign_last_entry_changed)

    def get_all_cand(beta):
        B_beta, y_beta = calc_B_y_beta(A, y, svdAAt_U, svdAAt_S, beta)
        J = np.setdiff1d(np.arange(A.shape[1]), support)
        return calc_all_cand(B_beta, y_beta, support, signum)

    def candidate_difference(beta, index1, index2):
        I_in_support = np.where(np.in1d(support, np.array([index1, index2])))[0]
        J = np.setdiff1d([index1, index2], support[I_in_support])
        B_betaJ, B_betaI, y_beta = calc_B_y_beta_selection(A, y,
                                                           svdAAt_U, svdAAt_S,
                                                           beta, I=support,
                                                           J=J)
        if index1 in support and index2 in support:
            candidates = calc_cross_cand_selection(B_betaI, y_beta, signum,
                                                    indices=I_in_support)
        elif index1 in support:
            cross_cand = calc_cross_cand_selection(B_betaI, y_beta, signum,
                                                    indices=I_in_support)
            hit_cand, used_signs= calc_hit_cand_selection(B_betaI, B_betaJ,
                                                          y_beta, signum)
            candidates = [cross_cand, hit_cand]
        elif index2 in support:
            cross_cand = calc_cross_cand_selection(B_betaI, y_beta, signum,
                                                    indices=I_in_support)
            hit_cand, used_signs = calc_hit_cand_selection(B_betaI, B_betaJ,
                                                           y_beta, signum)
            candidates = [cross_cand, hit_cand]
        else:
            candidates, used_signs = calc_hit_cand_selection(B_betaI, B_betaJ,
                                                             y_beta, signum)
        return (candidates[0] - candidates[1])

    def candidate_difference_squared(beta, index1, index2):
        I_in_support = np.where(np.in1d(support, np.array([index1, index2])))[0]
        J = np.setdiff1d([index1, index2], support[I_in_support])
        B_betaJ, B_betaI, y_beta = calc_B_y_beta_selection(A, y,
                                                           svdAAt_U, svdAAt_S,
                                                           beta, I=support,
                                                           J=J)
        if index1 in support and index2 in support:
            candidates = calc_cross_cand_selection(B_betaI, y_beta, signum,
                                                    indices=I_in_support)
        elif index1 in support:
            cross_cand = calc_cross_cand_selection(B_betaI, y_beta, signum,
                                                    indices=I_in_support)
            hit_cand, used_signs= calc_hit_cand_selection(B_betaI, B_betaJ,
                                                          y_beta, signum)
            candidates = [cross_cand, hit_cand]
        elif index2 in support:
            cross_cand = calc_cross_cand_selection(B_betaI, y_beta, signum,
                                                    indices=I_in_support)
            hit_cand, used_signs = calc_hit_cand_selection(B_betaI, B_betaJ,
                                                           y_beta, signum)
            candidates = [cross_cand, hit_cand]
        else:
            candidates, used_signs = calc_hit_cand_selection(B_betaI, B_betaJ,
                                                             y_beta, signum)
        return (candidates[0] - candidates[1]) ** 2
    beta_eval_max = (1.0 - __DERIVATION_FROM_NODE_BETAS__) * beta_max + \
                           __DERIVATION_FROM_NODE_BETAS__ * beta_min
    beta_eval_min = (1.0 - __DERIVATION_FROM_NODE_BETAS__) * beta_min + \
                           __DERIVATION_FROM_NODE_BETAS__ * beta_max
    # Initialisation
    if recursion_level == 1:
        candidates_max, used_signs_max = get_all_cand(beta_eval_max)
        candidates_min, used_signs_min = get_all_cand(beta_eval_min)
        beta_mid = 0.5 * (beta_min + beta_max)
        candidates_mid, used_signs_mid = get_all_cand(beta_mid)
        # Filter out all candidates that are above the current curve
        boun_at_max = candidates_max[last_entry_changed]
        boun_at_min = candidates_min[last_entry_changed]
        boun_at_mid = candidates_mid[last_entry_changed]
        # If last_entry changed dropped out of the support, the next joining
        # time for last_entry_changed has to be calculated separately since
        # our calculation gives us the alpha_cross of the tiling before, ie. the
        # upper boundary. If an index directly immediately joins the support
        # again with opposite sign, we need to manually calculate this entries.
        if last_entry_changed not in support:
            # Calculate hit candidate to the opposite sign
            subst_candidate_min, subst_sign_min = \
                            get_substitute_for_last_entry_changed(beta_eval_min)
            subst_candidate_max, subst_sign_max = \
                            get_substitute_for_last_entry_changed(beta_eval_max)
            subst_candidate_mid, subst_sign_mid = \
                            get_substitute_for_last_entry_changed(beta_mid)
            if used_signs_min[last_entry_changed] == sign_last_entry_changed and \
                    used_signs_max[last_entry_changed] == sign_last_entry_changed:
                # Case the sign used was the correct sign to calculate the upper
                # boundary, hence we have to substitute the entries in the
                # candidate vector.
                used_signs_min[last_entry_changed] = subst_sign_min[0]
                used_signs_max[last_entry_changed] = subst_sign_max[0]
                used_signs_mid[last_entry_changed] = subst_sign_mid[0]
                candidates_min[last_entry_changed] = subst_candidate_min[0]
                candidates_max[last_entry_changed] = subst_candidate_max[0]
                candidates_mid[last_entry_changed] = subst_candidate_mid[0]
            elif used_signs_min[last_entry_changed] != sign_last_entry_changed and \
                    used_signs_max[last_entry_changed] != sign_last_entry_changed:
                # Case boundaries that are calculated are the actual entry
                # indices, hence we have to calculate the real boundaries.
                boun_at_max = subst_candidate_max[0]
                boun_at_min = subst_candidate_min[0]
                boun_at_mid = subst_candidate_mid[0]
            else:
                raise RuntimeError("Lasso-creation: Signs of an entry for the" + \
                " hit candidate change: {0}, {1}, {2}".format(
                used_signs_min[last_entry_changed],
                used_signs_max[last_entry_changed],
                sign_last_entry_changed))
        # Build neglected entries:
        # 1) Neglect all entries that are either at the beginning, the end or
        #    in the middle (three points for stability reasons) above the
        #    function to the last parameter up to a certain tolerance.
        neglect_entries_in_support_1 = np.where(np.logical_or(np.logical_or(
                        (candidates_min[support] - boun_at_min)/boun_at_min >
                                    -__TOL_CROSSING_NEGLECTION__,
                        (candidates_max[support] - boun_at_max)/boun_at_max >
                                    -__TOL_CROSSING_NEGLECTION__),
                        (candidates_mid[support] - boun_at_mid)/boun_at_mid >
                                    -__TOL_CROSSING_NEGLECTION__))[0]
        neglect_entries = support[neglect_entries_in_support_1]
        # 2) Add additionally spotted discontinuities if there exist some
        if spotted_discontinuities is None:
            spotted_discontinuities = []
        elif len(spotted_discontinuities) > 0:
            neglect_entries = np.append(neglect_entries,
                                        spotted_discontinuities)
        candidates_min[neglect_entries] = -1.0
        candidates_max[neglect_entries] = -1.0
        order_max = np.argsort(candidates_max)
        order_min = np.argsort(candidates_min)
        additional_indices = [order_max[-1]]
        used_signs = [used_signs_max[order_max[-1]]]
        boundary_parameters = []
        boundary_parameters.append((candidates_max[order_max[-1]], beta_max))
        boundary_parameters.append((candidates_min[order_min[-1]], beta_min))

    # In cases the 2 Postulates do not hold (ie. there are two occasions where)
    # the shrinkage term is going through zero, or there are unconnected
    # intervals where the maximum is formed by a certain entry, we can run into
    # trouble with our approach. Since the interval (beta_min, beta_max) will
    # converge to resulting discontinuities, we can make our approach robust to
    # such cases by checking whether the smallest index at min/largest index at
    # beta_min is the largest/smallest index at beta_max. This indicates a
    # disconitnuity and we will add it then to the neglected entries
    if recursion_level > 1 and order_min[-1] == order_max[0]:
        spotted_discontinuities.append(order_min[-1])
        return
    elif recursion_level > 1 and order_max[-1] == order_min[0]:
        spotted_discontinuities.append(order_max[-1])
        return
    elif order_min[-1] == order_max[-1]:
        pass
    elif order_min[-1] == order_max[-2] and \
            order_min[-2] == order_max[-1]:
        # Search for crossing
        if minimiser == "scipy_bounded":
            res = optimize.minimize_scalar(candidate_difference_squared,
                                           args=(order_min[-1], order_max[-1]),
                                           bounds=(beta_eval_min, beta_eval_max),
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
            beta_cross, res = fminfun(candidate_difference, beta_eval_min,
                                      beta_eval_max, xtol=1e-12,
                                      args=(order_max[-1], order_min[-1]),
                                      full_output=True)
            nfev = res.function_calls
            success = res.converged
        if success:
            candidates_cross, used_signs_cross = get_all_cand(beta_cross)
            if last_entry_changed not in support and \
                    last_entry_changed not in neglect_entries:
                subst_candidate_cross, subst_sign_cross = \
                                get_substitute_for_last_entry_changed(beta_cross)
                if used_signs_cross[last_entry_changed] == sign_last_entry_changed:
                    # Case the sign used was the correct sign to calculate the upper
                    # boundary, hence we have to substitute the entries in the
                    # candidate vector.
                    candidates_cross[last_entry_changed] = subst_candidate_cross[0]
                    used_signs_cross[last_entry_changed] = subst_sign_cross[0]
            candidates_cross[neglect_entries] = -1.0
            order_cross = np.argsort(candidates_cross)
            # Check if there is a third candidate larger than the two involved
            # candidates in the crossing
            if order_cross[-1] != order_min[-1] and \
                                            order_cross[-1] != order_max[-1]:
                # Case the crossing happened below another candidate. Make a
                # recursive call for both ends.
                divideAndConquer(support, signum, beta_cross, beta_max,
                                   minimiser, svdAAt_U, svdAAt_S, A, y,
                                   last_entry_changed,
                                   sign_last_entry_changed,
                                   recursion_level + 1,
                                   additional_indices=additional_indices,
                                   used_signs=used_signs,
                                   boundary_parameters=boundary_parameters,
                                   candidates_min=candidates_cross,
                                   used_signs_min=used_signs_cross,
                                   order_min=order_cross,
                                   candidates_max=candidates_max,
                                   used_signs_max=used_signs_max,
                                   order_max=order_max,
                                   neglect_entries=neglect_entries,
                                   spotted_discontinuities = spotted_discontinuities)
                divideAndConquer(support, signum, beta_min, beta_cross,
                                   minimiser, svdAAt_U, svdAAt_S, A, y,
                                   last_entry_changed,
                                   sign_last_entry_changed,
                                   recursion_level + 1,
                                   additional_indices=additional_indices,
                                   used_signs=used_signs,
                                   boundary_parameters=boundary_parameters,
                                   candidates_min=candidates_min,
                                   used_signs_min=used_signs_min,
                                   order_min=order_min,
                                   candidates_max=candidates_cross,
                                   used_signs_max=used_signs_cross,
                                   order_max=order_cross,
                                   neglect_entries=neglect_entries,
                                   spotted_discontinuities = spotted_discontinuities)
            else:
                additional_indices.append(order_min[-1])
                used_signs.append(used_signs_max[order_min[-1]])
                boundary_parameters.insert(-1, (candidates_cross[order_min[-1]],
                                                beta_cross))
        else:
            print """Warning: Could not find a crossing even if was suppposed to
                    find crossing at {0}, {1}, {2}, {3}""".format(x, res, fun,
                                                                  nfev)
    else:
        # Bisection step
        beta_mid = 0.5 * (beta_min + beta_max)
        candidates_mid, used_signs_mid = get_all_cand(beta_mid)
        if last_entry_changed not in support and \
                            last_entry_changed not in neglect_entries:
            subst_candidate_mid, subst_sign_mid = \
                                get_substitute_for_last_entry_changed(beta_mid)
            if used_signs_mid[last_entry_changed] == sign_last_entry_changed:
                # Case the sign used was the correct sign to calculate the upper
                # boundary, hence we have to substitute the entries in the
                # candidate vector.
                candidates_mid[last_entry_changed] = subst_candidate_mid[0]
                used_signs_mid[last_entry_changed] = subst_sign_mid[0]
        candidates_mid[neglect_entries] = -1.0
        order_mid = np.argsort(candidates_mid)
        if order_mid[-1] != order_max[-1]: # We could spare this also
            divideAndConquer(support, signum, beta_mid, beta_max,
                               minimiser, svdAAt_U, svdAAt_S, A, y,
                               last_entry_changed,
                               sign_last_entry_changed,
                               recursion_level + 1,
                               additional_indices=additional_indices,
                               used_signs=used_signs,
                               boundary_parameters=boundary_parameters,
                               candidates_min=candidates_mid,
                               used_signs_min=used_signs_mid,
                               order_min=order_mid,
                               candidates_max=candidates_max,
                               used_signs_max=used_signs_max,
                               order_max=order_max,
                               neglect_entries=neglect_entries,
                               spotted_discontinuities = spotted_discontinuities)
        if order_mid[-1] != order_min[-1]: # We could spare this also
            divideAndConquer(support, signum, beta_min, beta_mid,
                               minimiser, svdAAt_U, svdAAt_S, A, y,
                               last_entry_changed,
                               sign_last_entry_changed,
                               recursion_level + 1,
                               additional_indices=additional_indices,
                               used_signs=used_signs,
                               boundary_parameters=boundary_parameters,
                               candidates_min=candidates_min,
                               used_signs_min=used_signs_min,
                               order_min=order_min,
                               candidates_max=candidates_mid,
                               used_signs_max=used_signs_mid,
                               order_max=order_mid,
                               neglect_entries=neglect_entries,
                               spotted_discontinuities = spotted_discontinuities)
    if len(spotted_discontinuities) == 0 or \
            all(np.in1d(spotted_discontinuities, neglect_entries)):
        return additional_indices, boundary_parameters, used_signs
    else:
        # We found additional discontinuities during the processing of this
        # interval. Call again with neglecting this discontinuities.
        return divideAndConquer(support, signum, beta_min, beta_max,
                         minimiser, svdAAt_U, svdAAt_S, A, y,
                         last_entry_changed, sign_last_entry_changed, 1,
                         spotted_discontinuities = spotted_discontinuities)


def find_crossing_discontinuities(A, y, svdAAt_U, svdAAt_S, support, signum,
                                  beta_min, beta_max, only_entries = False):
    """
    Finds discontinuities with respect to beta in the cross_candidate that are
    calculated during the Lasso-path algorithm. The cross candidates, or penalty
    parameters yield the value at which a specific entry crosses zero again. If
    the shrinkage part of a specific entry of the solution u_{\beta,alpha} cross
    zero, discontinuities in the parameter functions can appear. Therefore we
    search for those here in a pre-processing step.

    Parameters
    -------------
    A : array, shape (n_measurements, n_features)
        Sensing matrix of the problem.

    y : array, shape (n_measurements)
        The vector of measurements.

    svdAAt_U : array, shape (n_measurements, n_measurements)
        Matrix U of the singular value decomposition of A*A^T.

    svdAAt_S : array, shape (n_measurements)
        Array S with singular values of singular value decomposition of A*A^T.

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

    only_entries : Boolean
        If True, instead of calculating the beta-value to an actual
        discontinuity of some curve of the family, we only return the respective
        entries.

    Returns
    -------------
    If 'only_entries' == False: A python list with all beta's at which at least
                                one of the candidate curves has a discontinuity.
    If 'only_entries' == True: np.array with subset of indices from the support
                               that exhibit discontinuities.
    """
    # Calculate denominator for crossing betas for the lower beta
    dummy, B_betaI_min, dummy2 = calc_B_y_beta_selection(A, y, svdAAt_U,
                                svdAAt_S, beta_min, I=support, J = [])
    cc_bottom_min = calc_cross_cand_bottom_selection(B_betaI_min, signum,
                                                    np.arange(support.shape[0]))
    dummy, B_betaI_max, dummy2 = calc_B_y_beta_selection(A, y, svdAAt_U,
                                svdAAt_S, beta_max, I=support, J = [])
    cc_bottom_max = calc_cross_cand_bottom_selection(B_betaI_max, signum,
                                                    np.arange(support.shape[0]))
    entries = np.where(np.logical_or(
                        np.logical_and(cc_bottom_min < 0, cc_bottom_max > 0),
                        np.logical_and(cc_bottom_min > 0, cc_bottom_max < 0)))[0]
    if only_entries:
        return support[entries]
    else:
        discontinuities = []
        for entry in entries:
            discontinuities.append(find_discontinuity_for_entry(
                                         A, y, svdAAt_U, svdAAt_S, support,
                                         signum, beta_min, beta_max, entry))
        return discontinuities

def find_discontinuity_for_entry(A, y, svdAAt_U, svdAAt_S, support, signum,
                                  beta_min, beta_max, entry):
    """
    Finds discontinuity with respect to beta for a specific crossing candidate
    regularisation parameter. A crossing candidate is equal to the penalty
    parameter that is necessary in the Lasso-functional to set the respective
    entry (from the support) to zero, ie. the entry crosses zero. With respect
    to beta, these can exhibit discontinuities if the shrinkage part in the
    entry vanishes for some beta.
    Assumes that it is ensured that the respective curve has a discontinuity
    (otherwise exceptions are thrown).

    Parameters
    -------------
    A : array, shape (n_measurements, n_features)
        Sensing matrix of the problem.

    y : array, shape (n_measurements)
        The vector of measurements.

    svdAAt_U : array, shape (n_measurements, n_measurements)
        Matrix U of the singular value decomposition of A*A^T.

    svdAAt_S : array, shape (n_measurements)
        Array S with singular values of singular value decomposition of A*A^T.

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

    entry : integer
        Index to crossing parameters for which we search the discontinuity
        (related to entries in the support, ie. support[entry] is a value
        between (1,n_features)).

    Returns
    -------------
    The value of the discontinuity. Throws RuntimeError if the starting values
    are such that there is no intersection.
    """
    def cc_bot(beta):
        dummy, B_betaI_min, dummy2 = calc_B_y_beta_selection(A, y, svdAAt_U,
                                    svdAAt_S, beta, I=support, J = [])
        cross_cand_bottom = calc_cross_cand_bottom_selection(B_betaI_min,
                                    signum, np.array([entry]).astype("int"))
        return cross_cand_bottom[0]
    try:
        intersection, res = optimize.bisect(cc_bot, beta_min,
                                    beta_max, xtol=1e-15, full_output=True)
    except ValueError:
        # Loose up the x-tolerance and check if one of the endpoints is almost zeros
        if np.abs(cc_bot(beta_min)) < 1e-12:
            return beta_min
        elif np.abs(cc_bot(beta_max)) < 1e-12:
            return beta_max
        else:
            raise
    nfev = res.function_calls
    success = res.converged
    if success:
        return intersection
    else:
        raise RuntimeError("Could not find discontinuity {0} {1} {2} {3}".format(
            res, beta_min, beta_max, res.f))

def lasso_post_process_children(additional_indices, boundary_parameters,
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
        if index in old_support:
            pos = bisect.bisect_left(old_support, index)
            new_support = np.delete(old_support, pos)
            new_signum = np.delete(old_signum, pos)
        else:
            new_support = np.append(old_support, index)
            new_signum = np.append(old_signum, sign)
            # Restore order
            order = np.argsort(new_support)
            new_support, new_signum = new_support[order], new_signum[order]
        param_min = boundary_parameters[i+1]
        param_max = boundary_parameters[i]
        region_refinement.append([param_min, param_max, new_support,
            new_signum])
    return region_refinement

def lasso_children_merge(children):
    """ Method to perform an in-layer merge for children of a tiling element.
    Since we have to refine the interval when searching for children with the
    Lasso-path algorithm into small sub-intervals where we have a distinct
    parent, and no discontinuities, it can occur that we find the same children
    for two sub-intervals, and that they have continuous beta-ranges. Then we
    have to merge these children using this method.

    Parameters
    ------------
    children: list of tuples
        Each tuple contains key information about a new tiling element, ie. the
        tuples contain entries (minimal_parameter, maximal_parameters,
        new_support, new_signum). Thus the method expects an output of
        'lasso_post_process_children'.

    Returns
    --------------
    The same list of children but with children merged together if they
    share a common support and sign pattern, and if they have consecutive
    beta-ranges.

    Remarks
    --------------
    Expects the children to be sorted by increasing beta-ranges, ie. only
    compares two consecutive children in the list for merging.
    """
    ctr = 1
    n_children = len(children)
    while ctr < n_children:
        # Check if support and signum coincide
        if np.array_equal(children[ctr-1][2], children[ctr][2]) and \
                np.array_equal(children[ctr-1][3], children[ctr][3]):
            children[ctr-1][1] = children[ctr][1] # Replacing maximum parameters
            del children[ctr]
            n_children = n_children - 1
        else:
            ctr += 1
    return children
