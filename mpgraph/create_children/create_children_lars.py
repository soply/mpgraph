# coding: utf8

import numpy as np
from scipy import optimize

from lasso_path_utils import calc_hit_cand, calc_hit_cand_selection
from ..mp_utils import calc_B_y_beta, calc_B_y_beta_selection

def create_children_lars(tiling_element, beta_min, beta_max):
    additional_indices, boundary_parameters, used_signs = \
                aux_create_children_lars(tiling_element.support,
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

def aux_create_children_lars(support, signum, beta_min, beta_max,
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
        boundary_parameters.append((hit_candidates_max[order_max[-1]],
                                    beta_max))
        boundary_parameters.append((hit_candidates_min[order_min[-1]],
                                    beta_min))
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
                aux_create_children_lars(support, signum, beta_min, beta_cross,
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
                aux_create_children_lars(support, signum, beta_cross, beta_max,
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
            aux_create_children_lars(support, signum, beta_mid, beta_max,
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
            aux_create_children_lars(support, signum, beta_min, beta_mid,
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
    """ Takes the results of 'aux_create_children_lars' method and translates them
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
