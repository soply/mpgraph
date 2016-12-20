# coding: utf8
import copy
from itertools import repeat

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy import optimize

from lasso_path_utils import calc_hit_cand, calc_hit_cand_selection
from mp_utils import calc_B_y_beta, calc_B_y_beta_selection


def create_children_LARS(support, signum, beta_min, beta_max,
                         minimiser, svdAAt_U, svdAAt_S, A, y,
                         additional_indices=None,
                         used_signs=None,
                         boundary_parameters=None,
                         hit_candidates_min=None,
                         used_signs_min=None,
                         order_min=None,
                         hit_candidates_max=None,
                         used_signs_max=None,
                         order_max=None):
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
            x, res = fminfun(candidate_difference, beta_min,
                             beta_max, xtol=1e-12,
                             args=(order_max[-1], order_min[-1]),
                             full_output=True)
            nfev = res.function_calls
            fun = 1e-12
            success = res.converged
        if success:
            hit_candidates_cross, used_signs_cross = get_all_hit_cand(x)
            additional_indices.append(order_min[-1])
            used_signs.append(used_signs_max[order_min[-1]])
            boundary_parameters.insert(-1, (hit_candidates_cross[order_min[-1]],
                                            x))
        else:
            print """Warning: Could not find a crossing even if was suppposed to
                    find crossing at {0}, {1}, {2}, {3}""".format(x, res, fun,
                                                                  nfev)
    else:
        # Bisection step
        beta_mid = 0.5 * (beta_min + beta_max)
        hit_candidates_mid, used_signs_mid = get_all_hit_cand(beta_mid)
        order_mid = np.argsort(hit_candidates_mid)
        if order_mid[-1] != order_max[-1]:
            create_children_LARS(support, signum, beta_mid, beta_max,
                               minimiser, svdAAt_U, svdAAt_S, A, y,
                               additional_indices=additional_indices,
                               used_signs=used_signs,
                               boundary_parameters=boundary_parameters,
                               hit_candidates_min=hit_candidates_mid,
                               used_signs_min=used_signs_mid,
                               order_min=order_mid,
                               hit_candidates_max=hit_candidates_max,
                               used_signs_max=used_signs_max,
                               order_max=order_max)
        if order_mid[-1] != order_min[-1]:
            create_children_LARS(support, signum, beta_min, beta_mid,
                               minimiser, svdAAt_U, svdAAt_S, A, y,
                               additional_indices=additional_indices,
                               used_signs=used_signs,
                               boundary_parameters=boundary_parameters,
                               hit_candidates_min=hit_candidates_min,
                               used_signs_min=used_signs_min,
                               order_min=order_min,
                               hit_candidates_max=hit_candidates_mid,
                               used_signs_max=used_signs_mid,
                               order_max=order_mid)
    return additional_indices, boundary_parameters, used_signs

def post_process_children(additional_indices, boundary_parameters,
        used_signs, support, old_sign):
    region_refinement = []
    for i, (index, sign) in enumerate(zip(additional_indices, used_signs)):
        new_support = np.append(support, index)
        new_signum = np.append(old_sign, sign)
        order = np.argsort(new_support)
        new_support, new_signum = new_support[order], new_signum[order]
        param_min = boundary_parameters[i+1]
        param_max = boundary_parameters[i]
        region_refinement.append([param_min, param_max, new_support,
            new_signum])
    return region_refinement
