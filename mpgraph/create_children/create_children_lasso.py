# coding: utf8
import bisect
import numpy as np
from scipy import optimize

from lasso_path_utils import (calc_all_cand, calc_cross_cand,
                              calc_cross_cand_selection, calc_hit_cand,
                              calc_hit_cand_selection)
from ..mp_utils import calc_B_y_beta, calc_B_y_beta_selection


def create_children_lasso(support, signum, beta_min, beta_max,
                         minimiser, svdAAt_U, svdAAt_S, A, y,
                         last_entry_changed,
                         additional_indices=None,
                         used_signs=None,
                         boundary_parameters=None,
                         candidates_min=None,
                         used_signs_min=None,
                         order_min=None,
                         candidates_max=None,
                         used_signs_max=None,
                         order_max=None,
                         neglect_entries=None):

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
        
    # Initialisation
    if additional_indices is None:
        candidates_max, used_signs_max = get_all_cand(beta_max)
        candidates_min, used_signs_min = get_all_cand(beta_min)
        # Filter out all candidates that are above the current curve
        boun_at_max = candidates_max[last_entry_changed]
        boun_at_min = candidates_min[last_entry_changed]
        # FIXME: It is safer to add even a small tolerance such that we really
        #        only consider cross_candidates that are smaller in the whole
        #        range [beta_min, beta_max].
        neglect_entries = np.where(np.logical_or(candidates_min >= boun_at_min,
                                   candidates_max >= boun_at_max))[0]
        # Neglect also entries that have a minus value either at boun_at_max or
        # at boun_at_min
        neglect_entries_in_support = find_negligible_entries(A, y, svdAAt_U,
                                                             svdAAt_S, support,
                                                             signum, beta_min,
                                                             beta_max)
        neglect_entries_in_support = support[neglect_entries_in_support]
        neglect_entries = np.intersect1d(neglect_entries, support) # Because hit candidates are automatically smaller (should be automatically null-intersection here)
        neglect_entries = np.append(neglect_entries, neglect_entries_in_support)
        candidates_min[neglect_entries] = -1.0
        candidates_max[neglect_entries] = -1.0
        order_max = np.argsort(candidates_max)
        order_min = np.argsort(candidates_min)
        additional_indices = [order_max[-1]]
        used_signs = [used_signs_max[order_max[-1]]]
        boundary_parameters = []
        boundary_parameters.append((candidates_max[order_max[-1]],
                                    beta_max))
        boundary_parameters.append((candidates_min[order_min[-1]],
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
            candidates_crossing, used_signs_crossing = get_all_cand(x)
            additional_indices.append(order_min[-1])
            used_signs.append(used_signs_max[order_min[-1]])
            boundary_parameters.insert(-1, (candidates_crossing[order_min[-1]],
                                            x))
        else:
            print """Warning: Could not find a crossing even if was suppposed to
                    find crossing at {0}, {1}, {2}, {3}""".format(x, res, fun,
                                                                  nfev)
    else:
        # Bisection step
        beta_mid = 0.5 * (beta_min + beta_max)
        candidates_mid, used_signs_mid = get_all_cand(beta_mid)
        candidates_mid[neglect_entries] = -1.0
        order_mid = np.argsort(candidates_mid)
        if order_mid[-1] != order_max[-1]:
            create_children_lasso(support, signum, beta_mid, beta_max,
                               minimiser, svdAAt_U, svdAAt_S, A, y,
                               last_entry_changed,
                               additional_indices=additional_indices,
                               used_signs=used_signs,
                               boundary_parameters=boundary_parameters,
                               candidates_min=candidates_mid,
                               used_signs_min=used_signs_mid,
                               order_min=order_mid,
                               candidates_max=candidates_max,
                               used_signs_max=used_signs_max,
                               order_max=order_max,
                               neglect_entries=neglect_entries)
        if order_mid[-1] != order_min[-1]:
            create_children_lasso(support, signum, beta_min, beta_mid,
                               minimiser, svdAAt_U, svdAAt_S, A, y,
                               last_entry_changed,
                               additional_indices=additional_indices,
                               used_signs=used_signs,
                               boundary_parameters=boundary_parameters,
                               candidates_min=candidates_min,
                               used_signs_min=used_signs_min,
                               order_min=order_min,
                               candidates_max=candidates_mid,
                               used_signs_max=used_signs_mid,
                               order_max=order_mid,
                               neglect_entries=neglect_entries)
    return additional_indices, boundary_parameters, used_signs

def lasso_post_process_children(additional_indices, boundary_parameters,
        used_signs, old_support, old_signum):
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

def find_negligible_entries(A, y, svdAAt_U, svdAAt_S, support, signum,
                            beta_min, beta_max):
    # Calculate denominator for crossing betas for the lower beta
    B_beta_min, dummy = calc_B_y_beta(A, y, svdAAt_U, svdAAt_S, beta_min)
    AI_min = B_beta_min[:, support]
    inverseAtA_min = np.linalg.inv(AI_min.T.dot(AI_min))
    aux_bot_min = inverseAtA_min.dot(signum)
    # Calculate denominator for crossing betas for the upper beta
    B_beta_max, dummy = calc_B_y_beta(A, y, svdAAt_U, svdAAt_S, beta_max)
    AI_max = B_beta_max[:, support]
    inverseAtA_max = np.linalg.inv(AI_max.T.dot(AI_max))
    aux_bot_max = inverseAtA_max.dot(signum)
    return np.where(np.multiply(aux_bot_min, aux_bot_max) < 0)[0]