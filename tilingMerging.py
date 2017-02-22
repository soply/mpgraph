# coding: utf8
""" Merging operations for tiling creation. Currently implemented:
    -merge_new_children_faster for create_tiling in Tiling.py
    -merge_new_children_simple for create_tiling_simple in Tiling.py """

__author__ = "Timo Klock"

import numpy as np


def merge_new_children_faster(children, stack, n_sparsity):
    """ Merging operation for the fast tiling creation approach.

    This method is used to merge a given set of newly discovered children (that
    is a list of tiling elements that have just been found, ie. that have
    no children and are not in the stack) into an existing tiling. The
    tiling itself is implicitly defined by the parent relation given for the
    children. This merging operation is sensitive to its requirements on the
    inputs in the following sense:
        -all tiling elements in children have no children themselves and are
        not in the current stack of nodes that need to be processed later
        -all tiling elements in stack have no children yet since they have
        not been processed yet. In this fast merging procedure, tiling elements
        do not become member of the stack twice. Whenever the beta-range of an
        already processed tiling element has been extended, making the
        respective node unfinished again, we process it immediately and do not
        put in on the stack again (check tiling.py and compare the
        create_tiling procedures -> this is exactly the difference between the
        fast and simple approach).

    Assuming these requirements are satisfied, the merging procedure operates as
    follows:

        1) Search for left and right candidates to merge the first and the last
        node in the given children set. Note that the nodes in the middle can
        never be merged at this stage since the respective neighbors have been
        estimated outgoing from the same tiling such that they would not have
        been seperated in the first place.

        2) Depending on the situation, perform one of the following actions:

            a) If left and right candidates have been found, and there is only
            one children that means we have a 'merge_both' situation in which
            we create a single tiling element from one child and two nodes
            that are already in the tiling. Depending on whether or not they are
            on the stack (ie. have already been processed), different actions
            have to be taken (see comments). At the end, the left candidate will
            be the surviving one and, in case one of the candidates was
            already processed, the left candidates with extended beta range will
            become an uncompleted_child. If both candidates hadn't been processed
            before, we keep only the left candidate with extended beta range
            on the stack.

            b) If only a left candidate has been found, we have to merge it
            with children[0], and depending on whether or not left_candidate is
            still on the stack or not, we have to create an uncompleted_child
            and process it immediately and just change the respective beta
            range of the left candidate and wait until it's pop'ed from the
            stack. Note that children[0] has to be removed from the list of
            children since we put the remaining children on the stack in the
            end.

            c) Similar to b) but with right_candidate and children[-1].

        3) The remaining children will be appended to the current stack. The
        uncompleted_children that have been formed due to merging procedures
        with tiling elements that already have children will be processed
        immediately in the while loop in tiling.py.

    Parameters
    ------------
    children : python list containing objects of class tilingElement.
        Corresponds to a list of children that has been found last by applying
        find_children routines on a specific tiling element.

    stack : python list containing objects of class tilingElement.
        Discovered, but so far unprocessed tiling elements in the current
        tiling. Tiling elements in stack do not have any children (otherwise
        the algorithm does not work).

    n_sparsity : integer > 0
        Upper bound on the support size until which we new elements are added
        to the stack.

    Returns
    -----------
    uncompleted_children, that is a list of tuples (A, B, C) where A is a
    tiling element that has been processed before but becomes unfinished due to
    merging actions, (B, C) are lower and upper bounds on the beta parameter
    forming the range of beta's that we have to process and find children for.

    Remarks
    ------------
    Method can alter children and stack input arguments.
    """
    if len(children) == 0:
        return []
    children.sort(key=lambda x: x.beta_max)
    left_candidate = children[0].find_left_merge_candidate()
    right_candidate = children[-1].find_right_merge_candidate()
    uncompleted_children = []
    if len(children) == 1 and left_candidate is not None and \
            right_candidate is not None:
        # This is a special case since left and right candidate belong
        # to the same area which was previously intersected by the given
        # children[0] node. Here we have to be especially careful to restore
        # the correct relations, and the stack with which we go on
        # afterwards.

        # Assemble uncompleted children area first since we override children
        if len(left_candidate.children) + len(right_candidate.children) > 0:
            # Case at least one of the nodes is NOT on the stack and we have
            # to build the correct uncompleted children and process it
            # immediately. Stack has to be cleared up from tiling elements.
            if len(left_candidate.children) == 0:
                uncompleted_children.append((left_candidate,
                                             left_candidate.beta_min,
                                             children[0].beta_max))
                # Remove left candidate from the stack since it will be
                # handled in the uncompleted_children loop
                stack.remove(left_candidate)
            elif len(right_candidate.children) == 0:
                # Remark: The left candidate will be the surviving one with
                # the correct relations, hence we need to append the left
                # one here as well.
                uncompleted_children.append((left_candidate,
                                             children[0].beta_min,
                                             right_candidate.beta_max))
                # Remove right candidate from the stack since it will be
                # handled in the uncompleted_children loop
                stack.remove(right_candidate)
            else:
                # Remark: The left candidate will be the surviving one with
                # the correct relations, hence we need to append the left
                # one here as well.
                # Case where both sourrounding nodes have already been
                # processed.
                uncompleted_children.append((left_candidate,
                                             children[0].beta_min,
                                             children[0].beta_max))
        else:
            # Case both candidates are still on the stack. Since only the left
            # candidate will remain, we remove the right_candidate from the
            # stack.
            stack.remove(right_candidate)

        if children[0].options['verbose'] > 1:
            print "Merging both {0} with {1} and {2}".format(left_candidate,
                                                            children[0],
                                                            right_candidate)
        # Case left_candidate + right_candidate + children node belong
        # to the same tiling element.
        left_candidate.alpha_max = right_candidate.alpha_max
        left_candidate.beta_max = right_candidate.beta_max
        left_candidate.parents += children[0].parents + right_candidate.parents
        left_candidate.uniquefy_parents()
        left_candidate.children += right_candidate.children
        left_candidate.uniquefy_children()
        # Fix parents of right candidate by replacing the child
        for parent in right_candidate.parents:
            parent[0].replace_child(right_candidate, left_candidate)
            parent[0].sort_children()
            parent[0].uniquefy_children()
        # Fix parents of children[0] by replacing the child
        for parent in children[0].parents:
            parent[0].replace_child(children[0], left_candidate)
            parent[0].sort_children()
            parent[0].uniquefy_children()
        # Fix children of right_candidate by replacing the parent
        for child in right_candidate.children:
            child[0].replace_parent(right_candidate, left_candidate)
            child[0].sort_parents()
            child[0].uniquefy_children()
        # Remove child from children list, since it will be processed
        # immediately
        del children[0]
    else:
        if left_candidate is not None:
            if children[0].options['verbose'] > 1:
                print "Merging left {0} with {1}".format(left_candidate,
                                                         children[0])
            # Case left_candidate + children node belong
            # to the same tiling element.
            left_candidate.alpha_max = children[0].alpha_max
            left_candidate.beta_max = children[0].beta_max
            left_candidate.parents += children[0].parents
            left_candidate.sort_parents()
            left_candidate.uniquefy_parents()
            # Fix parents of children[0] by replacing the child
            for parent in children[0].parents:
                parent[0].replace_child(children[0], left_candidate)
                parent[0].sort_children()
                parent[0].uniquefy_children()
            if len(left_candidate.children) > 0:
                # In this case the left candidate is not in the 'to-process'
                # stack anymore, hence we need to compute the left-over
                # children directly.
                uncompleted_children.append((left_candidate,
                                             children[0].beta_min,
                                             children[0].beta_max))
            # Remove child from children list, since it will either be processed
            # immediately (if left_candidate had children) or later due to the
            # enlarged left_candidate range.
            del children[0]
        if right_candidate is not None:
            if children[-1].options['verbose'] > 1:
                print "Merging right {0} with {1}".format(children[-1],
                                                          right_candidate)
            # Case right_candidate + children node belong
            # to the same tiling element.
            right_candidate.alpha_min = children[-1].alpha_min
            right_candidate.beta_min = children[-1].beta_min
            right_candidate.parents += children[-1].parents
            right_candidate.sort_parents()
            right_candidate.uniquefy_parents()
            # Fix parents of children[-1] by replacing the child
            for parent in children[-1].parents:
                parent[0].replace_child(children[-1], right_candidate)
                parent[0].sort_children()
                parent[0].uniquefy_children()
            if len(right_candidate.children) > 0:
                # In this case the right candidate is not in the 'to-process'
                # stack anymore, hence we need to compute the left-over
                # children directly.
                uncompleted_children.append((right_candidate,
                                             children[-1].beta_min,
                                             children[-1].beta_max))
            # Remove child from children list, since it will either be processed
            # immediately (if right_candidate had children) or later due to the
            # enlarged right_candidate range.
            del children[-1]
    # Extend the stack with non-uncompleted children
    stack.extend(list(filter_children_sparsity(children, n_sparsity)))
    return uncompleted_children


def merge_new_children_simple(children, stack, n_sparsity):
    """ Merging operation for the simple tiling creation approach.

    This method is used to merge a given set of newly discovered children (that
    is a list of tiling elements that have just been found, ie. that have
    no children and are not in the stack) into an existing tiling. The
    tiling itself is implicitly defined by the parent relation given for the
    children. This merging operation is sensitive to its requirements on the
    inputs in the following sense:
        -all tiling elements in children have no children themselves and are
        not in the current stack of nodes that need to be processed later

    Assuming this requirement is satisfied, the merging procedure operates as
    follows:

        1) Search for left and right candidates to merge the first and the last
        element in the given children set. Note that the nodes in the middle can
        never be merged at this stage since the respective neighbors have been
        estimated outgoing from the same tiling such that they would not have
        been seperated in the first place.

        2) Depending on the situation, perform one of the following actions:

            a) If left and right candidates have been found, and there is only
            one children that means we have a 'merge_both' situation in which
            we create a single tiling element from one child and two nodes
            that are already in the tiling. Depending on whether or not they are
            on the stack (ie. have already been processed), different actions
            have to be taken (see comments). At the end, the left candidate will
            be the surviving one, and it will be the only one remaining on the
            stack with an adjusted beta range that may incoporate the old left
            candidate, the child, and the right candidate.

            b) If only a left candidate has been found, we have to merge it
            with children[0], and depending on whether or not left_candidate is
            still on the stack or not, we have to add it to the stack again or
            we just have to adjust its respective beta range to incorporate the
            range of the child node.

            c) Similar to b) but with right_candidate and children[-1].

        3) The remaining children will be appended to the current stack with the
        range that was assigned to them during their discovery.

    Parameters
    ------------
    children : python list containing objects of class tilingElement.
        Corresponds to a list of children that has been found last by applying
        find_children routines on a specific tiling element.

    stack : python list containing objects of class tilingElement.
        Discovered, but so far unprocessed tiling elements in the current
        tiling.

    n_sparsity : integer > 0
        Upper bound on the support size until which we new elements are added
        to the stack.

    Remarks
    ------------
    Method always alters stack and maybe also children input element.
    """
    if len(children) == 0:
        return [], []
    children.sort(key=lambda x: x.beta_max)
    left_candidate = children[0].find_left_merge_candidate()
    index_l = [i for i, cand in enumerate(stack)
                                            if cand[0] == left_candidate]
    right_candidate = children[-1].find_right_merge_candidate()
    index_r = [i for i, cand in enumerate(stack)
                                            if cand[0] == right_candidate]
    if len(children) == 1 and left_candidate is not None and \
            right_candidate is not None:
        # This is a special case since left and right candidate belong
        # to the same area which was previously intersected by the given
        # children[0] node. Here we have to be especially careful to restore
        # the correct relations, and the stack with which we go on
        # afterwards.

        if len(index_l) + len(index_r) > 0:
            # Case at least one of the nodes is on the current stack. We have to
            # figure out which and leave the correct one on the stack. Merging
            # will happen afterwards.
            if len(index_l) == 0:
                # Only right node is on the stack -> Replace it by the left
                # candidate with the beta range incorporating the new child
                # as well as the right candidates left-over range
                left_boun = np.minimum(children[0].beta_min,
                                       stack[index_r[0]][1])
                right_boun = np.maximum(children[0].beta_max,
                                       stack[index_r[0]][2])
                # Replace right_candidate on the stack since this tiling
                # element will be erased after the merging
                stack[index_r[0]] = (left_candidate, left_boun, right_boun)
            elif len(index_r) == 0:
                # Only the left node is on the stack -> Adjust the related
                # of betas by incorporating the children range.
                left_boun = np.minimum(children[0].beta_min,
                                       stack[index_l[0]][1])
                right_boun = np.maximum(children[0].beta_max,
                                       stack[index_l[0]][2])
                # Replace left_candidate with updated version
                stack[index_l[0]] = (left_candidate, left_boun, right_boun)
            else:
                # Remark: The left candidate will be the surviving one with
                # the correct relations, hence we need to append the left
                # one here as well.
                # Case both elements are currently on the stack
                left_boun = np.minimum(np.minimum(children[0].beta_min,
                                                  stack[index_l[0]][2],
                                                  stack[index_r[0]][2]))
                right_boun = np.maximum(np.maximum(children[0].beta_max,
                                                  stack[index_l[0]][2],
                                                  stack[index_r[0]][2]))
                # Replace left candidate on stack with adjusted ranges.
                # Remove right candidate from the stack
                stack[index_l[0]] = (left_candidate, left_boun, right_boun)
                del stack[index_r[0]]
        else:
            # Case no element is on the stack anymore. In this case we have to
            # add the left candidate with they left and right boundaries of
            # the new child
            stack.append((left_candidate, children[0].beta_min,
                         children[0].beta_max))

        if children[0].options['verbose'] > 1:
            print "Merging both {0} with {1} and {2}".format(left_candidate,
                                                            children[0],
                                                            right_candidate)
        # Case left_candidate + right_candidate + children node belong
        # to the same tiling element.
        left_candidate.alpha_max = right_candidate.alpha_max
        left_candidate.beta_max = right_candidate.beta_max
        left_candidate.parents += children[0].parents + right_candidate.parents
        left_candidate.uniquefy_parents()
        left_candidate.children += right_candidate.children
        left_candidate.uniquefy_children()
        # Fix parents of right candidate by replacing the child
        for parent in right_candidate.parents:
            parent[0].replace_child(right_candidate, left_candidate)
            parent[0].sort_children()
            parent[0].uniquefy_children()
        # Fix parents of children[0] by replacing the child
        for parent in children[0].parents:
            parent[0].replace_child(children[0], left_candidate)
            parent[0].sort_children()
            parent[0].uniquefy_children()
        # Fix children of right_candidate by replacing the parent
        for child in right_candidate.children:
            child[0].replace_parent(right_candidate, left_candidate)
            child[0].sort_parents()
            child[0].uniquefy_children()
        # Remove the child node since it will not need to be to be put on the
        # stack in the end again (because we merge here always).
        del children[0]
    else:
        if left_candidate is not None:
            if children[0].options['verbose'] > 1:
                print "Merging left {0} with {1}".format(left_candidate,
                                                         children[0])
            # First adjust the stack, afterwards we take care of relations
            if len(index_l) > 0:
                # Case left candidate is on the stack
                left_boun = np.minimum(stack[index_l[0]][1],
                                       children[0].beta_min)
                right_boun = np.maximum(stack[index_l[0]][2],
                                       children[0].beta_max)
                stack[index_l[0]] = (left_candidate, left_boun, right_boun)
            elif len(left_candidate.support) < n_sparsity:
                left_boun = children[0].beta_min
                right_boun = children[0].beta_max
                stack.append((left_candidate, left_boun, right_boun))
            # Fix relations
            left_candidate.alpha_max = children[0].alpha_max
            left_candidate.beta_max = children[0].beta_max
            left_candidate.parents += children[0].parents
            left_candidate.sort_parents()
            left_candidate.uniquefy_parents()
            # Fix parents of children[0] by replacing the child
            for parent in children[0].parents:
                parent[0].replace_child(children[0], left_candidate)
                parent[0].sort_children()
                parent[0].uniquefy_children()
            # Remove from the children since we do not need to put it on the
            # stack at the end of the method
            del children[0]
        if right_candidate is not None:
            if children[-1].options['verbose'] > 1:
                print "Merging right {0} with {1}".format(children[-1],
                                                          right_candidate)
            # First adjust the stack, afterwards we take care of relations
            if len(index_r) > 0:
                left_boun = np.minimum(stack[index_r[0]][1],
                                       children[-1].beta_min)
                right_boun = np.maximum(stack[index_r[0]][2],
                                       children[-1].beta_max)
                stack[index_r[0]] = (right_candidate, left_boun, right_boun)
            elif len(right_candidate.support) < n_sparsity:
                left_boun = children[-1].beta_min
                right_boun = children[-1].beta_max
                stack.append((right_candidate, left_boun, right_boun))
            # Fix relations
            right_candidate.alpha_min = children[-1].alpha_min
            right_candidate.beta_min = children[-1].beta_min
            right_candidate.parents += children[-1].parents
            right_candidate.sort_parents()
            right_candidate.uniquefy_parents()
            # Fix parents of children[0] by replacing the child
            for parent in children[-1].parents:
                parent[0].replace_child(children[-1], right_candidate)
                parent[0].sort_children()
                parent[0].uniquefy_children()
            # Remove from the children since we do not need to put it on the
            # stack at the end of the method
            del children[-1]
    children = list(filter_children_sparsity(children, n_sparsity))
    to_extend_with = [(child, child.beta_min, child.beta_max) for child in
                                                                    children]
    stack.extend(to_extend_with)


def filter_children_sparsity(children, sparsity_bound):
    """ Creates and iterator out of a given list of children that yields only
    children whose support is below a specific size. The size is specified by
    sparsity_bound.

    Parameters
    -----------------
    children : python iterable of objects of class TilingElement
        List of tiling elements that are candidates to be yielded by the
        resulting iterator.

    sparsity_bound : integer
        Upper bound for the support size of tiling elements that we will yield.

    Returns
    ------------------
    Python iterator of tiling elements with supports below a given upper bound.
    """
    for child in children:
        if len(child.support) < sparsity_bound:
            yield child
