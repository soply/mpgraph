#coding: utf8
from itertools import groupby

import numpy as np

from create_children_LARS import create_children_LARS, post_process_children


class TilingElement(object):

    def __init__(self, beta_min, beta_max, support, sign_pattern, parents,
                 A, y, svdAAt_U, svdAAt_S, n_element, options=None):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.support = support
        self.sign_pattern = sign_pattern
        self.parents = parents
        self.children = []
        # References to global data matrices and data
        self.A = A
        self.y = y
        self.svdAAt_U = svdAAt_U
        self.svdAAt_S = svdAAt_S
        self.options = options
        # Identifier for tiling element that also defines unique order for the
        # tiling elements.
        self.n_element = n_element

    def __repr__(self):
        return ("TilingElement(Support: {0}, Sign pattern: {1}" + \
                " Beta range: {2})").format(self.support, self.sign_pattern,
                                        [self.beta_min, self.beta_max])



    def find_children(self, n_element, beta_min=None, beta_max=None):
        if beta_min is None:
            beta_min = self.beta_min
        if beta_max is None:
            beta_max = self.beta_max
        if self.options["mode"] == "LARS":
            additional_indices, boundary_parameters, used_signs = \
                create_children_LARS(self.support, self.signum, beta_min,
                                     beta_max,
                                     self.options["env_minimiser"],
                                     self.svdAAt_U, self.svdAAt_S, self.A,
                                     self.y)
            children = post_process_children(additional_indices,
                                             boundary_parameters,
                                             used_signs, self.support,
                                             self.old_sign)
        elif self.options["mode"] == "TEST":
            children = self.options["test_iterator"].next()
        else:
            raise NotImplementedError("Mode not implemented.")
        children.sort(key = lambda x: x[0][1])
        new_tes = []
        for (i, child) in enumerate(children):
            new_tes.append(self.add_child(child[0][1], child[1][1], child[2],
                                            child[3], n_element + i))
        return new_tes

    def shares_support_with(self, tilingelement):
        return np.array_equal(tilingelement.support, self.support) and \
            np.array_equal(tilingelement.sign_pattern, self.sign_pattern)

    def add_child(self, beta_min, beta_max, support, signum, n_element):
        te = TilingElement(beta_min, beta_max, support, signum,
                            [self], self.A, self.y, self.svdAAt_U,
                            self.svdAAt_S, n_element, self.options)
        self.children.append(te)
        return te

    def sort_children(self):
        # self.children.sort(key=lambda x: x.beta_min)
        self.children.sort(key=lambda x: x.n_element)

    def sort_parents(self):
        self.children.sort(key=lambda x: x.n_element)

    def uniquefy_children(self):
        # Assumes that children are in sorted order!
        self.children = [x[0] for x in groupby(self.children)]

    def uniquefy_parents(self):
        # Assumes that children are in sorted order!
        self.parents = [x[0] for x in groupby(self.parents)]

    def replace_child(self, child_to_replace, replacement):
        ctr=0
        while ctr < len(self.children):
            if self.children[ctr] != child_to_replace:
                ctr += 1
            else:
                self.children[ctr]=replacement
                return 1
        else:
            print ("Could not find child {0} in the children of child {1} and" + \
                " thus could not replace it with child {2}").format(self,
                child_to_replace, replacement)

    def replace_parent(self, parent_to_replace, replacement):
        ctr=0
        while ctr < len(self.parents):
            if self.parents[ctr] != parent_to_replace:
                ctr += 1
            else:
                self.parents[ctr]=replacement
                return 1
        else:
            print ("Could not find child {0} in the parents of child {1} and" + \
                " thus could not replace it with child {2}").format(self,
                parent_to_replace, replacement)

    def oldest_child(self):
        if len(self.children) > 0:
            return self.children[0]
        else:
            print ("Tiling element: {0}\nNo children but asking for the" + \
                " oldest child. ").format(self)
            return None

    def youngest_child(self):
        if len(self.children) > 0:
            return self.children[-1]
        else:
            print ("Tiling element: {0}\nNo children but asking for the" + \
                " youngest child. ").format(self)
            return None

    def oldest_parent(self):
        if self.parents is not None and len(self.parents) > 0:
            return self.parents[0]
        else:
            print ("Tiling element: {0}\nNo parent but asking for the oldest" + \
                " parent. ").format(self)
            return None

    def youngest_parent(self):
        if self.parents is not None and len(self.parents) > 0:
            return self.parents[-1]
        else:
            print ("Tiling element: {0}\nNo parent but asking for the" + \
                " oldest parent. ").format(self)
            return None

    def find_next_older_neighbor(self, child):
        ctr=1
        n_children=len(self.children)
        while ctr < n_children:
            if self.children[ctr] == child:
                return self.children[ctr - 1]
            ctr = ctr + 1
        else:
            print ("Could not find next older neighbor of node {0} in " + \
                " children of node {1}.").format(child, self)
            return None

    def find_next_younger_neighbor(self, child):
        ctr=0
        n_children=len(self.children)
        while ctr + 1 < n_children:
            if self.children[ctr] == child:
                return self.children[ctr + 1]
            ctr = ctr + 1
        else:
            print ("Could not find next younger neighbor of node {0} in" + \
                " children of node {1}.").format(child, self)
            return None

    # def merge_left(self):
    #     """ Applicable to merge a new node into an existing tiling. By new node
    #     we mean a node that has no children yet and that has only a single
    #     parent. """
    #     assert len(self.children) == 0 and len(self.parents) == 1
    #     current_node=self
    #     while current_node.oldest_parent() is not None and \
    #             current_node.oldest_parent().oldest_child() == current_node:
    #         current_node=current_node.oldest_parent()
    #     # Reaching the root without finding younger children -> self is node of
    #     # the leftmost's part of the graph.
    #     if current_node.oldest_parent() is None:
    #         print "Stopping merge_left operation on {0}".format(self)
    #         return 0, []
    #     # Get parent
    #     parent=current_node.oldest_parent()
    #     # Get next older sibling
    #     current_node=parent.find_next_older_neighbor(current_node)
    #     # Find matching support and sign pattern
    #     while not current_node.shares_support_with(self) and \
    #                 len(current_node.children) > 0:
    #         current_node=current_node.youngest_child()
    #     if current_node is not None and current_node != self and \
    #             current_node.shares_support_with(self):
    #         # Found a node to merge with
    #         # Fix beta max and parents! Note that the self node should not have
    #         # children yet.
    #         print "Merging {0} with {1}".format(current_node, self)
    #         current_node.beta_max=self.beta_max
    #         for parent in self.parents:
    #             if current_node in parent.children:
    #                 parent.children.remove(self)
    #             else:
    #                 parent.replace_child(self, current_node)
    #         if len(current_node.children) > 0:
    #             new_childless_tilingelements=current_node.find_children(
    #                                             beta_min=self.beta_min,
    #                                             beta_max=self.beta_max)
    #             return 1, new_childless_tilingelements
    #         else:
    #             import pdb
    #             pdb.set_trace()
    #             return 1, []
    #     else:
    #         print "Stopping merge_left operation on {0}".format(self)
    #         return 0, []
    #
    # def merge_right(self):
    #     current_node=self
    #     while current_node.youngest_parent() is not None and \
    #             current_node.youngest_parent().youngest_child() == current_node:
    #         current_node=current_node.youngest_parent()
    #     if current_node.youngest_parent() is None:
    #         print "Stopping merge_right operation on {0}".format(self)
    #         # In case we reached the root without having found a candidate for
    #         # merging the node
    #         return 0, []
    #     # Get oldest parent
    #     parent=current_node.youngest_parent()
    #     # Get next older sibling
    #     current_node=parent.find_next_younger_neighbor(current_node)
    #     # Find matching support and sign pattern
    #     while not current_node.shares_support_with(self) and \
    #                 len(current_node.children) > 0:
    #         current_node=current_node.oldest_child()
    #     if current_node is not None and current_node != self and \
    #             current_node.shares_support_with(self):
    #         # Found a node to merge with
    #         # Fix beta max and parents! Note that the self node should not have
    #         # children yet.
    #         print "Merging {0} with {1}".format(current_node, self)
    #         current_node.beta_min=self.beta_min
    #         for parent in self.parents:
    #             if current_node in parent.children:
    #                 parent.children.remove(self)
    #             else:
    #                 parent.replace_child(self, current_node)
    #         if len(current_node.children) > 0:
    #             new_childless_tilingelements=current_node.find_children(
    #                                             beta_min=self.beta_min,
    #                                             beta_max=self.beta_max)
    #             return 1, new_childless_tilingelements
    #         else:
    #             return 1, []
    #     else:
    #         print "Stopping merge_right operation on {0}".format(self)
    #         return 0, []

    def find_left_merge_candidate(self):
        assert len(self.children) == 0 and len(self.parents) == 1
        current_node=self
        while current_node.oldest_parent() is not None and \
                current_node.oldest_parent().oldest_child() == current_node:
            current_node=current_node.oldest_parent()
        # Reaching the root without finding younger children -> self is node of
        # the leftmost's part of the graph.
        if current_node.oldest_parent() is None:
            print "(1) Stopping left search operation on {0}".format(self)
            return None
        # Get parent
        parent=current_node.oldest_parent()
        # Get next older sibling
        current_node=parent.find_next_older_neighbor(current_node)
        # Find matching support and sign pattern
        while not current_node.shares_support_with(self) and \
                    len(current_node.children) > 0:
            current_node=current_node.youngest_child()
        if current_node is not None and current_node.shares_support_with(self):
            print "(2) Found left merge partner {0} for {1}".format(current_node,
                                                                self)
            return current_node

        else:
            print "(3) Stopping left search operation on {0}".format(self)
            return None

    def find_right_merge_candidate(self):
        assert len(self.children) == 0 and len(self.parents) == 1
        current_node=self
        while current_node.youngest_parent() is not None and \
                current_node.youngest_parent().youngest_child() == current_node:
            current_node=current_node.youngest_parent()
        # Reaching the root without finding younger children -> self is node of
        # the leftmost's part of the graph.
        if current_node.youngest_parent() is None:
            print "(1) Stopping right search operation on {0}".format(self)
            return None
        # Get parent
        parent=current_node.youngest_parent()
        # Get next older sibling
        current_node=parent.find_next_younger_neighbor(current_node)
        # Find matching support and sign pattern
        while not current_node.shares_support_with(self) and \
                    len(current_node.children) > 0:
            current_node=current_node.oldest_child()
        if current_node is not None and current_node.shares_support_with(self):
            print "(2) Found right merge partner {0} for {1}".format(current_node,
                                                                self)
            return current_node

        else:
            print "(3) Stopping right search operation on {0}".format(self)
            return None

    @staticmethod
    def merge_new_children(children):
        """ The beta ranges corresponding to the given children should be
        connected. Do not input only first and last children of a recently
        analysed node but all the children that have been found."""
        # Check: Parents of children should be empty
        # import pdb
        # pdb.set_trace()
        # for child in children:
        #     assert len(child.parents) == 0
        if len(children) == 0:
            return [], []
        children.sort(key = lambda x: x.beta_max)
        left_candidate = children[0].find_left_merge_candidate()
        right_candidate = children[-1].find_right_merge_candidate()
        uncompleted_children = []
        children_for_stack = children[1:-1] # Every element except first and last
        if len(children) == 1 and left_candidate is not None and \
            right_candidate is not None:
            # This is a special case since left and right candidate belong
            # to the same area which was previously intersected by the given
            # children[0] node. Here we have to be especially careful to restore
            # the correct relations, and the stack with which we go on
            # afterwards.

            # Assemble uncompleted children area first since we override
            # children
            if len(left_candidate.children) + len(right_candidate.children) > 0:
                if len(left_candidate) == 0:
                    uncompleted_children.append((left_candidate,
                                            left_candidate[0].beta_min,
                                            children[0].beta_max))
                elif len(right_candidate) == 0:
                    # Remark: The left candidate will be the surviving one with
                    # the correct relations, hence we need to append the left
                    # one here as well.
                    uncompleted_children.append((left_candidate,
                                            children[0].beta_min,
                                            right_candidate[0].beta_max))
                else:
                    # Remark: The left candidate will be the surviving one with
                    # the correct relations, hence we need to append the left
                    # one here as well. Case where both sourrounding nodes have
                    # already been processed.
                    uncompleted_children.append((left_candidate,
                                            children[0].beta_min,
                                            children[0].beta_max))
            print "Merging {0} with {1} and {2}".format(left_candidate,
                                                        children[0],
                                                        right_candidate)
            # Case we have searched for left and right candidates of single node
            # Case left_candidate + right_candidate + children node belong
            # to the same tiling element.
            left_candidate.beta_max = right_candidate.beta_max
            left_candidate.parents += children[0].parents + right_candidate.parents
            left_candidate.uniquefy_parents()
            left_candidate.children += right_candidate.children
            left_candidate.uniquefy_children()
            # Fix parents of right candidate by replacing the child
            for parent in right_candidate.parents:
                parent.replace_child(right_candidate, left_candidate)
                parent.sort_children()
                parent.uniquefy_children()
            # Fix parents of children[0] by replacing the child
            for parent in children[0].parents:
                parent.replace_child(children[0], left_candidate)
                parent.sort_children()
                parent.uniquefy_children()
            # Fix children of right_candidate by replacing the parent
            for child in right_candidate.children:
                child.replace_parent(right_candidate, left_candidate)
                child.sort_parents()
                parent.uniquefy_children()
        else:
            if left_candidate is not None:
                print "Merging {0} with {1}".format(left_candidate,
                                                            children[0])
                # Case left_candidate + right_candidate + children node belong
                # to the same tiling element.
                left_candidate.beta_max = children[0].beta_max
                left_candidate.parents += children[0].parents
                left_candidate.uniquefy_parents()
                # Fix parents of children[0] by replacing the child
                for parent in children[0].parents:
                    parent.replace_child(children[0], left_candidate)
                    parent.sort_children()
                    parent.uniquefy_children()
                if len(left_candidate.children) > 0:
                    # In this case the left candidate is not in the 'to-process'
                    # stack anymore, hence we need to compute the left-over
                    # children directly.
                    uncompleted_children.append((left_candidate,
                                                children[0].beta_min,
                                                children[0].beta_max))
                # Fix n_element number by decreasing it by one (since we merged)
                # the left-most node
                for child in children[1:]:
                    child.n_element -= 1
            else:
                children_for_stack.insert(0, children[0])
            if right_candidate is not None:
                print "Merging {0} with {1}".format(children[-1],
                                                    right_candidate)
                # Case left_candidate + right_candidate + children node belong
                # to the same tiling element.
                right_candidate.beta_min = children[-1].beta_min
                right_candidate.parents += children[-1].parents
                right_candidate.uniquefy_parents()
                # Fix parents of children[0] by replacing the child
                for parent in children[-1].parents:
                    parent.replace_child(children[-1], right_candidate)
                    parent.sort_children()
                    parent.uniquefy_children()
                if len(right_candidate.children) > 0:
                    # In this case the right candidate is not in the 'to-process'
                    # stack anymore, hence we need to compute the left-over
                    # children directly.
                    uncompleted_children.append((right_candidate,
                                                children[-1].beta_min,
                                                children[-1].beta_max))
            elif len(children) > 1:
                # If len(children) == 1 it is already on the stack if it
                # corresponding candidates were None
                children_for_stack.insert(len(children_for_stack), children[-1])
        return uncompleted_children, children_for_stack


    def verify_tiling(self):
        """ Testing the integrity of the tree from this node on and below. A
        tree is considered as being correct if
        (1) it has no children at all,
        (2) the children's beta_min's and beta_max's of the children yield a
            discretisation of this node's [beta_min, beta_max] range without any
            holes and they span the same range [beta_min, beta_max] in total.
        """
        ctr = 0
        n_children = len(self.children)
        if n_children == 0:
            return True
        elif self.beta_min != self.oldest_child().beta_min:
            return False
        elif self.beta_max != self.youngest_child().beta_max:
            return False

        else:
            while ctr + 1 < n_children:
                if self.children[ctr].beta_max == self.children[ctr+1].beta_min:
                    ctr = ctr + 1
                else:
                    return False
            # A tree is only correct if the same holds for all its children
            return all([child.verify_tiling() for child in self.children])
