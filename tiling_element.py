from create_children_LARS import (create_children_LARS, post_process_children)


class TilingElement(object):

    def __init__(self, beta_min, beta_max, support, sign_pattern, parents,
                 A, y, svdAAt_U, svdAAt_S, options=None):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.support = support
        self.sign_pattern = sign_pattern
        self.parents = parents
        self.children = []
        self.mode = mode
        # References to global data matrices and data
        self.A = A
        self.y = y
        self.svdAAt_U = svdAAt_U
        self.svdAAt_S = svdAAt_S

    def create_children(self):
        if self.options["mode"] == "LARS":
            additional_indices, boundary_parameters, used_signs = \
                create_children_LARS(self.support, self.signum, self.beta_min,
                                     self.beta_max,
                                     self.options["env_minimiser"],
                                     self.svdAAt_U, self.svdAAt_S, self.A,
                                     self.y)
            children = post_process_children(additional_indices,
                                             boundary_parameters,
                                             used_signs, self.support,
                                             self.old_sign)
        else:
            raise NotImplementedError("Mode not implemented.")
        for child in children:
            self.children.append(TilingElement(
                child[0][1], child[1][1], child[2], child[3], [self],
                self.A, self.y, self.svdAAt_U, self.svdAAt_S, self.options)
        self.children[-1].merge_left() # Leftmost child
        self.children[0].merge_right() # Rightmost child

    def merge_right(self):
        pass

    def merge_left(self):
        pass
