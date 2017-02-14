# coding: utf8
import operator

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from create_children.lasso_path_utils import calc_all_cand
from mp_utils import calc_B_y_beta

# Settings for plotting
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
matplotlib.rc('font', **font)


def verify_tiling(tilingelement):
    """ Testing the integrity of the tree from this node on and below. A
    tree is considered valid if itself is valid and all its children are
    valid. A node is valid if the exists a distinct(!) child for all betas
    from which the node under consideration can be reached; or a node is
    valid if it has not a single node.

    Parameters
    -----------
    tilingelement : object of class TilingElement
        Root element of the (sub)-tree that shall be verified.

    Returns
    ------------
    True, if the (sub)-tree of this node is valid; false otherwise.
    """
    elements = tilingelement.bds_order()
    for element, layer in elements.iteritems():
        n_children = len(element.children)
        if n_children == 0:
            continue
        if element.children[0][1] != element.beta_min:
            print """Verification failed:
                     element.children[0][1]!=element.beta_min: {0}, {1}""".format(
                element.children[0], element)
            return False
        ctr = 1
        while ctr < n_children:
            if element.children[ctr - 1][2] != element.children[ctr][1]:
                print """Verification failed:
                         element.children[ctr-1][2]!=element.children[ctr][1]: {0}, {1}""".format(
                    element.children[ctr - 1], element.children[ctr])
                return False
            ctr = ctr + 1
        if element.children[-1][2] != element.beta_max:
            print """Verification failed:
                     element.children[-1][2]!=element.beta_max: {0}, {1}""".format(
                element.children[-1], element)
            return False
    print "Verification passed"
    return True

def plot_tiling(tilingelement, n_disc=3):
    """ Plot the support tiling outgoing from the given node as a root. We
    use bds_order and the distance from the root_node to define some notion
    of a layered structure. """
    elements = tilingelement.bds_order()
    max_layer = max(elements.iteritems(), key=operator.itemgetter(1))[1]
    # Calculate elements in inverse relation ship, ie. layer to elements
    # instead of elements to layer (we will omit zeroth layer)
    reordered_dict = []
    for i in range(1, max_layer + 1):
        reordered_dict.append([e for e in elements.keys() if
                               elements[e] == i])
        reordered_dict[-1].sort(key=lambda x: x.beta_min)
    # Make an adaptive beta discretisation that guarantees at least n_disc
    # points to represent the boundaries of each tiling element.
    beta_disc_coarse = []
    for element, layers in elements.iteritems():
        beta_disc_coarse.append(element.beta_min)
        beta_disc_coarse.append(element.beta_max)
    beta_disc_coarse = np.unique(np.array(beta_disc_coarse))
    beta_disc = np.zeros((len(beta_disc_coarse) - 1) * n_disc)
    for i in range(len(beta_disc_coarse) - 1):
        beta_disc[i * n_disc:(i + 1) * n_disc] = \
            np.linspace(beta_disc_coarse[i], beta_disc_coarse[i + 1],
                        n_disc)

    points = [np.zeros((beta_disc.shape[0], 2)) for _ in range(max_layer)]
    colors = [[] for _ in range(max_layer)]
    colorlist = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for j, beta in enumerate(beta_disc):
        B_beta, y_beta = calc_B_y_beta(tilingelement.A,
                                       tilingelement.y,
                                       tilingelement.svdAAt_U,
                                       tilingelement.svdAAt_S,
                                       beta)
        for i, layer in enumerate(reordered_dict):
            idx, element = [(idx, e) for (idx, e) in enumerate(layer)
                            if e.beta_min <= beta and e.beta_max >= beta][0]
            prev_element = [pe[0] for pe in element.parents if pe[1] <= beta
                            and beta <= pe[2]][0]
            J = np.setxor1d(element.support, prev_element.support)
            alpha, sign = calc_all_cand(B_beta, y_beta,
                                        prev_element.support,
                                        prev_element.sign_pattern)
            points[i][j, 0] = beta
            points[i][j, 1] = alpha[J]
            colors[i].append(colorlist[(idx + 2 * i) % 6])
    plt.figure()
    ax = plt.gca()
    plt.title(r'Support tiling $u_{\beta, \alpha}$')
    for i in range(max_layer):
        for j in range(len(points[i]) - 1):
            plt.semilogx(points[i][j:j + 2, 0], points[i][j:j + 2, 1],
                     color = 'k', linewidth=2.0, alpha=0.5)
                     # or chose c='colors[i][j + 1]' if line colors should match
                     # the face color of the area
            if i == 0:
                ax.fill_between(points[i][j:j + 2, 0],
                                points[i][j:j + 2, 1],
                                np.max(points[0][:, 1]),
                                facecolor=colorlist[-1],
                                alpha=0.5, linewidth=0.0)
            elif i < max_layer:
                ax.fill_between(points[i][j:j + 2, 0],
                                points[i][j:j + 2, 1],
                                points[i - 1][j:j + 2, 1],
                                facecolor=colors[i - 1][j + 1],
                                alpha=0.5, linewidth=0.0)
                # To cover the last areas that close with the horizontal axis,
                # use the following code (may provide misleading information
                # though if wrongly interpreted):
                # if i == max_layer - 1:
                #     ax.fill_between(points[i][j:j + 2, 0], 0,
                #                     points[i][j:j + 2, 1],
                #                     facecolor=colors[i][j + 1],
                #                     alpha=0.5, linewidth=0.0)
    plt.ylim(np.min(points[-1][:,1]), np.max(points[1][:,1]))
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$\alpha$')
    plt.show()

def plot_tiling_graph(tilingelement, y_mode='layered'):
    vertices = tilingelement.bds_order()
    plt.figure()
    for element, layer in vertices.iteritems():
        # Skip root element
        if layer == 0:
            continue
        # Draw nodes
        if y_mode == 'layered':
            ycoord = -layer
        else:
            ycoord = 0.5 * (element.alpha_min + element.alpha_max)
        plt.scatter(0.5 * (element.beta_min + element.beta_max), ycoord,
                    s=50.0)
        # Draw edges
        for child in element.children:
            child_entry = [item for item in list(vertices.iteritems())
                           if item[0] == child[0]]
            xstart = 0.5 * (element.beta_min + element.beta_max)
            xend = 0.5 * (child[0].beta_min + child[0].beta_max) - \
                0.5 * (element.beta_min + element.beta_max)
            if y_mode == 'layered':
                ystart = -layer
                yend = -child_entry[0][1] + layer
            else:
                ystart = 0.5 * (element.alpha_min + element.alpha_max)
                yend = 0.5 * (child[0].alpha_min + child[0].alpha_max) - \
                    0.5 * (element.alpha_min + element.alpha_max)
            plt.arrow(xstart, ystart, xend, yend, head_width=0.04,
                      head_length=0.075, fc="k", ec="k")
            if y_mode == 'layered':
                plt.annotate("({0:.2f},{1:.2f})".format(child[1], child[2]),
                             xy=(0.5 * (2 * xstart + xend),
                                 0.5 * (2 * ystart + yend)),
                             xytext=(0.5 * (2 * xstart + xend),
                                     0.5 * (2 * ystart + yend)))
                plt.ylabel('Lasso-path steps from root node (*-1)')
            else:
                plt.ylabel(r'$\alpha$')

    plt.xlabel(r'$\beta$')
    plt.title('Support tiling graph (mode: {0})'.format(y_mode))
    plt.show()
