# coding: utf8
""" Verification & visualisation methods for tiling. Currently implemented:
    -verify_tiling: Checks tilings validity.
    -plot_tiling: Plots reconstruction of support tiling in 2D parameter space.
    -plot_tiling_graph: Plots tiling as a graph structure. """

__author__ = "Timo Klock"

import operator
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from create_children.lasso_path_utils import calc_all_cand
from mp_utils import calc_B_y_beta

# Settings for plotting
font = {'weight' : 'bold',
        'size'   : 36}
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
    """ Plot the support tiling in the 2D parameter space (x = beta, y = alpha),
    starting from the given tiling element until the maximum depth.

    Parameters
    -------------
    tilingelement : object of class TilingElement
        Root element from which downwards the support tiling will be plottet.

    n_disc : Integer
        Number of equidistant anchor points that is used for approximating the
        boundaries of a tiling element. n_disc = 3 for example means there are
        in total 3 + 2 points used in each tiling element to interpolate the
        boundary. Increasing this increases the interpolation accuracy but
        also quickly increases the computational effort. n_disc > 1 only works
        for tilings up to a small sparsity level.
    """
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
    fig, ax = plt.subplots(figsize = (16,11))
    ax.set_title(r'Support tiling $u_{\beta, \alpha}$')
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
    plt.xlim(np.min(points[-1][:,0]), np.max(points[1][:,0]))
    plt.ylim(np.min(points[-1][:,1]), np.max(points[1][:,1]))
    ax.set_xticks([1, 10], [str(1), str(10)])
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'$\alpha$')
    plt.show()

def plot_tiling_graph(tilingelement, y_mode='layered'):
    """ Plot the support tiling as a graph. Two modes are possible: in 'layered'
    the alpha parameter is replaced by the number of Lasso-path steps from the
    root node (this is usually the most preferable mode providing the best
    visual quality). If y_mode differs from layered, approximated
    coordinate midpoints are used to place the nodes in a 2D parameter space.
    This is usually not advisiable since the midpoints are clustered together
    for close to zero, if larger solutions to larger sparsity levels are
    calculated.

    Parameters
    -------------
    tilingelement : object of class TilingElement
        Root element from which downwards the support tiling will be plottet.

    y_mode : python string
        y_mode == 'layered': Number of Lasso-path steps from root node will be
                             used as y-axis entry.
        y_mode != 'layered': Approximated tiling centers will be used
                             (1/2*(beta_min+beta_max), 1/2*(alpha_min+alpha_max)).
    """
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
