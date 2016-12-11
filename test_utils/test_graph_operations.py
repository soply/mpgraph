#coding: utf8
""" Test cases to test the tiling element graph operations of adding and merging
    nodes and reconstructing a proper graph as soon as a new tiling element
    is been added. The test cases should consist of a generator that generates
    regions from a list as they would be found from the create_children_LARS
    respectively find_children function. This is because we do solely want to
    test the outcome of the graph operations, not if the find_children function
    operates correctly.
"""

import numpy as np

base_region = [(0.0, 0.0), (0.0, 1.0), [], []]

def create_test_case_iterator(testcase):
    if testcase == "TC1":
        # Test case 1 regions (see notes with Markus)
        # First three regions
        regions = iter([
            [base_region],
            [[(0.0, 0.0), (0.0, 0.3), [1], [1.0]],
            [(0.0, 0.3), (0.0, 0.7), [5], [1.0]],
            [(0.0, 0.7), (0.0, 1.0), [10], [1.0]]],
            # Siblings of first offspring
            [[(0.0, 0.0), (0.0, 0.15), [1, 2], [1.0, -1.0]],
            [(0.0, 0.15), (0.0, 0.3), [1, 3], [1.0, -1.0]]],
            # Siblings of second offspring
            [[(0.0, 0.3), (0.0, 0.45), [1], [1.0]], # <- Merge operation
            [(0.0, 0.45), (0.0, 0.6), [5, 6], [1.0, -1.0]],
            [(0.0, 0.6), (0.0, 0.7), [10], [1.0]]], # <- Merge operation
            # From merge operation:
            [[(0.0, 0.3), (0.0, 0.6), [1, 3], [1.0, -1.0]]], # <- Merge operation
            # Siblings from third offpsring
            [[(0.0, 0.6), (0.0, 1.0), [10, 11], [1.0, -1.0]]]
        ])
    else:
        raise NotImplementedError("TC {0} not implemented.".format(testcase))
    return regions

def create_dummy_variables():
    A = np.ones((5,5))
    y = np.ones(5)
    U, S, V = np.linalg.svd(A.dot(A.T))
    return A, y, U, S
