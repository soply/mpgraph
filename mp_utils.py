#coding: utf8
import numpy as np

def calc_B_y_beta(A, y, svdAAt_U, svdAAt_S, beta):
    """ Auxiliary function calculating the matrix B_beta and vector y_beta
    given by

            B_beta = (Id + A*A.T/beta)^(-1/2) * A,
            y_beta = (Id + A*A.T/beta)^(-1/2) * y.

    For speed-up, we rely on reusing the SVD of AAt that has been calculated in
    the initialisation. The modification through beta and the exponent is
    realised by manipulating the SVDs and readjusting the matrix.

    Parameters
    ----------
    A : array, shape (n_measurements, n_features)
        The sensing matrix A in the problem 1/2 || Ax - y ||_2^2 + alpha ||x||_1.

    y : array, shape (n_measurements)
        The vector of measurements

    svdAAt_U : array, shape (n_measurements, n_measurements)
        Matrix U of the singular value decomposition of A*A^T.

    svdAAt_S : array, shape (n_measurements)
        Array S with singular values of singular value decomposition of A*A^T.

    beta : Positive real number
        Parameter beta

    Returns
    ----------
    Tuple (B_beta, y_beta) calculated from input data.
    """
    tmp = svdAAt_U.dot(np.diag(np.sqrt(beta/(beta + svdAAt_S)))).dot(svdAAt_U.T)
    return tmp.dot(A), tmp.dot(y)

def calc_B_y_beta_selection(A, y, svdAAt_U, svdAAt_S, beta, I, J):
    """ Auxiliary function calculating the matrices B_beta[:,I] and B_beta[:,J]
    for given index set's I and J by

            B_beta[:,I]= (Id + A*A.T/beta)^(-1/2) * A[:,I],
            B_beta[:,J]= (Id + A*A.T/beta)^(-1/2) * A[:,J],
            y_beta = (Id + A*A.T/beta)^(-1/2) * y,

    hence B_beta[:,I] and B_beta[:,J] are the restriction to subsets of columns.
    For speed-up, we rely on reusing the SVD of AAt that has been calculated in
    the initialisation. The modification through beta and the exponent is
    realised by manipulating the SVDs and readjusting the matrix.

    Parameters
    ----------
    A : array, shape (n_measurements, n_features)
        The sensing matrix A in the problem 1/2 || Ax - y ||_2^2 + alpha ||x||_1.

    y : array, shape (n_measurements)
        The vector of measurements

    svdAAt_U : array, shape (n_measurements, n_measurements)
        Matrix U of the singular value decomposition of A*A^T.

    svdAAt_S : array, shape (n_measurements)
        Array S with singular values of singular value decomposition of A*A^T.

    I : array of integers, shape (n_support)
        Python list or numpy array with integer corresponding to the index set I.

    J : array of integers, shape (n_J)
        Python list or numpy array with integer corresponding to the index set J.

    beta : Positive real number
        Parameter beta

    Returns
    ----------
    Tuple (B_beta[:,I], B_beta[:,J], y_beta) calculated from input data. If one
    of the index sets I or J is empty, None is returned instead of the
    respective matrix.
    """
    tmp = svdAAt_U.dot(np.diag(np.sqrt(beta/(beta + svdAAt_S)))).dot(svdAAt_U.T)
    if len(J) > 0:
        B_betaJ = tmp.dot(A[:,J])
    else:
        B_betaJ = None
    if len(I) > 0:
        B_betaI = tmp.dot(A[:,I])
    else:
        B_betaI = None
    return B_betaJ, B_betaI, tmp.dot(y)
