#coding: utf8
import numpy as np
from pykrylov.lls import LSMRFramework


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

def least_squares_regression(support, matrix, rhs):
    """ Method performs a least squares regression for the problem

    ||matrix[:,support] * x[support] - rhs||_2^2 -> min (1)

    The output will have the size of the matrix second dimension, although only
    values on the indices given by the support can be nonzero.

    Parameters
    -------------
    support : array, shape (n_support)
        Python list or numpy array with integer corresponding to the support on
        which the solution is allowed to be nonzero.

    matrix : array, shape (n_measurements, n_features)
        Matrix in the problem (1).

    rhs : array, shape (n_measurements)
        Measurements in the problem (1).

    Returns
    --------------
    Numpy array of shape (n_features) that is the solution to the unpenalised
    least squares regression problem given above. Is only non-zero in entries
    specified by the support.

    Remarks
    -------------
    Uses the LSMRFramework with Golub-Kahan bidiagonalization process to solve
    the least squares problem. If the solution is not unique (underdetermined)
    system, consult the docs of LSMR to see which solution is provided.
    """
    return regularised_least_squares_regression(0.0, support, matrix, rhs)

def regularised_least_squares_regression(reg_param, support, matrix, rhs):
    """ Method performs a regularised least squares regression, i.e. solves

    ||matrix[:,support]*x[support]-rhs||_2^2+beta*||x[support]||_2^2 -> min  (1)

    The output will have the size of the matrix second dimension, although only
    values at indices given by the support can be nonzero.

    Parameters
    -------------
    reg_param : Positive, real number
        Regularisation parameter in the problem (1).

    support : array, shape (n_support)
        Python list or numpy array with integer corresponding to the support on
        which the solution is allowed to be nonzero.

    matrix : array, shape (n_measurements, n_features)
        Matrix in the problem (1).

    rhs : array, shape (n_measurements)
        Measurements in the problem (1).

    Returns
    --------------
    Numpy array of shape (n_features) that is the solution to the unpenalised
    least squares regression problem given above. Is only non-zero in entries
    specified by the support.

    Remarks
    -------------
    Uses the LSMRFramework with Golub-Kahan bidiagonalization process to solve
    the least squares problem. If the solution is not unique (ie. if reg_param=0)
    system, consult the docs of LSMR to see which solution is provided.
    The implementation is based on the Golub-Kahan bidiagonalization process
    from the pykrylov package. """
    sol = np.zeros(matrix.shape[1])
    lsmr_solver = LSMRFramework(matrix[:,support])
    lsmr_solver.solve(rhs, damp = reg_param)
    sol[support] = lsmr_solver.x
    return sol

def solve_mp_fixed_support(alpha, beta, support, signum, B_beta, y_beta, A, y):
    """ Calculate (u_ab, v_ab) in a fast way on a fixed support. The related
    equations are given by:

    u_ab = (B_beta,I.T * B_beta,I)^(-1) * (B_beta,I.T * y_beta - alpha * sign(I) (1)
    v_ab = (beta + A.T * A)^(-1)(A.T*y - A.T * A * u_ab)                         (2)

    The outputs will be given by vectors of the same size as the second matrix
    dimension, although the solution u_ab is only nonzero on support.

    Parameters
    -------------
    alpha : Positive, real number
        Regularisation paramater alpha in (1).

    beta : Positive, real number
        Regularisation parameter beta in (1) and (2).

    support : array, shape (n_support)
        Python list or numpy array with integer corresponding to the support on
        which the solution u_ab is allowed to be nonzero.

    signum : array, shape (n_support)
        Numpy array with -1.0 and 1.0 prescribing the sign pattern of the
        solution u_ab on I.

    B_beta : array, shape (n_measurements, n_features)
        Parameterised measurement matrix B_beta from (1).

    y_beta : array, shape (n_measurements, n_features)
        Parameterised measurements y_beta from (1).

    A : array, shape (n_measurements, n_features)
        Measurement matrix A from (2).

    y : array, shape (n_measurements)
        Measurements y from (2).

    Returns
    --------------
    Tuple of two numpy vectors of shape (n_features) corresponding to the
    solution u_ab and v_ab in (1) and (2).
    """
    matrix = B_beta[:,support].T.dot(B_beta[:,support])
    rhs = B_beta[:,support].T.dot(y_beta)
    rhs = rhs - alpha * signum
    u_ab_I = np.linalg.solve(matrix, rhs)
    rhs = y - A[:,support].dot(u_ab_I)
    v_ab = regularised_least_squares_regression(beta,
                                                support = np.arange(A.shape[1]),
                                                matrix = A, rhs = rhs)
    u_ab = np.zeros(A.shape[1])
    u_ab[support] = u_ab_I
    return u_ab, v_ab

def approximate_solve_mp_fixed_support(support, matrix, rhs):
    """ Method calculates an approximative solution (u_(a,b), v_(a,b)) for a
    known support. Instead of using alpha, the support is fixed as given
    and u is determined through least squares regression on the support:

    ||matrix[:,support] * u_I[support] - rhs||_2^2 -> min (1)

    Instead of using beta, the noise v_I is calculated as the least squares
    regression of the residual:

    ||matrix[:,support] * v_I[support] - (rhs-matrix[:,support] * u_I)||_2^2 -> min (2)

    Hence we calculate a solution (u_I, v_I) without using reg. parameters.

    Parameters
    -------------
    support : array, shape (n_support)
        Python list or numpy array with integer corresponding to the support on
        which the solution u_I is allowed to be nonzero.

    matrix : array, shape (n_measurements, n_features)
        Matrix in the problem (1) and (2).

    rhs : array, shape (n_measurements)
        Measurements in the problem (1) and (2).

    Returns
    --------------
    Tuple of two numpy vectors of shape (n_features) corresponding to the
    solution u_I and v_I in (1) and (2).
    """
    u_I = least_squares_regression(support, matrix, rhs)
    rhs = rhs - matrix[:, support].dot(u_I[support])
    v_I = least_squares_regression(support = np.arange(matrix.shape[1]),
                                   matrix, rhs)
    return u_I, v_I
