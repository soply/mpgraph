#coding: utf8
import numpy as np

def calc_B_y_beta(A, y, svdAAt_U, svdAAt_S, beta):
    """ Auxiliary function calculating the matrix B_beta and vector y_beta
    given by

            B_beta = (Id + A*A.T/beta)^(-1/2) * A,
            y_beta = (Id + A*A.T/beta)^(-1/2) * y

    as fast as possible. This is achieved by reusing the SVD of AAt that
    has been calculated in the initialisation. The modification through beta
    and the exponent is realised by manipulating the SVDs and readjusting
    the matrix. """
    tmp = svdAAt_U.dot(np.diag(np.sqrt(beta/(beta + svdAAt_S)))).dot(svdAAt_U.T)
    return tmp.dot(A), tmp.dot(y)

def calc_B_y_beta_selection(A, y, svdAAt_U, svdAAt_S, beta, I, J):
    """ Auxiliary function calculating the matrix B_beta and vector y_beta
    given by

            B_beta = (Id + A*A.T/beta)^(-1/2) * A,
            y_beta = (Id + A*A.T/beta)^(-1/2) * y

    as fast as possible. This is achieved by reusing the SVD of AAt that
    has been calculated in the initialisation. The modification through beta
    and the exponent is realised by manipulating the SVDs and readjusting
    the matrix. """
    tmp = svdAAt_U.dot(np.diag(np.sqrt(beta/(beta + svdAAt_S)))).dot(svdAAt_U.T)
    if len(J) > 0:
        AJ = tmp.dot(A[:,J])
    else:
        AJ = None
    if len(I) > 0:
        AI = tmp.dot(A[:,I])
    else:
        AI = None
    return AJ, AI, tmp.dot(y)
