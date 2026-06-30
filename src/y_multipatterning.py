"""
y_multipatterning.py
"""

import numpy as np
from scipy.stats import multivariate_normal
from scipy.integrate import quad

from src import y_math


###############################################################################
# STEP 1: Joint statistics of (grad I1, I1, grad I2, I2) at a point
###############################################################################

def getMean_Multi(cov1, cov2, pt):
    """
    Returns the mean of v = (dI1/dx, dI1/dy, I1, dI2/dx, dI2/dy, I2) at pt,
    as a numpy array of shape (6,).
    """
    mu1 = y_math.getMean_X(cov1.I, pt)
    mu2 = y_math.getMean_X(cov2.I, pt)
    return np.concatenate([mu1, mu2])


def getCov_Multi(cov1, cov2, pt):
    """
    Returns the 6x6 covariance matrix of v = (dI1/dx, dI1/dy, I1, dI2/dx, dI2/dy, I2).
    Assumes I1, I2 independent, so off-diagonal 3x3 blocks are zero.
    """
    Sigma1 = y_math.getCov_XX(cov1, pt)
    Sigma2 = y_math.getCov_XX(cov2, pt)

    Sigma = np.zeros((6, 6))
    Sigma[0:3, 0:3] = Sigma1
    Sigma[3:6, 3:6] = Sigma2
    return Sigma


IDX = dict(dx1=0, dy1=1, I1=2, dx2=3, dy2=4, I2=5)


###############################################################################
# STEP 2: Encode the tangency constraint, evaluate density at it (Eq. 34-35)
###############################################################################

def build_constraint(lam, th1, th2, u):
    """
    Tangency condition (Eq. 34):
        I1 = th1 + u
        I2 = th2 + u
        dI1/dx = lam * dI2/dx
        dI1/dy = lam * dI2/dy

    Encoded as A @ v = x0.
    """
    A = np.zeros((4, 6))
    A[0, IDX['I1']] = 1.0
    A[1, IDX['I2']] = 1.0
    A[2, IDX['dx1']] = 1.0
    A[2, IDX['dx2']] = -lam
    A[3, IDX['dy1']] = 1.0
    A[3, IDX['dy2']] = -lam

    x0 = np.array([th1 + u, th2 + u, 0.0, 0.0])
    return A, x0


def density_at_tangency(mu, Sigma, lam, th1, th2, u):
    """
    Computes p_Psi(u, u, 0, 0): density of the tangency condition at
    given (u, lambda). A@v is Gaussian with mean A@mu, cov A@Sigma@A.T.
    """
    A, x0 = build_constraint(lam, th1, th2, u)

    mu_C = A @ mu
    Sigma_C = A @ Sigma @ A.T

    return multivariate_normal.pdf(x0, mean=mu_C, cov=Sigma_C, allow_singular=True)


###############################################################################
# STEP 3: Integrate over u and lambda to get the scalar defect density D_MP(x)
###############################################################################

def getDefectDensity_MultiPatterning(cov1, cov2, pt, th1, th2,
                                             lam_bounds=(-10, 10)):
    """
    Computes the SIMPLIFIED D_MP(pt), Eq. (36) with the geometric factor
    E(sqrt(det(J_Psi J_Psi^T))) replaced by the placeholder value 1
    (mirrors y_math.ABS_DET_FLAG == 0 for single-patterning).
    """
    mu = getMean_Multi(cov1, cov2, pt)
    Sigma = getCov_Multi(cov1, cov2, pt)

    I0_1 = cov1.I.get(pt)
    I0_2 = cov2.I.get(pt)

    if (I0_1 < th1) and (I0_2 < th2):
        u_lower, u_upper = 0.0, np.inf
    elif (I0_1 > th1) and (I0_2 > th2):
        u_lower, u_upper = -np.inf, 0.0
    else:
        return 0.0

    def integrand_u(u, lam):
        return density_at_tangency(mu, Sigma, lam, th1, th2, u)

    def integrand_lam(lam):
        val, _ = quad(integrand_u, u_lower, u_upper, args=(lam,), limit=100)
        return val

    result, _ = quad(integrand_lam, lam_bounds[0], lam_bounds[1], limit=100)

    return result / np.pi
