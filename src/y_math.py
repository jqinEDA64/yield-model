import numpy as np


#############################################
# Schur complement
#############################################


def condition_gaussian(mu_y, mu_x, Sigma_yy, Sigma_xx, Sigma_yx, x_0):
  """
  Computes P(y | x = observation) for a joint Gaussian distribution.
  """
  # Sigma_xy is the transpose of Sigma_yx
  Sigma_xy = Sigma_yx.T

  # We solve (Sigma_xx * K = Sigma_yx) for K instead of inverting Sigma_xx
  # K corresponds to: Sigma_yx @ inv(Sigma_xx)
  K = np.linalg.solve(Sigma_xx, Sigma_xy).T

  # New Mean: mu_y + K * (x - mu_x)
  mu_cond = mu_y + K @ (x_0 - mu_x)

  # New Covariance: Sigma_yy - K * Sigma_xy
  Sigma_cond = Sigma_yy - K @ Sigma_xy

  return mu_cond, Sigma_cond

