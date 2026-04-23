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


################################################
# Statistics for correlated random fields of an
# image and covariance.
#
# X = (\phi_x, \phi_y, \phi)^T
# Y = (\phi_xx, \phi_yy, \phi_xy)^T
################################################

def getMean_X(img, pt) :
  out = np.zeros((3,))
  out[0] = img.compute_der_pt(pt, dir="x")
  out[1] = img.compute_der_pt(pt, dir="y")
  out[2] = img.get(pt)
  return out

def getCov_XX(cov, pt) :
  out = np.zeros((3, 3))
  for i in range(3) :
    for j in range(0, i+1) :

      # Compute the appropriate order of derivative
      orders = [0, 0, 0, 0]
      if i == 0 :
        orders[0] += 1
      if i == 1 :
        orders[1] += 1
      if j == 0 :
        orders[2] += 1
      if j == 1 :
        orders[3] += 1

      num = cov.derivative(pt, orders=tuple(orders))
      out[i, j] = num
      out[j, i] = num

  return out

def getMean_Y(img, pt) :
  out = np.zeros((3,))
  out[0] = img.compute_der_pt(pt, dir="xx")
  out[1] = img.compute_der_pt(pt, dir="yy")
  out[2] = img.compute_der_pt(pt, dir="xy")
  return out

def getCov_YY(cov, pt) :
  out = np.zeros((3, 3))
  for i in range(3) :
    for j in range(0, i+1) :

      # Compute the appropriate order of derivative
      orders = [0, 0, 0, 0]
      if i == 0 :
        orders[0] += 2
      if i == 1 :
        orders[1] += 2
      if i == 2 :
        orders[0] += 1
        orders[1] += 1
      if j == 0 :
        orders[2] += 2
      if j == 1 :
        orders[3] += 2
      if j == 2 :
        orders[2] += 1
        orders[3] += 1

      num = cov.derivative(pt, orders=tuple(orders))
      out[i, j] = num
      out[j, i] = num

  return out

def getCov_YX(cov, pt) :
  out = np.zeros((3, 3))
  for i in range(3) :
    for j in range(3) :

      # Compute the appropriate order of derivative
      # Here, i corresponds to Y and j corresponds to X.
      orders = [0, 0, 0, 0]

      # This is like getCov_YY
      if i == 0 :
        orders[0] += 2
      if i == 1 :
        orders[1] += 2
      if i == 2 :
        orders[0] += 1
        orders[1] += 1

      # This is like getCov_XX
      if j == 0 :
        orders[2] += 1
      if j == 1 :
        orders[3] += 1

      num = cov.derivative(pt, orders=tuple(orders))
      out[i, j] = num

  return out

def getMean_Cov_Y_cond(img, cov, pt, th) :
  mu_x = getMean_X(img, pt)
  mu_y = getMean_Y(img, pt)
  Sigma_xx = getCov_XX(cov, pt)
  Sigma_yy = getCov_YY(cov, pt)
  Sigma_yx = getCov_YX(cov, pt)

  # Value to condition on
  x_0 = np.array([0, 0, th])

  # Compute the conditional distribution
  return condition_gaussian(mu_y, mu_x, Sigma_yy, Sigma_xx, Sigma_yx, x_0)