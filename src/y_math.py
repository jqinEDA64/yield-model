import numpy as np
from scipy.integrate import quad
from scipy.stats import multivariate_normal
import warnings
import math


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


##################################################
# Compute the expected value of the absolute value
# of the determinant of a symmetric 2x2 matrix with
# Gaussian entries. 
##################################################

def expected_abs_det(mu, sigma, print_warning=True):
  """
  Computes E[|XZ - Y^2|] for a symmetric 2x2 matrix where [X, Y, Z] ~ N(mu, sigma).
  Incorporates scaling to handle high-variance inputs.
  """
  # --- 1. Scaling Logic ---
  # We want the 'typical' variance to be around 1.0
  # Using the trace is a robust way to get a scale factor
  scale_factor = np.sqrt(np.trace(sigma) / 3.0)

  # Scale inputs down
  mu_s = mu / scale_factor
  sigma_s = sigma / (scale_factor**2)

  Q = np.array([[0,   0,   0.5],
                [0,  -1,   0.0],
                [0.5, 0,   0.0]])

  # Calculate moments on scaled data
  mu_det_s = mu_s.T @ Q @ mu_s
  mu_det_moment_s = mu_det_s + np.trace(Q @ sigma_s)

  QS_s = Q @ sigma_s
  var_det_s = 2 * np.trace(QS_s @ QS_s) + 4 * (mu_s.T @ Q @ sigma_s @ Q @ mu_s)
  sigma_det_s = np.sqrt(max(var_det_s, 1e-16))

  snr = abs(mu_det_moment_s) / sigma_det_s
  snr_threshold = 3.0

  # --- 2. Moment Method (Scaled) ---
  if snr > snr_threshold:
    z = mu_det_moment_s / (sigma_det_s * np.sqrt(2))
    res_s = abs(mu_det_moment_s * math.erf(z) + sigma_det_s * np.sqrt(2/np.pi) * np.exp(-z**2))
    return res_s * (scale_factor**2) # Scale back up

  # --- 3. Integration Logic (Scaled) ---
  def char_func_centered_s(t):
    M = np.eye(3) - 2j * t * sigma_s @ Q
    det_M = np.linalg.det(M)
    inv_M_mu = np.linalg.solve(M, mu_s)
    exponent_full = 1j * t * (mu_s.T @ Q @ inv_M_mu)
    phi_centered = np.exp(exponent_full - 1j * t * mu_det_s) / np.sqrt(det_M)
    return phi_centered

  def integrand_s(t):
    if t < 1e-8:
      m2_s = 2 * np.trace(Q @ sigma_s @ Q @ sigma_s) + 4 * mu_s.T @ Q @ sigma_s @ Q @ mu_s
      return 0.5 * (mu_det_s**2 + m2_s)
    phi_c = char_func_centered_s(t)
    re_phi = np.cos(t * mu_det_s) * np.real(phi_c) - np.sin(t * mu_det_s) * np.imag(phi_c)
    return (1 - re_phi) / (t**2)

  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    # Integrating scaled integrand is now numerically stable
    integral_val, error_est = quad(integrand_s, 0, np.inf, limit=500)

    failed = len(w) > 0

  if failed:
    if print_warning:
      print(f"WARNING: Integration failed (SNR: {snr:.2f}). Using Monte Carlo.")
    # Perform Monte Carlo on original scale for safety
    samples = multivariate_normal.rvs(mean=mu, cov=sigma, size=100000)
    dets = samples[:, 0] * samples[:, 2] - samples[:, 1]**2
    return np.mean(np.abs(dets))

  # Scale the result back up: Result * S^2
  return (2 / np.pi) * integral_val * (scale_factor**2)


# Compute the expected value of the absolute value of the determinant
# of the Y-vector for a given image, covariance, point, and threshold.
def comp_E_AbsDet(img, cov, pt, th) :
  mu, cov_mat = getMean_Cov_Y_cond(img, cov, pt, th)
  return expected_abs_det(mu, cov_mat)


###############################################
# Compute the PDF of the X-vector, given an
# image, covariance, point, and threshold.
###############################################


def getProb_GradPhi_Phi(img, cov_img, pt, threshold, log = False) :
  mu  = getMean_X(img    , pt)
  cov = getCov_XX(cov_img, pt)
  x   = np.array([0, 0, threshold])
  logpdf = multivariate_normal.logpdf(x, mean=mu, cov=cov, allow_singular=False)
  if log :
    return logpdf
  else :
    return np.exp(logpdf)
  

#################################
# Kac-Rice integration
#################################


def getDefectDensity(cov, pt, th) :

  img     = cov.I        # Use the diffused aerial image
  img_val = img.get(pt)

  # Define boundaries for integration
  lower   = -np.inf if img_val > th else th
  upper   =  th     if img_val > th else np.inf

  # Extract the value of log(p(\phi', \phi)) where it is
  # around the highest. This makes the integrand larger
  # and relieves some stress on the floating-point capabilities
  # of the integrator.
  log_p_0 = getProb_GradPhi_Phi(img, cov, pt, th, log = True)

  # Define the integrand
  def integrand(u):
    log_p     = getProb_GradPhi_Phi(img, cov, pt, u, log = True)
    log_diff  = log_p - log_p_0
    if log_diff < -12 :  # Numerical cutoff for the integration
      return 0
    
    #E_Abs_Det = 1  # TODO jqin: Better approximation for tough computation
    E_Abs_Det = comp_E_AbsDet(img, cov, pt, u)

    return E_Abs_Det * np.exp(log_diff)

  # Perform the numerical integration
  result, _ = quad(integrand, lower, upper)

  # Restore the small prefactor
  return np.exp(log_p_0)*result