import y_basics
import y_math

import numpy as np
from scipy.interpolate import CubicSpline


###############################################
# Gaussian correlation function
###############################################


class GaussianCovariance(y_basics.Covariance):
  def __init__(self, img, sigma, scale = 1):
    g = y_basics.getGaussian(sigma, img.pixel_size)
    super().__init__(img, g, scale)


###############################################
# Power-law correlation function
###############################################

# Simplest power Spectral Density:
#     S(f) = 1 / (1 + (|f| * L)^(2H+1))
def psd_function(f_arr, xi_val, H_val):
  alpha_val = 2 * H_val + 1
  return 1.0 / (1 + (np.abs(f_arr) * xi_val)**alpha_val)

# Same PSD function, but filter very high
# frequencies with a Gaussian
def psd_function_filtered(f_arr, xi_val, H_val, L_smooth):
  out = psd_function(f_arr, xi_val, H_val)
  return out*np.exp(-(f_arr*L_smooth)**2)

# Compute spatial correlation function C(x) from PSD using Wiener-Khinchin theorem.
# Uses rfft/irfft for real, even PSD.
def compAutoCorr(xi_val, H_val, L_smooth = 0, N_fft=2**16, f_max_factor=50):

  # Maximum frequency to sample
  f_max = f_max_factor / xi_val

  # For rfft: positive frequencies from 0 to f_max
  # rfft of N real points gives N/2+1 complex frequencies
  n_freq = N_fft // 2 + 1
  df = f_max / (N_fft // 2)  # Frequency spacing

  # Create frequency array for positive frequencies
  f_vec = np.arange(n_freq) * df

  # Evaluate PSD at positive frequencies only
  S_f = psd_function_filtered(f_vec, xi_val, H_val, L_smooth)

  # Compute inverse Fourier transform using irfft
  # irfft automatically assumes the negative frequencies are mirror of positive
  # The factor of 2 comes from integrating both positive and negative frequencies
  C_x_raw = np.fft.irfft(S_f, n=N_fft) * 2 * f_max

  # Real-space sampling: x ranges from 0 to (N_fft-1)*dx
  dx = 1 / (2 * f_max)

  # Create x-axis centered around zero
  x_axis = np.arange(N_fft) * dx
  x_axis_centered = x_axis - N_fft * dx / 2
  C_x_centered    = np.fft.fftshift(C_x_raw)

  #plt.plot(x_axis_centered, C_x_centered)
  #plt.show()

  return x_axis_centered, C_x_centered

# TODO jqin: check if derivatives here are correct near x = 0.
class PhotonCovariance(y_basics.Covariance):
  def __init__(self, Image, L_corr, H_corr, L_smooth, scale):
    super().__init__(Image)
    self.L_corr = L_corr
    self.H_corr = H_corr
    self.L_smooth = L_smooth
    self.scale  = scale

    self.x_vals, self.c_vals = self.compAutoCorr()
    self.interp_func = CubicSpline(self.x_vals, self.c_vals, bc_type='clamped')

  def compAutoCorr(self) :
    # Compute the correlation function
    # based on the correlation length
    # and exponent
    x_vals = np.linspace(0, 10*self.L_corr, 1000)
    c_vals = np.zeros(x_vals.shape)

    x_vals, c_vals = compAutoCorr(self.L_corr, self.H_corr, self.L_smooth)

    # Keep only positive values of x
    # and values of x less than 10*correlation length.
    isUpperBound = (x_vals >= 0) & (x_vals < 10*self.L_corr)
    x_vals = x_vals[isUpperBound]
    c_vals = c_vals[isUpperBound]

    c_vals /= np.max(c_vals) # Normalize so C(x = 0) = 1

    # Correlation should never be negative
    # so zero any negative values
    c_vals[c_vals < 0] = 0

    return x_vals, c_vals

  def get(self, pt1, pt2) :
    i1 = self.img.get(pt1) + 0.01  # TODO jqin: formalize into FLARE
    i2 = self.img.get(pt2) + 0.01
    size = self.scale*np.sqrt(i1*i2)
    r    = np.linalg.norm([pt1.x-pt2.x, pt1.y-pt2.y])
    #corr = np.interp(r, self.x_vals, self.c_vals)
    corr = self.interp_func(r)
    return size*corr