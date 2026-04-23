import numpy as np


def generate_LS_1D(pitch, duty_cycle, cutoff_freq, x_range) :
  """
  :param pitch: Period of the binary pattern
  :param duty_cycle: Fraction of the pitch that is 'high' (0 to 1)
  :param cutoff_freq: The width of the step-function in Fourier space
  :param x_range: NumPy array of x coordinates
  """
  # 1. Create the Binary Pattern (Rectangular Wave)
  # Using a simple modulo to create a periodic binary grating
  binary_pattern = ((x_range % pitch) < (pitch * duty_cycle)).astype(float)

  # 2. Fourier Transform
  obj_fft = np.fft.fft(binary_pattern)
  freqs = np.fft.fftfreq(len(x_range), d=(x_range[1] - x_range[0]))

  # 3. Apply the Step-Function Kernel (Low-pass Filter)
  # This is the 'pupil' of the system
  kernel = (np.abs(freqs) <= cutoff_freq).astype(float)
  filtered_fft = obj_fft * kernel

  # 4. Inverse Fourier Transform to get the 'Field' (E-field)
  field = np.fft.ifft(filtered_fft)

  # 5. Square the result to get Intensity
  intensity = np.abs(field)**2

  return intensity

def generate_LS_2D(pitch, duty_cycle, cutoff_freq, x_range) :
  profile = generate_LS_1D(pitch, duty_cycle, cutoff_freq, x_range)
  repeats = x_range.shape[0]
  profile = np.repeat(profile, repeats).reshape(-1, repeats).T
  if np.isnan(profile).any() :
    raise ValueError("profile contains NaN values")
  return profile

def comp_Space_CD(pitch, duty_cycle, cutoff_freq, x_range, th):
  """
  Computes the width of the region where the intensity is below threshold 'th'.
  """
  # 1. Generate the intensity profile using your previous logic
  # (Assuming generate_coherent_profile is available)
  intensity = generate_LS_1D(pitch, duty_cycle, cutoff_freq, x_range)

  # 2. Check global conditions
  if np.all(intensity >= th):
      return 0.0  # No space: the image never dips below threshold
  if np.all(intensity < th):
      return pitch # All space: the image is entirely below threshold

  # 3. Analyze a single period
  # To get an accurate measurement, we look at the middle period
  # to avoid edge effects from the convolution
  mask = (x_range >= 0) & (x_range < pitch)
  x_period = x_range[mask]
  int_period = intensity[mask]

  # 4. Find where the intensity crosses the threshold
  # Logic: Intensity < th gives a boolean array.
  # The sum of True values multiplied by the step size gives the width.
  is_space = int_period < th

  # Calculate the sampling interval (dx)
  dx = x_range[1] - x_range[0]

  # CD is the total width of the 'space' within one period
  space_cd = np.sum(is_space) * dx

  return space_cd