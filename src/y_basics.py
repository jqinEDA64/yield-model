import numpy as np
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator, CubicSpline
import matplotlib.pyplot as plt
from dataclasses import dataclass
plt.rcParams['figure.dpi'] = 150

# Convenient function which converts float to string with given precision.
# Use scientific notation (isSci == True) if printing numbers which could be
# very large or small.
def flToStr(val, precision=2, isSci=False):
  if isSci:
    return f"{val:.{precision}e}"
  else:
    return f"{val:.{precision}f}"

# Useful for FFT sizes
def next_power_of_2(n):
    res = 1
    while res < n:
        res *= 2
    return res

################################################################################
# Gaussian distribution (commonly used for kernels and noise models).
################################################################################


# Gaussian kernel
def getGaussian(sigma, dx) :
  # 1. Create a coordinate grid for the kernel
  # We go out to 4-5 sigma to capture the tail
  limit = int(5 * sigma / dx)
  x = np.arange(-limit, limit + 1) * dx
  y = np.arange(-limit, limit + 1) * dx
  xx, yy = np.meshgrid(x, y)

  # 2. Define the 2D Gaussian kernel g
  g = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
  g /= np.sum(g) # Normalize for acid conservation

  return g


################################################################################
# Data classes: Point, VectorField, Image, and Covariance.
################################################################################


# Stores the coordinates of a point (x, y).
@dataclass
class Point:
    x: float
    y: float


# Vector field (used for gradients, for example)
class VectorField:
  def __init__(self, x_component, y_component):
    self.u = x_component
    self.v = y_component

  def get(self, pt, component='both'):
    """
    Samples the vector field at (x, y).
    :param component: 'x', 'y', or 'both'
    """

    val_x = self.u.get(pt)
    val_y = self.v.get(pt)

    if component == 'x':
      return val_x
    elif component == 'y':
      return val_y
    elif component == "both" :
      return (val_x, val_y)
    else :
      raise ValueError(f"Invalid component: {component}")

  def plot(self, title = ""):
    """
    Visualizes the vector field using a quiver plot.
    :param stride: Increase to skip pixels (useful for high-res images).
    """
    # Create a meshgrid of the world coordinates
    X, Y = np.meshgrid(self.u.x_coords, self.u.y_coords)
    U = self.u.data
    V = self.v.data

    plt.figure(figsize=(8, 6))
    plt.quiver(X, Y, U, V, color='teal')
    plt.xlabel("$x$ [nm]")
    plt.ylabel("$y$ [nm]")
    plt.title(title)
    plt.axis('equal')
    plt.show()

  def plot_components(self, cmap='jet', title = ""):
    """
    Plots the U (X) and V (Y) components side-by-side as heatmaps.
    :param cmap: Colormap.
    """
    extent = [
        self.u.x_coords[0], self.u.x_coords[-1],
        self.u.y_coords[0], self.u.y_coords[-1]
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Plot X component (U)
    im1 = axes[0].imshow(self.u.data, origin='lower', extent=extent, cmap=cmap)
    axes[0].set_title("Re")
    axes[0].set_xlabel("$x$ [nm]")
    axes[0].set_ylabel("$y$ [nm]")
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Plot Y component (V)
    im2 = axes[1].imshow(self.v.data, origin='lower', extent=extent, cmap=cmap)
    axes[1].set_title("Im")
    axes[1].set_xlabel("$x$ [nm]")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.title(title)
    plt.show()


# Image field (used for scalar fields)
class Image:
  def __init__(self, data, ll_x, ll_y, pixel_size):
    """
    :param data: 2D NumPy array (the image pixels)
    :param ll_x: X-coordinate of the center of the lower-left pixel
    :param ll_y: Y-coordinate of the center of the lower-left pixel
    :param pixel_size: The width/height of a single pixel
    """
    self.data = np.asanyarray(data)
    self.ll_x = ll_x
    self.ll_y = ll_y
    self.pixel_size = pixel_size

    # Derived attributes
    self.height, self.width = self.data.shape

    # Define the coordinate axes for the pixel centers
    # We use 'ij' indexing: rows correspond to Y and columns to X
    self.x_coords = ll_x + np.arange(self.width) * pixel_size
    self.y_coords = ll_y + np.arange(self.height) * pixel_size

    # Initialize the interpolator (cubic)
    # Note: RegularGridInterpolator expects coords in (y, x) order for (row, col) data
    self._interpolator = RegularGridInterpolator(
        (self.y_coords, self.x_coords),
        self.data,
        method='cubic',
        bounds_error=False,
        fill_value=None   # Extrapolates or returns NaN based on preference
    )

    self.data_FFT = None  # Placeholder for FFT-based convolution if needed

  def get(self, pt):
    """
    Computes the value at a specific (x, y) coordinate using cubic interpolation.
    """
    if pt.x < self.x_coords[0] or pt.x >= self.x_coords[-1] or \
       pt.y < self.y_coords[0] or pt.y >= self.y_coords[-1]:
       print(pt)
       print(self.x_coords[0])
       print(self.x_coords[-1])
       print(self.y_coords[0])
       print(self.y_coords[-1])
       raise ValueError("Point is outside the image boundaries: ")

    # Interpolator takes a point as [[y, x]]
    return self._interpolator([[pt.y, pt.x]])[0]

  def getFFTData(self):
    """
    Computes the 2D FFT of the image data for convolution purposes.
    Caches the result to avoid redundant calculations.
    """
    if self.data_FFT is None:
        self.data_FFT = np.fft.rfft2(self.data)
    return self.data_FFT

  def compute_der_pt(self, pt, dir = ""):
    """
    Computes the first or second derivative of the image at a specific (x, y) coordinate.
    """
    d  = self.pixel_size*0.05

    if   dir == "x"  :
      return (self.get(Point(pt.x+d, pt.y)) - self.get(Point(pt.x-d, pt.y)))/(2*d)
    elif dir == "y"  :
      return (self.get(Point(pt.x, pt.y+d)) - self.get(Point(pt.x, pt.y-d)))/(2*d)
    elif dir == "xx" :
      return (self.compute_der_pt(Point(pt.x+d, pt.y), dir="x") - \
              self.compute_der_pt(Point(pt.x-d, pt.y), dir="x"))/(2*d)
    elif dir == "yy" :
      return (self.compute_der_pt(Point(pt.x, pt.y+d), dir="y") - \
              self.compute_der_pt(Point(pt.x, pt.y-d), dir="y"))/(2*d)
    elif dir == "xy" :
      return (self.compute_der_pt(Point(pt.x, pt.y+d), dir="x") - \
              self.compute_der_pt(Point(pt.x, pt.y-d), dir="x"))/(2*d)
    elif dir == "yx" :
      return self.compute_der_pt(pt, dir = "xy")
    else:
      raise ValueError(f"Invalid direction: {dir}")

  def compute_gradient(self):
    """
    Computes the gradient of the image data using periodic boundaries.
    Returns a VectorField object.
    """
    # np.gradient returns [grad_y, grad_x] for 2D arrays
    # edge_order=1 is standard; 'periodicity' is handled by the 'cyclic' mode
    # We divide by pixel_size to scale the gradient correctly to world units
    grad_y, grad_x = np.gradient(self.data, self.pixel_size, edge_order=1)

    # Handle periodic boundary conditions manually for the edges
    # (Note: np.gradient doesn't have a native 'periodic' flag,
    # so we recalculate the edges manually)

    # Periodic X gradient
    grad_x[:,  0] = (self.data[:, 1] - self.data[:, -1]) / (2 * self.pixel_size)
    grad_x[:, -1] = (self.data[:, 0] - self.data[:, -2]) / (2 * self.pixel_size)

    # Periodic Y gradient
    grad_y[0,  :] = (self.data[1, :] - self.data[-1, :]) / (2 * self.pixel_size)
    grad_y[-1, :] = (self.data[0, :] - self.data[-2, :]) / (2 * self.pixel_size)

    # Create new SpatialImage objects for the components
    u_img = Image(grad_x, self.ll_x, self.ll_y, self.pixel_size)
    v_img = Image(grad_y, self.ll_x, self.ll_y, self.pixel_size)

    return VectorField(u_img, v_img)

  def plot(self, cmap='jet', title = "", maxval = 0):
    """Plots the image data with proper spatial extent."""
    # 'extent' maps the data to the world coordinates in the plot
    # [left, right, bottom, top]
    extent = [
        self.x_coords[0], self.x_coords[-1],
        self.y_coords[0], self.y_coords[-1]
    ]

    plt.figure(figsize=(8, 6))
    plt.imshow(self.data, origin='lower', extent=extent, cmap=cmap)
    if maxval > 0 :
      plt.clim(0, maxval)
    plt.colorbar()
    plt.xlabel("$x$ [nm]")
    plt.ylabel("$y$ [nm]")
    if title != "":
      plt.title(title)
    plt.show()

def convolve(image, kernel):
    """
    Performs 2D convolution with exact periodic boundary conditions using FFT.
    """
    h, w = image.height, image.width
    
    # 1. Transform both the image and the kernel to the frequency domain
    #    s=(h, w) automatically pads the kernel to match the image dimensions
    kernel_fft = np.fft.rfft2(kernel, s=(h, w))
    
    # 2. Element-wise multiplication in the frequency domain
    filtered_fft = image.getFFTData() * kernel_fft
    
    # 3. Inverse transform back to spatial domain, enforcing original shape
    conv_result = np.fft.irfft2(filtered_fft, s=(h, w))
    
    # 4. Crucial: OpenCV's filter2D shifts the kernel to keep it centered.
    #    FFT convolution introduces a phase shift if the kernel origin isn't at (0,0).
    #    We use np.roll to center the result exactly like OpenCV does.
    k_h, k_w = kernel.shape
    conv_result = np.roll(conv_result, shift=(-(k_h // 2), -(k_w // 2)), axis=(0, 1))
    
    return conv_result

# The Covariance class computes the covariance between points in the image
# based on a given kernel, and derivatives of the covariance with respect
# to the coordinates of these points. The kernel is typically a Gaussian or similar smoothing function.
class Covariance:
    def __init__(self, img, k, scale=1):
        """
        Ultra-Accelerated EDA-Periodic Covariance Class.
        Uses pure spectral operators for derivatives and pre-cached real-space products.
        """
        # 1. Metadata and dimensions
        dx = img.pixel_size
        dx_2 = 1.0 / (dx * dx)
        ll_x = img.ll_x
        ll_y = img.ll_y
        h, w = img.height, img.width
        self._scale = scale

        # 2. Get the pre-cached Real-FFT data of the original image (Computed ONCE)
        img_fft = img.getFFTData()

        # 3. COMPUTE K-DERIVATIVES IN FOURIER SPACE (Exact EDA Periodicity)
        # s=(h, w) ensures the kernel matches layout canvas dimensions
        k_f = np.zeros((h, w))
        k_h, k_w = k.shape
        k_f[:k_h, :k_w] = k
        
        # Take the FFT of our full-sized kernel array
        k_fft = np.fft.fft2(k_f)
        
        # Frequency grids scaled to radians/unit
        u = np.fft.fftfreq(w, d=dx) * 2 * np.pi
        v = np.fft.fftfreq(h, d=dx) * 2 * np.pi
        UU, VV = np.meshgrid(u, v)

        # High-order derivatives computed exactly with zero boundary leakage
        kx  = np.real(np.fft.ifft2(1j * UU * k_fft))
        ky  = np.real(np.fft.ifft2(1j * VV * k_fft))
        kxx = np.real(np.fft.ifft2(-UU**2 * k_fft))
        kyy = np.real(np.fft.ifft2(-VV**2 * k_fft))
        kxy = np.real(np.fft.ifft2(-UU * VV * k_fft))

        # 4. Map the Real-Space Combinations 
        kernel_map = {
            (0, 0, 0, 0): k_f * k_f,   (1, 0, 0, 0): kx * k_f,    (0, 1, 0, 0): ky * k_f,
            (2, 0, 0, 0): kxx * k_f,   (0, 2, 0, 0): kyy * k_f,   (1, 1, 0, 0): kxy * k_f,
            (1, 0, 1, 0): kx * kx,     (1, 0, 0, 1): kx * ky,     (0, 1, 0, 1): ky * ky,
            (2, 0, 1, 0): kxx * kx,    (2, 0, 0, 1): kxx * ky,    (0, 2, 1, 0): kyy * kx,
            (0, 2, 0, 1): kyy * ky,    (1, 1, 1, 0): kxy * kx,    (1, 1, 0, 1): kxy * ky,
            (2, 0, 2, 0): kxx * kxx,   (0, 2, 2, 0): kyy * kxx,   (0, 2, 0, 2): kyy * kyy,
            (2, 0, 1, 1): kxx * kxy,   (0, 2, 1, 1): kyy * kxy,   (1, 1, 1, 1): kxy * kxy
        }

        # 5. BATCH SPECTRAL CONVOLUTIONS
        # Since the FFT places the kernel origin at (0,0), we precompute the exact spatial 
        # phase roll parameter to keep everything perfectly centered aligned with OpenCV.
        k_h, k_w = k.shape
        shift_h, shift_w = -(k_h // 2), -(k_w // 2)
        
        self.storage = {}
        for orders, prod_kernel in tqdm(kernel_map.items(), desc="Pre-computing covariance derivatives"):
            # Real FFT of the product matrix matching layout grid dimensions
            prod_kernel_fft = np.fft.rfft2(prod_kernel, s=(h, w))
            
            # Element-wise frequency filter multiplication
            conv_fft = img_fft * prod_kernel_fft
            
            # Inverse transform back to spatial domain
            conv = np.fft.irfft2(conv_fft, s=(h, w))
            conv = np.roll(conv, shift=(shift_h, shift_w), axis=(0, 1))
            
            self.storage[orders] = Image(dx_2 * scale * conv, ll_x, ll_y, dx)

        # 6. Standard Nominal Convolution I_0 = img * k
        k_padded_fft = np.fft.rfft2(k, s=(h, w))
        nominal_conv = np.fft.irfft2(img_fft * k_padded_fft, s=(h, w))
        nominal_conv = np.roll(nominal_conv, shift=(shift_h, shift_w), axis=(0, 1))
        
        self.I = Image(nominal_conv, ll_x, ll_y, dx)

    def derivative(self, p, orders=(0, 0, 0, 0)):
        """
        Returns the covariance derivative for a given point.
        Uses symmetry to map redundant order requests to pre-computed results.
        """
        o = list(orders)

        # Symmetry Mapping Logic:
        # 1. (a,b,c,d) is symmetric to (c,d,a,b). We ensure the "larger" tuple is first.
        if (o[2], o[3]) > (o[0], o[1]):
            o = [o[2], o[3], o[0], o[1]]

        # 2. Specific alias mappings (e.g., (1,0,0,0) and (0,0,1,0) are the same)
        # We convert to a tuple to use as a dictionary key.
        target = tuple(o)

        # Fallback aliases for redundant indices
        alias_map = {
            (0, 0, 1, 0): (1, 0, 0, 0),
            (0, 0, 0, 1): (0, 1, 0, 0),
            (0, 0, 1, 1): (1, 1, 0, 0),
            (0, 1, 1, 0): (1, 0, 0, 1),
            (0, 0, 2, 0): (2, 0, 0, 0),
            (0, 0, 0, 2): (0, 2, 0, 0),
            (1, 0, 2, 0): (2, 0, 1, 0),
            (0, 1, 2, 0): (2, 0, 0, 1),
            (1, 0, 0, 2): (0, 2, 1, 0),
            (0, 1, 0, 2): (0, 2, 0, 1),
            (1, 0, 1, 1): (1, 1, 1, 0),
            (0, 1, 1, 1): (1, 1, 0, 1),
            (2, 0, 0, 2): (0, 2, 2, 0),
            (1, 1, 2, 0): (2, 0, 1, 1),
            (1, 1, 0, 2): (0, 2, 1, 1),
        }

        final_key = alias_map.get(target, target)

        if final_key in self.storage:
            return self.storage[final_key].get(p)
        else:
            raise ValueError(f"Derivative order {orders} (mapped to {final_key}) not computed.")

    def sample_noise_field(self, seed=None):
        """
        VIRTUAL METHOD: Must be overridden by derived classes to supply 
        the target random distribution logic.
        """
        raise NotImplementedError(
            "sample_noise_field() must be implemented by a specific distribution subclass."
        )

    def sample_noisy_image(self, seed=None):
        """
        Synthesizes a complete stochastic realization using whatever distribution 
        the child class implements.
        """
        noise_field = self.sample_noise_field(seed=seed)
        return Image(self.I.data + noise_field.data, self.I.ll_x, self.I.ll_y, self.I.pixel_size)