import cv2
import numpy as np
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
        fill_value=None # Extrapolates or returns NaN based on preference
    )

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
  Performs 2D convolution with periodic (circular) boundary conditions.
  """
  k_h, k_w = kernel.shape
  # Pad by half the kernel size on all sides using 'wrap'
  pad_h, pad_w = k_h // 2, k_w // 2
  padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='wrap')

  # Perform convolution on the padded image
  # We use BORDER_CONSTANT here because the padding already handles the wrap
  conv = cv2.filter2D(padded, -1, kernel, borderType=cv2.BORDER_CONSTANT)

  # Slice back to the original image dimensions
  return conv[pad_h:-pad_h, pad_w:-pad_w]


# The Covariance class computes the covariance between points in the image
# based on a given kernel, and derivatives of the covariance with respect
# to the coordinates of these points. The kernel is typically a Gaussian or similar smoothing function.
class Covariance:
    def __init__(self, img, k, scale=1):
        """
        Accelerated Covariance class.
        Pre-calculates only unique kernel products and batches convolutions.
        """
        # Metadata about original image
        dx = img.pixel_size
        dx2 = dx * dx
        dx_2 = 1.0 / dx2
        ll_x = img.ll_x
        ll_y = img.ll_y

        # 1. Compute Kernel Derivatives (First and Second Order)
        kx  = np.gradient(k, dx, axis=1)
        ky  = np.gradient(k, dx, axis=0)
        kxx = np.gradient(kx, dx, axis=1)
        kyy = np.gradient(ky, dx, axis=0)
        kxy = np.gradient(kx, dx, axis=0)

        # 2. Define the Unique Product Kernels
        # We map the orders (a, b, c, d) to the product of k_ab and k_cd.
        # This list covers every unique combination required by your derivative() method.
        kernel_map = {
            (0, 0, 0, 0): k * k,
            (1, 0, 0, 0): kx * k,
            (0, 1, 0, 0): ky * k,
            (2, 0, 0, 0): kxx * k,
            (0, 2, 0, 0): kyy * k,
            (1, 1, 0, 0): kxy * k,
            (1, 0, 1, 0): kx * kx,
            (1, 0, 0, 1): kx * ky,
            (0, 1, 0, 1): ky * ky,
            (2, 0, 1, 0): kxx * kx,
            (2, 0, 0, 1): kxx * ky,
            (0, 2, 1, 0): kyy * kx,
            (0, 2, 0, 1): kyy * ky,
            (1, 1, 1, 0): kxy * kx,
            (1, 1, 0, 1): kxy * ky,
            (2, 0, 2, 0): kxx * kxx,
            (0, 2, 2, 0): kyy * kxx,
            (0, 2, 0, 2): kyy * kyy,
            (2, 0, 1, 1): kxx * kxy,
            (0, 2, 1, 1): kyy * kxy,
            (1, 1, 1, 1): kxy * kxy
        }

        # 3. Optimized Batch Convolution
        # Perform padding once for the entire batch to save O(N) allocation time
        k_h, k_w = k.shape
        pad_h, pad_w = k_h // 2, k_w // 2
        padded_data = np.pad(img.data, ((pad_h, pad_h), (pad_w, pad_w)), mode='wrap')

        self.storage = {}
        for orders, prod_kernel in kernel_map.items():
            # Use BORDER_CONSTANT because padding is already handled by 'wrap'
            conv = cv2.filter2D(padded_data, -1, prod_kernel, borderType=cv2.BORDER_CONSTANT)
            # Slice back to original dimensions and store as an Image object
            self.storage[orders] = Image(
                dx_2 * scale * conv[pad_h:-pad_h, pad_w:-pad_w], 
                ll_x, ll_y, dx
            )

        # Smoothed version of the input image (standard convolution)
        self.I = Image(convolve(img.data, k), ll_x, ll_y, dx)

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