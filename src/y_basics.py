import copy
import cv2
import numpy as np
import math
from scipy.fftpack import idct
from scipy.stats import norm, qmc, multivariate_normal
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator, CubicSpline
import matplotlib.pyplot as plt
from dataclasses import dataclass
import warnings
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
# Data classes: Point, image, gradient, etc.
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
    x, y = pt.x, pt.y
    val_x = self.u.get(x, y)
    val_y = self.v.get(x, y)

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