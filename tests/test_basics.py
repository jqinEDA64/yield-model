from src import y_basics

import numpy as np
import pytest




def test_point() :
    # Example usage
    p = y_basics.Point(5.5, 12.0)
    assert True

def test_img_grad(do_plot) :
    # Example usage
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    z = np.exp(-(X**2 + Y**2) / 10)

    # jqin: If using SSH, need to start X server on local machine

    my_img = y_basics.Image(z, ll_x=-5, ll_y=-5, pixel_size=0.2)
    grad_field = my_img.compute_gradient()

    if do_plot:
        my_img.plot               (title = "Original image")      # Shows the hill
        grad_field.plot           (title = "Gradient field")      # Shows the gradient vectors
        grad_field.plot_components(title = "Gradient components") # Shows the x and y components separately

    assert grad_field.get(y_basics.Point(0, 0)) == pytest.approx((-0.02, -0.02), rel=1e-1)

def test_covariance(do_plot) :

    # Define a Gaussian kernel
    gk = y_basics.Image(y_basics.getGaussian(12, 4), ll_x = 0, ll_y = 0, pixel_size = 4)
    #gk.plot()
    print("Sum of Gaussian kernel data:", np.sum(gk.data))

    p1 = y_basics.Point(12.5, 12.5)
    p2 = y_basics.Point(2.0, 2.0)

    # Example usage
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    z = X**2 + Y**2

    my_img = y_basics.Image(z, ll_x=-5, ll_y=-5, pixel_size=0.2)
    if do_plot:
        my_img.plot(title = "Input image")

    # Covariance and its first-order derivative w.r.t x1 and x2
    cov     = y_basics.Covariance(my_img, gk.data, 0.1)
    cov_der = cov     .derivative(p1, orders=(1, 0, 1, 0))

    if do_plot:
        cov.S_1_0_1_0.plot(title = "Covariance derivative $\\partial x_1 \\partial x_2 \Sigma(x_1, x_2)$")

    assert cov_der == pytest.approx(0.86169, rel=1e-3)