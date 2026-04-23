from src import y_math, y_basics

import numpy as np
import pytest

def test_schur_complement(do_print) :

    # Example
    # Assume two correlated 1D variables
    mu  = np.array([0, 0])
    cov = np.array([[1.0, 0.8],
                    [0.8, 1.0]])

    # Condition on x = 1.0
    new_mu, new_cov = y_math.condition_gaussian(
        mu[0], mu[1],
        np.array([[cov[0,0]]]), np.array([[cov[1,1]]]),
        np.array([[cov[0,1]]]),
        np.array([1.0])
    )

    print("New mean:", new_mu)
    print("New covariance:", new_cov)

    assert new_mu [0]   == pytest.approx(0.80, rel=1e-3)
    assert new_cov[0,0] == pytest.approx(0.36, rel=1e-3)

def test_Y_distribution(do_plot, do_print) :
    p1 = y_basics.Point(12.5, 12.5)

    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    z = X**2 + Y**2

    my_img = y_basics.Image(z, ll_x=-5, ll_y=-5, pixel_size=0.2)
    
    if do_plot:
        my_img.plot(title = "Input image")

    gk = y_basics.Image(y_basics.getGaussian(12, 4), ll_x = 0, ll_y = 0, pixel_size = 4)
    cov = y_basics.Covariance(my_img, gk.data)

    mean, cov_mat = y_math.getMean_Cov_Y_cond(img = my_img, cov = cov, pt = p1, th = 27)
    if do_print:
        print(mean)
        print(cov_mat)

    mean_correct = np.array([4.24379458, 4.24379458, 0.51620564])
    cov_mat_correct = np.array([[2.02761663e+01, -7.71926415e-05, -6.80488761e-02],
                                [-7.71926415e-05, 2.02761663e+01, -6.80488761e-02],
                                [-6.80488761e-02, -6.80488761e-02, 1.12939996e+01]])
    
    assert np.linalg.norm(mean    - mean_correct   ) == pytest.approx(0, abs=1e-3)
    assert np.linalg.norm(cov_mat - cov_mat_correct) == pytest.approx(0, abs=1e-3)