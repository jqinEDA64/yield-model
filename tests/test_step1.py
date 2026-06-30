"""
Test for Step 1: getMean_Multi and getCov_Multi.

Checks:
1. Shapes are correct (mean is (6,), cov is (6,6))
2. The values actually match calling y_math directly on each image
   (i.e. we haven't mixed up I1/I2 or scrambled indices)
3. The matrix is symmetric (a covariance matrix always must be)
4. The off-diagonal blocks are exactly zero (independence assumption)
"""

import numpy as np
from src import y_basics, y_litho, y_stochastics, y_math
from src import y_multipatterning as ymp


def build_pattern(pitch, duty_cycle, cutoff_freq, x_vals, sigma, scale, orientation):
    img_data = y_litho.generate_LS_2D(pitch, duty_cycle, cutoff_freq, x_vals)
    if orientation == 'horizontal':
        img_data = img_data.T
    img = y_basics.Image(data=img_data, ll_x=x_vals[0], ll_y=x_vals[0],
                          pixel_size=x_vals[1] - x_vals[0])
    cov = y_stochastics.GaussianCovariance(img=img, sigma=sigma, scale=scale)
    return cov


def main():
    pitch = 40
    x_vals = np.linspace(-40, 40, 40)
    cutoff_freq = 1.0 / 13.5

    cov1 = build_pattern(pitch, 0.5, cutoff_freq, x_vals, sigma=6, scale=1.0, orientation='vertical')
    cov2 = build_pattern(pitch, 0.5, cutoff_freq, x_vals, sigma=6, scale=1.0, orientation='horizontal')

    pt = y_basics.Point(3.0, -2.0)  # arbitrary point, not the origin, to be a real test

    mu = ymp.getMean_Multi(cov1, cov2, pt)
    Sigma = ymp.getCov_Multi(cov1, cov2, pt)

    print("mu shape:", mu.shape, "  Sigma shape:", Sigma.shape)
    assert mu.shape == (6,), f"Expected mu shape (6,), got {mu.shape}"
    assert Sigma.shape == (6, 6), f"Expected Sigma shape (6,6), got {Sigma.shape}"
    print("PASS: shapes correct")

    # Check values against calling y_math directly
    mu1_direct = y_math.getMean_X(cov1.I, pt)
    mu2_direct = y_math.getMean_X(cov2.I, pt)
    assert np.allclose(mu[0:3], mu1_direct), "First 3 entries of mu don't match I1's direct mean!"
    assert np.allclose(mu[3:6], mu2_direct), "Last 3 entries of mu don't match I2's direct mean!"
    print("PASS: mean values match direct y_math calls")

    Sigma1_direct = y_math.getCov_XX(cov1, pt)
    Sigma2_direct = y_math.getCov_XX(cov2, pt)
    assert np.allclose(Sigma[0:3, 0:3], Sigma1_direct), "Top-left block doesn't match I1's direct covariance!"
    assert np.allclose(Sigma[3:6, 3:6], Sigma2_direct), "Bottom-right block doesn't match I2's direct covariance!"
    print("PASS: covariance diagonal blocks match direct y_math calls")

    # Symmetry check
    assert np.allclose(Sigma, Sigma.T), "Sigma is not symmetric -- covariance matrices must be!"
    print("PASS: Sigma is symmetric")

    # Independence check: off-diagonal blocks must be exactly zero
    assert np.allclose(Sigma[0:3, 3:6], 0), "Top-right block should be zero (independence)!"
    assert np.allclose(Sigma[3:6, 0:3], 0), "Bottom-left block should be zero (independence)!"
    print("PASS: off-diagonal blocks are zero (independence)")

    print("\nALL TESTS PASSED")
    print("\nmu =", mu)
    print("\nSigma diag =", np.diag(Sigma))


if __name__ == "__main__":
    main()