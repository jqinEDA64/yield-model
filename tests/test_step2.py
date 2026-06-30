"""
Test for Step 2: build_constraint and density_at_tangency.

Checks:
1. build_constraint produces the right shapes and correctly encodes the
   four equations by hand-checking a simple case.
2. density_at_tangency returns a valid probability density (non-negative,
   finite).
3. Sanity check: density should be HIGHER when (u, lambda) corresponds to
   values close to the actual mean behavior at that point, and LOWER far
   away -- i.e. it should behave like an actual peaked probability density,
   not a flat/broken function.
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
    # --- Test build_constraint in isolation, by hand ---
    A, x0 = ymp.build_constraint(lam=2.0, th1=0.5, th2=0.6, u=0.1)
    assert A.shape == (4, 6)
    assert x0.shape == (4,)

    v_test = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # dx1,dy1,I1,dx2,dy2,I2
    Av = A @ v_test
    expected = np.array([
        v_test[ymp.IDX['I1']],
        v_test[ymp.IDX['I2']],
        v_test[ymp.IDX['dx1']] - 2.0*v_test[ymp.IDX['dx2']],
        v_test[ymp.IDX['dy1']] - 2.0*v_test[ymp.IDX['dy2']],
    ])
    assert np.allclose(Av, expected), f"A@v mismatch: got {Av}, expected {expected}"
    assert np.allclose(x0, [0.6, 0.7, 0.0, 0.0]), f"x0 wrong: {x0}"
    print("PASS: build_constraint encodes the right linear equations")

    pitch = 40
    x_vals = np.linspace(-40, 40, 40)
    cutoff_freq = 1.0 / 13.5
    cov1 = build_pattern(pitch, 0.5, cutoff_freq, x_vals, sigma=6, scale=1.0, orientation='vertical')
    cov2 = build_pattern(pitch, 0.5, cutoff_freq, x_vals, sigma=6, scale=1.0, orientation='horizontal')

    pt = y_basics.Point(0.0, 0.0)
    mu = ymp.getMean_Multi(cov1, cov2, pt)
    Sigma = ymp.getCov_Multi(cov1, cov2, pt)

    I0_1 = cov1.I.get(pt)
    I0_2 = cov2.I.get(pt)
    print(f"I0_1={I0_1:.4f}, I0_2={I0_2:.4f}")

    th1 = th2 = I0_1 + 0.02
    u_close = 0.02
    u_far   = 5.0

    p_close = ymp.density_at_tangency(mu, Sigma, lam=1.0, th1=th1, th2=th2, u=u_close)
    p_far   = ymp.density_at_tangency(mu, Sigma, lam=1.0, th1=th1, th2=th2, u=u_far)

    print(f"density at u_close={u_close}: {p_close:.6e}")
    print(f"density at u_far={u_far}: {p_far:.6e}")

    assert p_close >= 0 and np.isfinite(p_close)
    assert p_far >= 0 and np.isfinite(p_far)
    assert p_close > p_far, "density should be HIGHER near the mean than far away!"
    print("PASS: density_at_tangency behaves like a real peaked probability density")

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
