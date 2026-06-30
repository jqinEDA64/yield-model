"""
Test for Step 3: getDefectDensity_MultiPatterning.

Checks:
1. Returns a finite, non-negative number at a point where the mean images
   agree in sign relative to threshold (near crossing).
2. Returns EXACTLY 0 at a point where mean images are on opposite sides
   of threshold (Eq. 35's sign condition).
3. Reproducibility: running it twice gives the EXACT same answer
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

    center_pt = y_basics.Point(0.0, 0.0)
    offcenter_pt = y_basics.Point(10.0, -5.0)
    th1 = th2 = cov1.I.get(center_pt) + 0.02

    # --- Center point: should be finite, non-negative, nonzero ---
    D_center = ymp.getDefectDensity_MultiPatterning(cov1, cov2, center_pt, th1, th2)
    print(f"D_MP (center) = {D_center:.6e}")
    assert np.isfinite(D_center), "D_center should be finite"
    assert D_center >= 0, "D_center should be non-negative"
    assert D_center > 0, "D_center should be nonzero at this point"
    print("PASS: center point gives a finite, positive defect density")

    # --- Off-center point: mean images on opposite sides -> exactly 0 ---
    D_offcenter = ymp.getDefectDensity_MultiPatterning(cov1, cov2, offcenter_pt, th1, th2)
    print(f"D_MP (off-center) = {D_offcenter:.6e}")
    assert D_offcenter == 0.0, "D_offcenter should be EXACTLY 0 (Eq. 35 sign condition)"
    print("PASS: off-center point gives exactly 0")

    # --- Reproducibility check ---
    D_center_again = ymp.getDefectDensity_MultiPatterning(cov1, cov2, center_pt, th1, th2)
    assert D_center == D_center_again, "Result should be EXACTLY reproducible (no randomness)!"
    print("PASS: result is exactly reproducible across repeated calls")

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()