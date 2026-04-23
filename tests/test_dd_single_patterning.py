from src import y_basics, y_litho, y_math, y_stochastics

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pytest


def test_dd_1D_LS(do_plot, do_print) :
  
    pitch  = 40   # [nm]
    x_vals = np.linspace(-1.5*pitch, 1.5*pitch, pitch*20)

    # The precomputation is quite slow
    img    = y_litho.generate_LS_2D(pitch, 0.55, 1.0/13.5, x_vals)
    RI     = y_basics.Image(data = img, ll_x = 0, ll_y = 0, pixel_size = x_vals[1]-x_vals[0])
    RI_Cov = y_stochastics.GaussianCovariance(img = RI, sigma = 8, scale = 1)

    # This part is not so bad, when E(|det|) is approximated as 1
    x_dd = np.linspace(RI.x_coords[0]+10, RI.x_coords[-1]-10, 100)
    d_vals = [y_math.getDefectDensity(RI_Cov, y_basics.Point(x, 60), th = 0.5) for x in tqdm(x_dd, desc="Computing defect densities")]

    dd_min = np.min(d_vals)
    dd_max = np.max(d_vals)

    if do_print :
        print("Minimum defect density = ", dd_min)
        print("Maximum defect density = ", dd_max)

    if do_plot :
        RI.plot(title = "Resist image")
        plt.plot(x_dd, d_vals)
        plt.yscale("log")
        plt.title("Defect density in 1D LS pattern")
        plt.show()
    
    assert np.log10(dd_min) == pytest.approx(np.log10(9.5e-184), rel=1e-1)
    assert np.log10(dd_max) == pytest.approx(np.log10(1.0e-13 ), rel=1e-1)
