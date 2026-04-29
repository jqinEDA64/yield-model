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
    
    if   y_math.ABS_DET_FLAG == 0 :
        assert np.log10(dd_min) == pytest.approx(np.log10(9.5e-184), rel=1e-1)
        assert np.log10(dd_max) == pytest.approx(np.log10(1.0e-13 ), rel=1e-1)
    elif y_math.ABS_DET_FLAG == 2 :
        assert np.log10(dd_min) == pytest.approx(np.log10(6.952503392675953e-190), rel=1e-1)
        assert np.log10(dd_max) == pytest.approx(np.log10(5.529240398477354e-19 ), rel=1e-1)


def getDDandCD(pitch, duty_cycle, th, scale = 1.0) :
  x_vals = np.linspace(-2*pitch, 2*pitch, int(pitch*20))
  cutoff_freq = 1.0/13.5
  img    = y_litho.generate_LS_2D(pitch, duty_cycle, cutoff_freq, x_vals)
  RI     = y_basics.Image(data = img, ll_x = 0, ll_y = 0, pixel_size = x_vals[1]-x_vals[0])
  RI_Cov = y_stochastics.GaussianCovariance(RI, sigma = 8, scale = scale)

  x_vals_pitch = 1.5*pitch + np.linspace(0, pitch, 50)
  d_vals_pitch = 1e14*np.array([y_math.getDefectDensity(RI_Cov, y_basics.Point(x, 2*pitch), th = th) for x in x_vals_pitch])
  ri_vals_pitch= [RI_Cov.I.get(y_basics.Point(x, 2*pitch)) for x in x_vals_pitch]

  # 2. Create the figure and the first axes (ax1)
  plt.close('all')
  fig, ax1 = plt.subplots()

  # 3. Plot the first line on the left y-axis
  color1 = 'tab:red'
  ax1.set_xlabel('$x$ [nm]')
  ax1.set_ylabel('Defect density [cm$^{-2}$]', color=color1)
  ax1.set_yscale("log")
  ax1.set_ylim(1e-80, 1e30)
  ax1.plot(x_vals_pitch, d_vals_pitch, color=color1)
  ax1.tick_params(axis='y', labelcolor=color1)

  # 4. Instantiate a second axes (ax2) that shares the same x-axis
  ax2 = ax1.twinx()

  # 5. Plot the second line on the right y-axis
  color2 = 'tab:blue'
  ax2.set_ylabel('RI($x$)', color=color2)
  ax2.plot(x_vals_pitch, ri_vals_pitch, color=color2)
  ax2.set_ylim(0, 1.0)
  ax2.axhline(y=th, color='darkblue', linestyle='--', label='Threshold')
  ax2.tick_params(axis='y', labelcolor=color2)

  # 6. Optional: Add a title and adjust layout
  fig.legend()
  fig.tight_layout() # Ensures labels don't get clipped

  # 7. Display the plot
  #plt.show()
  folder = "output/1D_LS_DD"
  plt.savefig(folder + f"/pitch={pitch}_duty_cycle={duty_cycle}_th={th}_scale={scale}.png", dpi=300)

  cd = y_litho.comp_Space_CD(pitch, duty_cycle, cutoff_freq, x_vals, th)
  return y_math.adaptive_log_integrate(x_vals_pitch, d_vals_pitch)/(np.ptp(x_vals_pitch)), cd


def test_IMEC_LS(do_plot) :
    dc_vals = np.linspace(0.34, 0.75, 20)
    results_1 = [getDDandCD(pitch = 36, duty_cycle = dc, th = 0.5, scale = 0.9)  for dc in tqdm(dc_vals, desc="Scale = 0.90")]
    results_2 = [getDDandCD(pitch = 36, duty_cycle = dc, th = 0.5, scale = 0.95) for dc in tqdm(dc_vals, desc="Scale = 0.95")]

    dd_vals_1, cd_vals_1 = zip(*results_1)
    dd_vals_2, cd_vals_2 = zip(*results_2)

    plt.close('all')
    plt.plot(cd_vals_1, dd_vals_1, label = "Scale = 0.90", color = "blue")
    plt.yscale("log")
    plt.ylabel("Defect density [cm$^{-2}$]")
    plt.xlabel("Space CD [nm]")
    plt.savefig("output/1D_LS_DD/DefectDensity_vs_SpaceCD_0.90.png", dpi=300)
    if do_plot :
        plt.show()
    plt.close('all')


    plt.plot(cd_vals_2, dd_vals_2, label = "Scale = 0.95", color = "red")
    plt.yscale("log")
    plt.ylabel("Defect density [cm$^{-2}$]")
    plt.xlabel("Space CD [nm]")
    plt.savefig("output/1D_LS_DD/DefectDensity_vs_SpaceCD_0.95.png", dpi=300)
    if do_plot :
        plt.show()
    plt.close('all')