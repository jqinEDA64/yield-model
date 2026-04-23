from src import y_basics, y_litho
import numpy as np
import matplotlib.pyplot as plt

def test_1D_LS_variable(do_plot) :

    if not do_plot :
        return

    pitch  = 40   # [nm]
    x_vals = np.arange(-2*pitch, 2*pitch, pitch/80)
    dc_vals= np.linspace(0.05, 0.95, 20)
    cd_vals= []

    for dc in dc_vals :
        f      = 1.0/13.5
        img    = y_litho.generate_LS_1D(pitch, dc, f, x_vals)
        cd     = y_litho.comp_Space_CD (pitch, dc, f, x_vals, 0.2)
        cd_vals.append(cd)
        plt.plot(x_vals, img, label = "duty cycle = " + y_basics.flToStr(dc) + ", CD = " + y_basics.flToStr(cd))
    plt.legend()
    plt.title("1D Line-Space Pattern @ 40 nm pitch\nwith varying duty cycle")
    plt.show()

    plt.plot(dc_vals, cd_vals)
    plt.title("CD vs Duty Cycle")
    plt.show()

def test_1D_LS_fixed(do_plot) :

    if not do_plot :
        return

    pitch  = 40   # [nm]
    x_vals = np.linspace(-1.5*pitch, 1.5*pitch, pitch*20)

    img    = y_litho.generate_LS_2D(pitch, 0.55, 1.0/13.5, x_vals)
    RI     = y_basics.Image(data = img, ll_x = 0, ll_y = 0, pixel_size = x_vals[1]-x_vals[0])
    RI.plot(title = "1D Line-Space Pattern")