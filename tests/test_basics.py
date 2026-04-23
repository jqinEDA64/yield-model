from src import y_basics

import numpy as np

def test_point() :
    # Example usage
    p = y_basics.Point(5.5, 12.0)
    print(p) # Output: Point(x=5.5, y=12.0)

def test_img_grad() :
    # Example usage
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    z = np.exp(-(X**2 + Y**2) / 10)

    # jqin: If using SSH, need to start X server on local machine

    my_img = y_basics.Image(z, ll_x=-5, ll_y=-5, pixel_size=0.2)
    my_img.plot() # Shows the hill

    grad_field = my_img.compute_gradient()
    grad_field.plot()
    grad_field.plot_components()