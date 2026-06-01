import numpy as np
from y_basics import Point


def checkTangency(x, y, img1, img2, th1, th2, tol=0.04):
    """
    Tangency occurs when both conditions are satisfied:
    1. I1 - th1 ≈ I2 - th2
    2. ∇I1 ∥ ∇I2
    """
    pt = Point(x, y)
    
    # Condition 1: boundaries cross at same height
    cross = abs((img1.get(pt) - th1) - (img2.get(pt) - th2)) < tol
    
    # Condition 2: gradients parallel
    grad_I1_x = img1.compute_der_pt(pt, dir="x")
    grad_I1_y = img1.compute_der_pt(pt, dir="y")
    grad_I2_x = img2.compute_der_pt(pt, dir="x")
    grad_I2_y = img2.compute_der_pt(pt, dir="y")
    
    # Check if ∇I1 = λ∇I2 for some λ
    if abs(grad_I2_x) > 1e-10 and abs(grad_I2_y) > 1e-10:
        lambda_x = grad_I1_x / grad_I2_x
        lambda_y = grad_I1_y / grad_I2_y
        parallel = abs(lambda_x - lambda_y) < tol
    else:
        parallel = False
    
    return cross and parallel


def findTangencyPoints(img1, img2, th1, th2, x_range, y_range):
    tangency_points = []
    
    for x in x_range:
        for y in y_range:
            if checkTangency(x, y, img1, img2, th1, th2):
                tangency_points.append((x, y))
    
    return tangency_points