from src import y_math

import numpy as np
import pytest

def test_schur_complement() :

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

    assert new_mu [0]   == pytest.approx(0.80, rel=1e-3)
    assert new_cov[0,0] == pytest.approx(0.36, rel=1e-3)