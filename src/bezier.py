from typing import Callable

import numpy as np


def line(
    t: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Linear interpolation between two points, x[0] at t==0 and x[1] at t==1.
    t:  The interpolation parameter. Expected to hold values between 0.0 and 1.0,
        but this is not actually required and extrapolation is possible.
    x:  The end points of the line segment are in x[0] and x[1].

    Returns:    (1.0 - t) * x[0].reshape((-1,1)) + t * x[1].reshape((-1,1))
    """
    if x.shape[0] != 2:
        raise ValueError(
            f"Interpolation function requires exactly two end points in x[0] and x[1]: {x.shape=}"
        )

    xx = (1.0 - t) * x[0].reshape((-1, 1)) + t * x[1].reshape((-1, 1))
    return xx


def cross(
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """3D cross product: Takes two 3D vectors and returns their cross-product."""
    if x.shape[0] != 3 or y.shape[0] != 3:
        raise ValueError(
            f"Cross product requires 3D vectors: Axis 0 of both inputs must have length 3; {x.shape=}: {y.shape=}"
        )
    z = np.stack(
        [
            x[1] * y[2] - x[2] * y[1],
            x[2] * y[0] - x[0] * y[2],
            x[0] * y[1] - x[1] * y[0],
        ]
    )
    return z


def bezier(
    t: np.ndarray,
    x: np.ndarray,
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Recursive definition of bezier curves.
    t:  The interpolation parameter. Expected to hold values between 0.0 and 1.0,
        but this is not actually required and extrapolation is possible.
    x:  Shape[n,d]: The n d-dimensional control points, including the end points at x[0] and x[n-1].
    f:  The base case functoin that interpolates between two end points.
    """
    if x.shape[0] == 1:
        return x
    elif x.shape[0] == 2:
        return f(t, x)
    else:
        # There is scope for significant memo-optimization here!
        x0 = bezier(t, x[:-1], f)
        x1 = bezier(t, x[1:], f)
        xx = np.stack([x0, x1])
        return f(t, xx)
