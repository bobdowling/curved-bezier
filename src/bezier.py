from typing import Callable

import numpy


def line(
    t: numpy.ndarray,
    x: numpy.ndarray,
) -> numpy.ndarray:
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


def bezier(
    t: numpy.ndarray,
    x: numpy.ndarray,
    f: Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray],
) -> numpy.ndarray:
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
        xx = numpy.stack([x0, x1])
        return f(t, xx)
