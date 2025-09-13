from typing import Callable

import numpy as np
import numpy.typing as npt


def line(
    t: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
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
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
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


def process(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64],  # xx
    npt.NDArray[np.float64],  # yy
    npt.NDArray[np.float64],  # mm
    npt.NDArray[np.float64],  # nn
    npt.NDArray[np.float64],  # a
    npt.NDArray[np.float64],  # b
    npt.NDArray[np.float64],  # c
    npt.NDArray[np.float64],  # theta
]:
    """Takes two 3D vectors, x and y, and returns
    ̂x: Normalized x
    ̂y: Normalized y
    ̂m: Normalized x×y (raises ValueError if this is zero)
    ̂n: Normalized x×(x×y)
    a: Coordinate of x in the (̂x, ̂n) coordinate system (x = (a,0))
    b, c: Coordinates of y in the (̂x, ̂n) coordinate system (y = (b,c))
    θ: The angle between  x and y (in radians)
    """
    EPSILON = 1.0e-7
    xx = x / np.sqrt(np.dot(x, x))
    yy = y / np.sqrt(np.dot(y, y))
    m = cross(xx, yy)
    if np.dot(m, m) < EPSILON:
        raise ValueError(f"Vectors too close to [anti-]parallel: {x=}: {y=}")
    mm = m / np.sqrt(np.dot(m, m))
    nn = cross(mm, xx)
    a = np.dot(x, xx)
    b = np.dot(y, xx)
    c = np.dot(y, nn)
    theta = np.acos(np.dot(xx, yy))
    return xx, yy, mm, nn, a, b, c, theta


def bezier(
    t: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    f: Callable[
        [npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]
    ],
) -> npt.NDArray[np.float64]:
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
