import numpy as np


def mollweide_projection(x3d, y3d, z3d, tol=1e-8, niter=1000):
    """Computes the mollweide projection of points lying on a unit sphere using Newton-Raphson iteration.
    See: https://en.wikipedia.org/wiki/Mollweide_projection
    """
    lon = np.atan2(y3d, x3d)
    lat = np.atan2(z3d, np.linalg.norm([x3d, y3d], axis=0))
    theta = np.array(lat)
    for i in range(niter):
        dtheta = (2 * theta + np.sin(2 * theta) - np.pi * np.sin(lat)) / (2 + 2 * np.cos(2 * theta))
        if np.abs(dtheta).max() < tol:
            break
        theta = theta - dtheta
    x_mollweide = 2 * np.sqrt(2) * lon * np.cos(theta) / np.pi
    y_mollweide = np.sqrt(2) * np.sin(theta)
    return np.stack([x_mollweide, y_mollweide], axis=-1)


def rotation_3d(angle, axis):
    """Generates a 3D rotation matrix that rotates a set of points around the given axis (right hand rule).

    Parameters
    ----------
    angle : float or 1d ndarray of floats
        Angle(s) in radians.
    axis : int
        Rotation axis. 0 for x, 1 for y, 2 for z.

    Returns
    -------
    R : ndarray of floats
        3D rotation matrix of shape (3, 3, ...).
    """
    ca = np.cos(angle)
    sa = np.sin(angle)
    R = np.array([[ca, -sa, sa],
                  [sa, ca, -sa],
                  [-sa, sa, ca]])
    R[axis] = 0
    R[:, axis] = 0
    R[axis, axis] = 1
    return R


def pq_to_xy(pq, flip=False):
    _basis = np.array([[-np.sqrt(3) / 2., 1. / 2.],
                       [np.sqrt(3) / 2., 1. / 2.]])
    if flip:
        _basis = _basis * (-1, 1)
    return pq @ _basis
