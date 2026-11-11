import numpy as np
from collections import namedtuple


OrientedEllipse = namedtuple("OrientedEllipse", ("center", "angle", "width", "height"))


def ellipse_approximation(xy, weights, a=1.):
    """Compute an ellipse from a probability distribution over hexels.

    Parameters
    ----------
    xy : array_like, shape (n_points,2)
        XY coordinates of hexels.
    weights : array_like, shape (n_points,)
        Weights associated with each hexel.
    a : float, optional
        Lattice constance (distance between neighboring hexels).

    Returns
    -------
    OrientedEllipse
        Centroid, orientation (of long axis), width (of long axis) and height (short axis) of the ellipse.
    """
    # Check shapes
    xy = np.atleast_2d(xy)
    weights = np.atleast_1d(weights)
    assert xy.shape[0] == len(weights)
    # Compute centroid
    weights_norm = weights / weights.sum()
    centroid = weights_norm @ xy
    # Center points
    xy_centered = xy - centroid
    # Compute covariance
    cov = np.einsum("n,ni,nj->ij", weights_norm, xy_centered, xy_centered)
    cov = cov + (np.eye(2) * (5. * (a ** 2) / 36.))  # correct for area of a single hexel
    # Compute long and short axis
    evals, evecs = np.linalg.eig(cov)
    short, long = np.argsort(evals)
    h, w = 2 * np.sqrt(evals[[short, long]])
    v = evecs[:, long]
    theta = np.arctan2(v[1], v[0])
    return OrientedEllipse(centroid, theta, w, h)
