import numpy as np
from matplotlib import pyplot as plt


class HexelMapper:

    _basis = np.array([[np.sqrt(3) / 2., -np.sqrt(3) / 2.],
                      [1. / 2., 1. / 2.]])
    _basis_inv = np.linalg.inv(_basis)

    def __init__(self, xscale=1., yscale=1.):
        self.xscale = xscale
        self.yscale = yscale

    def __call__(self, p, q) -> np.ndarray:
        xy = (self._basis @ np.vstack([np.atleast_1d(p), np.atleast_1d(q)])).T
        return xy * np.array([self.xscale, self.yscale])

    def invert(self, x, y):
        xy = np.vstack([np.atleast_1d(x / self.xscale), np.atleast_1d(y / self.yscale)])
        return (self._basis_inv @ xy).T


class HexelPlot:

    vertices = np.array([[np.sqrt(3) / 3., 0],
                         [np.sqrt(3) / 6., 1. / 2.],
                         [-np.sqrt(3) / 6., 1. / 2.],
                         [-np.sqrt(3) / 3., 0],
                         [-np.sqrt(3) / 6., -1. / 2.],
                         [np.sqrt(3) / 6., -1. / 2.]])

    def __init__(self, xscale=1., yscale=1., hexel_kwargs=None):
        self.xscale = xscale
        self.yscale = yscale
        self.hexel_kwargs = hexel_kwargs or {}

    def hexagon(self, x, y, c, alpha=1., **kwargs):
        vertices = self.vertices * np.array([[self.xscale, self.yscale]]) + np.array([[x, y]])
        kw = self.hexel_kwargs.copy()  # defaults
        kw.update(kwargs)  # overwrite
        kw.update(dict(color=c, alpha=alpha))  # color and alphs
        return plt.Polygon(vertices, closed=True, **kw)

    def create_patches(self, x, y, c, cmap=None, vmin=None, vmax=None, alpha=None, **kwargs):
        assert len(x) == len(y)
        # Block color
        if isinstance(c, str):
            color = [c] * len(x)
            alpha = np.array([alpha if alpha is not None else 1] * len(x))
        # Scalar mappable
        elif isinstance(c, np.ndarray) & (c.ndim == 1):
            if vmin is None:
                vmin = c.min()
            if vmax is None:
                vmax = c.max()
            cmap = plt.cm.ScalarMappable(plt.Normalize(vmin=vmin, vmax=vmax), cmap or "viridis")
            color = np.array([cmap.to_rgba(val) for val in c])[:, :3]
            if not isinstance(alpha, np.ndarray):
                alpha = np.ones(len(x)) * (alpha if alpha is not None else 1)
        # RGB or RGBA array
        elif isinstance(c, np.ndarray) & (c.shape[1] in (3, 4)):
            color = c[:, :3]
            if c.shape[1] == 4:
                alpha = c[:, -1]
            elif not isinstance(alpha, np.ndarray):
                alpha = np.ones(len(x)) * (alpha if alpha is not None else 1)
        else:
            raise ValueError("c must be str or ndarray")
        assert isinstance(alpha, np.ndarray)
        assert len(x) == len(color) == len(alpha)
        # Create patches
        patches = [self.hexagon(xi, yi, ci, alpha=ai, **kwargs) for xi, yi, ci, ai in zip(x, y, color, alpha)]
        return patches

    def hexplot(self, x, y, c=None, cmap=None, vmin=None, vmax=None, alpha=None, ax=None, **kwargs):
        """Add hexels to a plot.

        Parameters
        ----------
        x, y: array-like
        """
        ax = ax or plt.gca()
        c = "C0" if c is None else c
        hexels = self.create_patches(x, y, c, cmap, vmin, vmax, alpha, **kwargs)
        for hexel in hexels:
            ax.add_patch(hexel)
        return hexels
