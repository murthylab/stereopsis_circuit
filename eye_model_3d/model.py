import numpy as np
from .geometry import mollweide_projection, rotation_3d
import typing


class EyeModel:
    """Model of the Drosophila melanogaster eye based on microCT data from Zhao et al., 2025.

    This class is a model of a single fly eye with N ommatidia. Default behavior assumes the left eye. For a right eye
    model, set `is_right` flag to `True`.

    Parameters
    ----------
    xyz_l: (N, 3) ndarray of floats
        The xyz coordinates of the ommatidia lenses (in um relative to the center of the two eyes).
    xyz_c: (N, 3) ndarray of floats
        The xyz coordinates of the ommatidia cones (in um relative to the center of the two eyes).
    pq: (N, 2) ndarray of floats
        The pq hexagonal coordinates of the ommatidia.
    column_ids: (N, ) ndarray of ints, optional
        Unique id for each column.
    convergence: float, optional (default = 0)
        The amount of eye convergence (in radians).
    dorsal_convergence: float, optional (default = 0)
        The amount of dorsalward eye convergence (in radians).
    is_right: bool, optional (default = False)
        Whether model is for right eye. Changes the sign of ommatidia rotation angles.

    References
    ----------
    Zhao, A., Gruntman, E., Nern, A. et al. Eye structure shapes neuron function in Drosophila motion vision.
    Nature 646, 135–142 (2025). https://doi.org/10.1038/s41586-025-09276-5
    """
    def __init__(self, xyz_l, xyz_c, pq, column_ids=None, convergence=0, dorsal_convergence=0, is_right=False):
        self._is_right = is_right
        self.xyz_l = np.array(xyz_l)
        self.xyz_c = np.array(xyz_c)
        self.pq = np.array(pq)
        self.column_ids = column_ids
        self.convergence = convergence
        self.dorsal_convergence = dorsal_convergence
        assert self.xyz_l.shape == self.xyz_c.shape

    def get_params(self):
        """Returns a dictionary of parameters to initialize new instance."""
        return {
            "xyz_l": self.xyz_l,
            "xyz_c": self.xyz_c,
            "pq": self.pq,
            "convergence": self.convergence,
            "dorsal_convergence": self.dorsal_convergence,
            "is_right": self.is_right
        }

    def __repr__(self):
        return f"{type(self).__name__}(n_ommatidia={self.n}, eye={'right' if self.is_right else 'left'})"

    @property
    def n(self) -> int:
        """Number of ommatidia."""
        return self.xyz_l.shape[0]

    @property
    def I(self) -> np.ndarray:
        """Return a stack of (3 x 3) identity matrices equal to the number of ommatidia."""
        return np.repeat(np.eye(3)[None], self.n, axis=0)

    @property
    def xyz_u(self) -> np.ndarray:
        """Unit vectors pointing in the direction of the sight lines for all ommatidia."""
        u = self.xyz_l - self.xyz_c
        u = u / np.linalg.norm(u, axis=-1, keepdims=True)
        if self.convergence != 0:
            zRot = rotation_3d(self._eye_angle, axis=2)
            u = np.einsum("ij,nj->ni", zRot, u)
        if self.dorsal_convergence != 0:
            xRot = rotation_3d(self._eye_angle, axis=0)
            u = np.einsum("ij,nj->ni", xRot, u)
        return u

    @property
    def is_right(self) -> bool:
        """True if model is for a right eye."""
        return self._is_right

    @property
    def column_ids(self) -> np.ndarray:
        """Unique id for each column."""
        return self._column_ids

    @column_ids.setter
    def column_ids(self, column_ids):
        if column_ids is None:
            self._column_ids = np.arange(self.n)
        else:
            assert len(column_ids) == self.n
            self._column_ids = np.array(column_ids)

    @property
    def convergence(self) -> float:
        """Inward vergence angle of the ommatidia sight lines (in radians)."""
        if self.is_right:
            return self._eye_angle
        return -self._eye_angle

    @convergence.setter
    def convergence(self, value):
        if self.is_right:
            self._eye_angle = value
        else:
            self._eye_angle = -value

    @property
    def dorsal_convergence(self) -> float:
        """Upward vergence angle of the ommatidia sight lines (in radians)."""
        if self.is_right:
            return self._dorsal_eye_angle
        return -self._dorsal_eye_angle

    @dorsal_convergence.setter
    def dorsal_convergence(self, value):
        if self.is_right:
            self._dorsal_eye_angle = value
        else:
            self._dorsal_eye_angle = -value

    def project_onto_sightlines(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Find the shortest distance between a set of points and ommatidia sight lines.

        Parameters
        ----------
        points: (..., 3) ndarray of floats
            Set of points (xyz coordinates) for sight lines to be projected towards.

        Returns
        -------
        d, r: np.ndarray of floats
            Final dimensions matches number of ommatidia, other dimensions match input points shape. d is the distance
            from the lenses to the nearest point. r is the distance from the points to the sight lines.
        """
        points = np.atleast_2d(points)
        u = self.xyz_u
        outer = np.einsum("ni,nj->nij", u, u, optimize="greedy")
        v = points[..., None, :] - self.xyz_l
        q = np.einsum("nij,...nj->...ni", self.I - outer, v, optimize="greedy")
        d = np.einsum("ni,...ni->...n", u, v, optimize="greedy")
        s = np.linalg.norm(q, axis=-1)
        return d, s

    def column_activation(self, points: np.ndarray, radius: typing.Union[float, np.ndarray], newaxis=True) -> np.ndarray:
        """Compute the activation of each column for stimuli centered on points with the given radii.

        Parameters
        ----------
        points: (..., N, 3) ndarray of floats
            Centroids (xyz coordinates) of spherical stimuli.
        radius: float or (N,) or (M,) np.ndarray of floats
            Radius or radii of spherical stimuli.
        newaxis: bool, optional
            If True (default), compute the column activation for all pairwise combination of points and radii. If False,
            M must equal N and radii represent the radius of each stimulus in points.

        Returns
        -------
        activations: np.ndarray of floats, shape (N, K) or (N, M, K)
            Cosine of the angle of intersection of K ommatidia sight lines with the normals of N spheres with M radii.
        """
        # Compute projection of points onto sight lines
        d, s = self.project_onto_sightlines(points)
        # Manage axes
        radii = np.atleast_1d(radius)
        radii = radii[:, None]
        if newaxis:
            # radii = radii[..., None]
            s = s[..., None, :]
            d = d[..., None, :]
        else:
            assert len(radii) in (1, points.shape[-2]), "radius does match, try setting newaxis=True"
        # Compute angle of intersection
        sin = np.clip(s / radii, 0, 1)
        a = np.sqrt(1 - np.square(sin))
        # Remove activations behind retina
        a[np.where(d < 0)] = 0
        return a

    @property
    def angles(self):
        """Return the Euler angles of the sight lines."""
        return self.xyz_u @ np.eye(3)

    def mollweide(self, distance=None, flip=True, tol=1e-8, niter=1000):
        """Compute the mollweide projection of ommatidia sight lines at a given distance.

        Parameters
        ----------
        distance: float, optional
            Distance to which sight lines should be projected.
        flip: bool, optional (default=True)
            If True, flips the x coordinate so that sight lines are viewed as from inside the eye.
        tol: float, optional (default=1e-8)
            Passed to mollweide_projection.
        niter: int, optional (default=1000)
            Passed to mollweide_projection.
        """
        xyz = self.xyz_u
        if distance is not None:
            xyz = self.xyz_l + distance * xyz
            xyz = xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)
        xy_mol = mollweide_projection(*xyz.T, tol=tol, niter=niter)
        if flip:  # projection from inside sphere
            return xy_mol * (-1, 1)
        return xy_mol
