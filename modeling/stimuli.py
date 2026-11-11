from parameters import DATASETS, hexmap
import h5py
import numpy as np
from eye_model_3d import MappedDataset
from eye_model_3d.geometry import rotation_3d


class StimulusBuilder:
    """Linear dynamical system model.

    Parameters
    ----------
    azimuths: np.ndarray
        Azimuthal angles (radians).
    """

    def __init__(self, azimuths):
        self.azimuths = azimuths

    @property
    def arc(self):
        return np.column_stack((np.cos(self.azimuths), np.sin(self.azimuths), np.zeros_like(self.azimuths)))

    def __call__(self, phi):
        Ry = rotation_3d(phi, 1)
        points = np.einsum("ij...,nj->...ni", Ry, self.arc)
        return points


if __name__ == "__main__":

    # Load dataset
    dset = DATASETS["flywire"]
    directory = dset.directory / "modeling"
    directory.mkdir(parents=True, exist_ok=True)

    # Generate stimuli
    thetas = np.radians(np.arange(-30, 31, 1))
    phis = np.radians(np.arange(-90, 91, 5))
    distances = np.arange(500, 20_001, 500)
    radii = np.arange(100, 3001, 100)
    stimulus_builder = StimulusBuilder(thetas)
    stimulus = stimulus_builder(-phis)  # flip +/- angles so + is up
    stimulus = np.einsum("nmi,k->nmki", stimulus, distances)

    # Load eye model
    eye_model = MappedDataset.load("flywire", convergence=np.radians(5))

    # Compute column activations
    Ul = eye_model["left"].column_activation(stimulus, radii, newaxis=True)
    Ur = eye_model["right"].column_activation(stimulus, radii, newaxis=True)
    U = np.concat([Ul, Ur], axis=-1)
    U = np.moveaxis(U, -1, 0)  # columns x phis x thetas x distances x radii

    # Save
    with h5py.File(directory / "stimuli.h5", "w") as f:
        f.attrs["axes"] = ["columns", "phis", "thetas", "distances", "radii"]
        grp = f.create_group("axes")
        grp.create_dataset("phis", data=phis)
        grp.create_dataset("thetas", data=thetas)
        grp.create_dataset("distances", data=distances)
        grp.create_dataset("radii", data=radii)
        f.create_dataset("constant_radius", data=U)

    del Ul, Ur, U

    # Constant angle
    angle = 30.
    omega = np.radians(angle / 2)
    radii = distances * np.sin(omega)

    # Compute column activations
    Ul = eye_model["left"].column_activation(stimulus, radii, newaxis=False)
    Ur = eye_model["right"].column_activation(stimulus, radii, newaxis=False)
    U = np.concat([Ul, Ur], axis=-1)
    U = np.moveaxis(U, -1, 0)

    # Save
    with h5py.File(directory / "stimuli.h5", "r+") as f:
        d = f.create_dataset("constant_angle", data=U)
        d.attrs["stimulus_size"] = angle
