from pathlib import Path

from connquer.connectome import Connectome


class Neuprint:

    from .connectome.clients.neuprintClient import NeuprintClient
    from .connectome.authentication import NEUPRINT_TOKEN

    @staticmethod
    def male_adult_optic_lobe() -> Connectome:
        return type("Connectome", (Neuprint.NeuprintClient,), {})(
            "Janelia: MaleOpticLobe", token=Neuprint.NEUPRINT_TOKEN(), dataset='optic-lobe:v1.1'
        )

    @staticmethod
    def male_cns() -> Connectome:
        return type("Connectome", (Neuprint.NeuprintClient,), {})(
            "Janelia: MaleCNS", token=Neuprint.NEUPRINT_TOKEN(), dataset='male-cns:v1.0'
        )


class Flywire:

    @staticmethod
    def local(path=None, synapse_version="buhmann") -> Connectome:
        """Returns a FlyWire dataset.

        Parameters
        ----------
        path : str
            Path to local directory containing flywire data files.
        synapse_version : str {'princeton', 'buhmann'}
            Which synapse version to use.
        """
        from .connectome.clients.localClient import FlyWireLocal
        if path is None:
            from .connectome.authentication import FLYWIRE_LOCAL
            path = FLYWIRE_LOCAL()
        path = Path(path)
        kw = {"connections": f"connections_{synapse_version}_no_threshold"}
        if "column_assignment.csv.gz" in map(lambda x: x.name, path.glob("*.csv.gz")):
            kw["visual_columns"] = "column_assignment"
        return type("Connectome", (FlyWireLocal,), {})(path, **kw)

    @staticmethod
    def cave_client():
        from .connectome.clients.caveClient import CaveClient
        from .connectome.authentication import CAVE_TOKEN
        return type("Connectome", (CaveClient,), {})(
            "flywire-fafb_cave", token=CAVE_TOKEN(), dataset='flywire_fafb_public'
        )
