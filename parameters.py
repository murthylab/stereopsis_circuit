from connquer.datasets import Connectome, Flywire, Neuprint
from connquer.visual_system import VisualColumnMap, HexelMapper
from pathlib import Path
import json
import pandas as pd
import typing


class Dataset:

    THRESHOLD = 3

    _directory = Path()

    _visual_column_directory = _directory.joinpath("visual_column_assignment")
    _cell_type_matching_directory = _directory.joinpath("cell_type_matching")
    _analysis_directory = _directory.joinpath("analysis")

    optic_lobe_neuropils = ("LA", "ME", "LO", "LOP", "AME")

    def __init__(
            self,
            name: str,
            data: Connectome,
            visual_column_maps: typing.Mapping[str, VisualColumnMap],
            cell_type_map: typing.Mapping[str, typing.List[str]],
            neuropil_formatter: typing.Callable[[str, str], str]
    ):
        self.name = name
        self.data = data
        self.visual_column_maps = visual_column_maps
        self.cell_type_map = cell_type_map
        self.neuropil_formatter = neuropil_formatter

    @classmethod
    def read_visual_columns(cls, filename, as_dataframe=False):
        path = cls._visual_column_directory.joinpath(filename).with_suffix(".csv")
        df = pd.read_csv(path)
        if as_dataframe:
            return df
        return {side: VisualColumnMap(df.query("side == @side")) for side in ["right", "left"]}

    @classmethod
    def read_cell_type_map(cls, filename):
        path = cls._cell_type_matching_directory.joinpath(filename).with_suffix(".json")
        with open(path, "r") as f:
            cell_type_map = json.load(f)
        return cell_type_map

    @classmethod
    def flywire(cls, synapses="princeton"):
        data = Flywire.local(synapse_version=synapses)
        visual_columns = cls.read_visual_columns("flywire")
        cell_types = {ct: [ct] for ct in data.cell_types}
        return cls("flywire",
            data,
            visual_columns,
            cell_types,
            lambda region, side: f"{region}_{side[0].upper()}"
        )

    @classmethod
    def male_cns(cls):
        data = Neuprint.male_cns()
        visual_columns = cls.read_visual_columns("male_cns")
        cell_types = cls.read_cell_type_map("flywire_to_mcns")
        return cls("male_cns",
            data,
            visual_columns,
            cell_types,
            lambda region, side: f"{region}({side[0].upper()})"
        )

    @property
    def directory(self):
        directory = self._analysis_directory / self.name
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @property
    def nt_colormapper(self, default="0.5"):
        return NeurotransmitterColorMapper(self.data, default)


DATASETS = {
    "flywire": Dataset.flywire(),
    "male_cns": Dataset.male_cns(),
}

contralateral = dict(left="right", right="left")
hexmap = HexelMapper()

class NeurotransmitterColorMapper:

    nt_colors = {
        "ACH": "#93a1cd",
        "GABA": "#e8dc6c",
        "GLUT": "#86a757"
    }

    def __init__(self, dataset, default="0.5"):
        self.dataset = dataset
        self.default = default

    def __call__(self, cell_type, default=None):
        default = default or self.default
        nt = self.dataset.get_neurotransmitter(cell_type)
        return self.nt_colors.get(nt, default)


cell_type_colors = {

    "LC14a1": "#E81E25",
    "LC14b": "#F7941D",
    "LC14a2": "#f9ed32",
    "LC10a": "#2B3990",
    "LC43": "#315972",
    "LC31a": "#f27292",
    "LC31b": "#f27292",

    "LMa1": "#b2d235",
    "Tm5f": "#58595b",
    "T2": "#d1d3d4",
    "LC9": "#92d7e7",
    "LC10a(bino)": "#2B3990",
    "LC10a(mono)": "#8789be",

    "AOTU016(bino)": "#92278f",
    "AOTU016(mono)": "#bd8cbf",
    "AOTU002(bino)": "#da1c5c",
    "AOTU002(mono)": "#e9959d"

}
